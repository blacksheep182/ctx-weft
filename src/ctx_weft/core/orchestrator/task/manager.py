"""TaskManager：监听事件 + 调度 TaskQueue + parent resume 逻辑。

Phase 4 §4.2 + §4.7。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ctx_weft.core.models.discriminators import CancelReason, InterruptReason, TaskErrorCode
from ctx_weft.core.models.status import PARKED_TASK_STATUSES, TERMINAL_TASK_STATUSES
from ctx_weft.core.utils.event import emit_event
from ctx_weft.core.orchestrator.task.failure_threshold import plan_threshold_trip
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.models.errors import crash_error_code, crash_run_outcome
from ctx_weft.core.loop.park import RoundDiscarded
from ctx_weft.core.orchestrator.task.disposition import (
    Disposition,
    RunOutcome,
    RunOutcomeKind,
    disposition_for,
)
from ctx_weft.core.orchestrator.task.queue import QueueEntry, TaskQueue
from ctx_weft.core.utils.task_ref import task_ref, task_ref_parts
from ctx_weft.core.utils.verdict import VERDICT_SUCCESS, normalize_verdict
from ctx_weft.core.orchestrator.task.runner import AgentBinding, TaskRunner, effective_agent_id
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.status import TaskStatus
from ctx_weft.core.models.task import PORT_MAIN, CompactTaskSettings, MetadataFillerTaskSettings, NormalTaskSettings, Task
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.events import EventOrigin, EventType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ctx_weft.protocols.events import EventBus

logger = logging.getLogger(__name__)

_ORIGIN = EventOrigin.ORCHESTRATOR_TASK_MANAGER

#: 词表住在 `core.domain.status`（会话/task/agent 三套并排，见该模块 docstring）。
#: 这里的两个别名只为不改动本文件里的既有引用点。
_PARKED_STATUSES = PARKED_TASK_STATUSES

#: 这条守卫此前长在 `_run_loop` 的三个 except 支里，随发射一起搬来。
_TERMINAL_STATUSES = TERMINAL_TASK_STATUSES

# 默认值；实际值由 host 经 RuntimeConfig → TaskManager 构造参数注入。
_DEFAULT_MAX_RETRIES    = 3
#: 并发池的基准上限（交互 task 可临时顶上去，见 `drain` 的 docstring 第 3 条）。
_DEFAULT_MAX_CONCURRENT = 4
_DEFAULT_AUTONOMOUS_REQUEUE_MAX          = 3
_DEFAULT_AUTONOMOUS_REQUEUE_BACKOFF_BASE = 30.0


class TaskManager:
    """Manages a per-session TaskQueue and drives task execution.

    The caller (SessionRegistry / CtxWeftRuntime) must:
    1. Register a task_runner callback (runs a single task).
    2. Call push_task() to add tasks.
    3. Call tick() or rely on event-driven draining.
    """

    def __init__(
        self,
        session_id: str,
        max_concurrent: int | None = None,
        task_max_retries: int | None = None,
        event_bus: "EventBus | None" = None,
        autonomous_requeue_max: int | None = None,
        autonomous_requeue_backoff_base_sec: float | None = None,
    ) -> None:
        self._session_id = session_id
        self._max_concurrent = max_concurrent if max_concurrent is not None else _DEFAULT_MAX_CONCURRENT
        self._task_max_retries = task_max_retries if task_max_retries is not None else _DEFAULT_MAX_RETRIES
        # 自治作业的 INTERRUPTED 退避重排预算（见 `_schedule_autonomous_requeue`）。
        self._autonomous_requeue_max = (
            autonomous_requeue_max if autonomous_requeue_max is not None
            else _DEFAULT_AUTONOMOUS_REQUEUE_MAX
        )
        self._autonomous_requeue_backoff_base_sec = (
            autonomous_requeue_backoff_base_sec if autonomous_requeue_backoff_base_sec is not None
            else _DEFAULT_AUTONOMOUS_REQUEUE_BACKOFF_BASE
        )
        #: task_id → 在飞的退避定时器。**刻意不进 `_background_asyncio_tasks`**：
        #: 那个集合是 `_fire_session_done` 要 gather 的，把一个最长 120s 的定时器放进去
        #: 等于让会话收尾等它。取消由 `cancel_all` 显式做，协程自己醒来后也三重自检。
        self._autonomous_requeue_timers: dict[str, asyncio.Task] = {}
        self._queue: TaskQueue = TaskQueue()
        self._cancelled: bool = False
        self._tasks: dict[str, Task] = {}
        # 同一轮（一次 task run）内 delegate_task / delegate_plan 先投这里，
        # runner 正常返回后由 _flush_staged 统一入队，实现「同批次 FIFO」。
        # key = 正在运行的 task_id（即被 spawn 子任务的 parent_task_id）。
        self._staged: dict[str, list[tuple[Task, list[str] | None, str | None]]] = {}
        self._parent_map: dict[str, str] = {}  # child_task_id → parent_task_id
        self._children_of: dict[str, set[str]] = {}  # parent_task_id → set[child_task_ids]
        self._runner: TaskRunner | None = None
        self._running_tasks: set[str] = set()
        # 派发后登记的「真实执行 agent id」（task_id → binding.agent_id）——
        # 同 agent 串行判定对在跑任务用真值，只有队列候选才走 effective_agent_id 预测。
        self._running_agents: dict[str, str] = {}
        #: task_id → 这一轮的开窗快照（spec 2026-09-09）。键存在即「窗口开着」：
        #: 这个 task 的事件只到达进程内状态机，不落盘、不到 host，直到 act 收到本轮
        #: 第一个 chunk 才提交。详见 `EventBus` 的类 docstring 与 `begin_round`。
        self._rounds: dict[str, RoundSnapshot] = {}
        self._lock = asyncio.Lock()
        self._session: Session | None = None  # 注入后供 failure_counter 维护使用
        self._event_bus: "EventBus | None" = event_bus
        self._background_asyncio_tasks: set[asyncio.Task] = set()
        #: 一次性接线的 7 个回调（见 hooks.py）。整体替换，不逐字段合并。
        self._hooks = TaskManagerHooks()
        # spec: event-commit：会话存储健康检查（runtime 注入；None = 无健康面）。
        # drain 据此跳过隔离会话的派发——存储不可用时不再开新副作用。
        self._unhealthy_check = None
        # 归属权谓词：runtime 注入，返回本 TM 是否仍是该 session 的当前 owner。
        # None = 不受管（永远视为 current，保持旧行为）。session 的 TM 是单例，活 owner
        # 不会被顶替；只有被逐出（forget/purge）之后才返回 False → 迟到的收尾变 no-op
        # （不发 SessionFinished、不清之后新建的 TM 的控制信号）。
        # pause 弃子窗口标记（runtime.pause_session 置位、_on_idle/_release 复位）：
        # 置位期间任务取消不改 session 状态、run 收尾 staged 直接丢弃。
        self._pause_abandon = False
        # ── 熔断真终结（failure threshold trip）状态 ──────────────────────────────
        # 幂等闩：trip 后在途任务再失败会重进 FAILED 分支，没有它会重复清场 + 重复发事件。
        self._threshold_tripped: bool = False
        # spec: task-handoff——永久阻塞扫描的重入闩：处置里的 on_task_finished 会再进
        # 扫描入口，闩住后由外层不动点循环统一覆盖。
        self._in_blocked_dispose: bool = False
        # (title, reason) 随 failure_counter 同步积累（FAILED 追加、FINISHED 清空）；
        # 供 FAILURE_THRESHOLD_HIT payload 与 threshold_finalizer 引用。跨崩溃恢复不重建，接受。
        self._recent_failures: list[tuple[str, str]] = []
        # 三个 trip 序列的注入点（接线方式镜像 set_is_current）：None = 该副作用跳过，
        # trip 序列本身永远不因缺注入而崩溃。runtime 侧实现见 Task 10。
        # 统一取消胶囊闭合（Task 14）：cancel_all / 熔断清场（已启动挂起排队） / 在途协作取消 funnel
        # 三处调用点共用同一注入点。None-tolerant：缺注入时三处调用点自身各自跳过、不崩溃。
        # 事件侧 blob store 的注入点（set_event_blob_store / _event_blob_store /
        # _event_ctx）已删除：TASK_CREATED 自 Task 3 起、TASK_REQUEUED 自本任务
        # （blob-store 解耦 Task 5）起都改由调用方/task 上携带的现成 event jsonable
        # 供给，TaskManager 不再需要持有 event blob store——「本类不发起 blob 调用」
        # 是**结构性**保证：它根本没有能力发起一次。

    def track_background(self, t: "asyncio.Task") -> None:
        """Track a fire-and-forget background coroutine so the session awaits it before close."""
        self._background_asyncio_tasks.add(t)
        t.add_done_callback(self._background_asyncio_tasks.discard)

    def set_unhealthy_check(self, check) -> None:
        """注入会话健康检查（spec: event-commit）。drain 跳过隔离会话的派发。"""
        self._unhealthy_check = check

    def set_runner(self, runner: TaskRunner) -> None:
        self._runner = runner

    def set_hooks(self, hooks: TaskManagerHooks) -> None:
        """一次性装好全部回调。**整体替换**，不逐字段合并（见 TaskManagerHooks）。"""
        self._hooks = hooks

    def set_session(self, session: Session) -> None:
        """注入 Session 对象，供 failure_counter 维护使用。"""
        self._session = session

    @property
    def session(self) -> "Session | None":
        """注入的 Session 对象（复用路径据它读/写本轮 llm 参数——model=会话状态）。"""
        return self._session

    def register_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    def restore(
        self,
        all_tasks: list[Task],
        terminal_ids: set[str],
        parked_task_ids: set[str] | None = None,
        *,
        enqueue: bool = True,
    ) -> None:
        """Rebuild task registry and re-queue resumable tasks after a crash.

        ``parked_task_ids``: tasks with an unresolved pending HITL (spec/07 §9.1) —
        SUSPENDED but must STAY parked (not re-queued) until ``/answer`` resumes them.
        Empty/None → legacy behavior.

        ``enqueue=False`` loads the registry and dependency/child indexes only.
        Persisted statuses remain unchanged; the caller later requeue_resumable()
        after delivering input. No tasks can run during preparation.

        挂在**已终局** HITL 上的 task 不需要在这里另开一条重排口子：ACTIVE 的走下面的
        ``else`` 分支、SUSPENDED 且子任务都终态的走 children 闸门、SUSPENDED 且尚有活
        子任务的由 ``_try_resume_parent`` 在子任务收尾时唤醒（那些子任务本身也被这一趟
        restore 重排了）。**反而不能**在这里把它提前置 PENDING——`_try_resume_parent`
        以 ``status == "SUSPENDED"`` 为门，提前改状态会把那次合法的唤醒静默吞掉，父任务
        就只跑了「早的那一次」、拿不到子任务的产出（Task 9 复审 Finding 1）。

        compact / recognize_intent are no longer scheduled as Tasks; any such obsolete
        task found in a replayed event stream is skipped (never re-queued). Recovery is
        condition-based.
        """
        parked = parked_task_ids or set()

        for t in all_tasks:
            self._tasks[t.id] = t
            if t.parent_task_id:
                self._parent_map[t.id] = t.parent_task_id
                self._children_of.setdefault(t.parent_task_id, set()).add(t.id)

        # spec: task-handoff——只有 FINISHED 释放后继。终态但未成功的前序留在各条目的
        # blocked_by 里，交给恢复期的永久阻塞扫描处置（A 已 FAILED 落盘、B 的级联取消
        # 未落盘的崩溃窗口正是靠那一趟兜底）。
        succeeded_ids = {t.id for t in all_tasks if t.status == "FINISHED"}
        self._queue.seed_succeeded(succeeded_ids)
        if not enqueue:
            return

        for t in all_tasks:
            if t.status in TERMINAL_TASK_STATUSES:
                continue
            if isinstance(t.settings, (CompactTaskSettings, MetadataFillerTaskSettings)):
                continue  # obsolete ephemeral helpers — never re-scheduled (recovery is condition-based)
            # HITL-park：有未决 HITL → 保持挂起、不入队，**不论 ACTIVE 还是 SUSPENDED**。
            # 审批热等的任务恒为 ACTIVE（不会走 SUSPENDED 分支）；park 判据是"有无未决 HITL"
            # （parked_task_ids，源自 HitlRegistry.list_pending），而非 task.status（spec/07 §9.1，缺陷 A）。
            if t.id in parked:
                continue
            if t.status == "SUSPENDED":
                children = self._children_of.get(t.id, set())
                if all(cid in terminal_ids for cid in children):
                    t.status = "PENDING"
                    t.retry_count = 0  # 崩溃挂起带着耗尽的计数；恢复重跑从零重计
                    self._queue.push(QueueEntry(
                        task_id=t.id, session_id=self._session_id, priority=t.priority,
                    ))
            else:
                t.status = "PENDING"
                t.retry_count = 0
                blocked = {dep for dep in (t.dag_deps or []) if dep not in succeeded_ids}
                self._queue.push(QueueEntry(
                    task_id=t.id, session_id=self._session_id,
                    priority=t.priority, blocked_by=blocked,
                ))

    async def push_task(
        self,
        task: Task,
        blocked_by: list[str] | None = None,
        parent_task_id: str | None = None,
        *,
        user_prompt_event_jsonable: "str | list[dict] | None" = None,
        provisional: bool = False,
    ) -> None:
        """入队一个新任务并发 TASK_CREATED。

        ``provisional=True``：这个 task 由一条**用户消息**开出（`send_message` 的新建
        分支），在 LLM 真的开口之前它不算发生——开一道未提交窗口（`begin_round`，
        `owns_task=True`），本方法发出的 `TASK_CREATED` 连同随后的 `TASK_STARTED` /
        `RUN_STARTED` / `LLM_PROMPT_SENT` 一并只到达进程内状态机，不落盘、不到 host。
        act 收到本轮第一个 chunk 时 `commit_round` 补投，用户在那之前按暂停则
        `discard_round` 整轮丢弃。
        委派子任务（`_flush_staged`）**不走这条**：它们不是一轮对话的开端，且
        「创建即落盘」对它们仍然必要（见下方注释）。

        ``user_prompt_event_jsonable``：TASK_CREATED 里 user_prompt 的 event 侧载荷，
        由**入口**（`SessionRegistry.create_session` ← `CtxWeftRuntime.start_session`）
        从**归一化之前的原始** content 算好传进来。本方法不自己算：`task.user_prompt`
        到这里已是 memory 侧归一化过的内容，图片 part 是 memory ref，再算一次只会把
        一个 event store 打不开的引用写进事件（blob-store 解耦 Task 3）。

        不传（agent 派发的子任务经 `stage_task` → `_flush_staged` 走这条）时退回
        ``task.user_prompt`` 本身：那条路径的 prompt 恒是工具参数里的纯文本
        （见 `control_capability` 的 delegate_task / plan_tasks），jsonable 形态就是
        它自己，没有字节、也无处取原始字节。真出现 part 列表则**响亮拒绝**——静默
        塞进 payload 会直接击穿「事件库恒不含字节」。
        """
        # 不变式 `unattended=False ⟹ port_key 非空`（见 `Task.port_key`）：要等对端，
        # 就得先有个口。校验只落在这一条**新建**路径上——retry / resume / 恢复重排走
        # `_queue.push`，`restore` 重建的是既有事实，在那些地方拦只会让恢复在坏数据上
        # 炸。排在任何副作用之前，与下方 user_prompt 那条同一口径（入口即拒、不落库）。
        if not task.unattended and not task.port_key:
            raise ValueError(
                f"push_task({task.id}): unattended=False 的 task 必须接在某个交互口上"
                f"（port_key 非空）——要等对端应答，就得先有个口把应答路由回来。"
                f"自治作业请一并标 unattended=True。"
            )
        # 统一用 TaskManager 级别的 max_retries，覆盖 Task 模型的硬编码默认值
        task.max_retries = self._task_max_retries
        # 开窗**必须先于**下面那条 TASK_CREATED——晚一步它就已经落盘了。
        if provisional:
            self.begin_round(task.id, owns_task=True)
        self._tasks[task.id] = task
        if parent_task_id:
            self._parent_map[task.id] = parent_task_id
            self._children_of.setdefault(parent_task_id, set()).add(task.id)
        if blocked_by:
            task.dag_deps = list(blocked_by)

        entry = QueueEntry(
            task_id=task.id,
            session_id=self._session_id,
            priority=task.priority,
            blocked_by=set(task.dag_deps or []),
        )
        self._queue.push(entry)
        logger.debug("TaskManager.push_task: %s blocked_by=%s", task.id, entry.blocked_by)
        # 创建即落盘：未跑过的排队 / blocked task 也进事件日志，
        # 崩溃恢复时 restore() 可由 dag_deps 重建依赖链，无需父任务重新 spawn。
        # push_task 是唯一的「新建」路径（retry / resume / active 重排都走 _queue.push），
        # 故此处恰好 emit 一次。
        user_prompt_jsonable = user_prompt_event_jsonable
        if user_prompt_jsonable is None:
            if isinstance(task.user_prompt, list):
                raise ValueError(
                    f"push_task({task.id}): 非纯文本 user_prompt 必须由调用方传 "
                    "user_prompt_event_jsonable（由归一化之前的原始 content 算出）——"
                    "事件库恒不含字节，这里没有原始字节可用"
                )
            user_prompt_jsonable = task.user_prompt   # str | None，本身即 jsonable
        # 挂在 task 上（见 Task.user_prompt_event_jsonable 的字段注释）。
        task.user_prompt_event_jsonable = user_prompt_jsonable
        await self._emit(
            EventType.TASK_CREATED, task_id=task.id,
            payload=task_payload(task, user_prompt_jsonable=user_prompt_jsonable),
        )

    # ── 未提交窗口（spec 2026-09-09）────────────────────────────────────────
    #
    # 窗口的所有权在 TaskManager：它是 task 生命周期的主人，「这一轮算不算数」是它的
    # 判断。总线只负责执行分流（谁收得到），act 只负责在正确的时刻喊一声提交或丢弃。

    def begin_round(
        self, task_id: str, *, owns_task: bool, hitl_id: str = "",
    ) -> None:
        """开窗 + 拍这一轮的回滚快照。

        ``owns_task``：这个 task 是不是**这条消息开出来的**。
          · True（`send_message` 的新建分支）—— 丢弃要连 task 一起摘掉，agent 回 `idle`。
          · False（消息注入一个既有 task：HITL 冷续跑 / `/messages` 注入）—— task 早就
            存在、是一段正在进行的对话，丢弃只把它退回开窗前的样子，agent 回
            `waiting_human`。

        ``hitl_id``：被这条消息收口的那个旧气泡（有的话）。丢弃时经 `revert_round`
        钩子 `release` 回 pending —— 那正是「撤销前的世界」。

        幂等：重复开窗不覆盖已拍的快照、也不清空已攒的缓冲（retry 重排会重进同一条路径）。
        """
        if not task_id or task_id in self._rounds:
            return
        task = self._tasks.get(task_id)
        self._rounds[task_id] = RoundSnapshot(
            owns_task=owns_task,
            hitl_id=hitl_id,
            task_status=getattr(task, "status", ""),
            retry_count=getattr(task, "retry_count", 0),
            outputs=getattr(task, "outputs", None),
            process_report=getattr(task, "process_report", None),
            process_report_at=getattr(task, "process_report_at", None),
            user_prompt_in_memory=getattr(task, "user_prompt_in_memory", False),
        )
        bus = self._event_bus
        if bus is not None:
            bus.begin_provisional(task_id)

    def stage_memory(self, task_id: str, memory: Any, event: Any, provider_ctx: Any) -> bool:
        """这一轮开着窗 → 把一次 memory 写入暂存进窗口，返回 True；没开窗 → False。

        与事件缓冲同一个道理、同一个生命周期：窗口里发生的事在这一轮算数之前都不算发生。
        暂存的写入由 `commit_round` 交给钩子按序落盘，丢弃时随快照一起扔掉——memory 里
        从头到尾不出现一条「还没算数」的记录，撤销也就不需要任何补偿写（fold）。

        TM 不碰 memory：这里只替它保管三件不透明的东西（写到哪、写什么、以谁的身份写），
        真正的 `ingest` 在 runtime 的钩子里做。
        """
        snap = self._rounds.get(task_id) if task_id else None
        if snap is None:
            return False
        if getattr(event, "id", None) is None:
            # 叠加读取（装配）要给它一个稳定的记录 id；落盘时 provider 按给定 id 采用。
            from ctx_weft.core.utils.ids import generate_id
            event.id = generate_id("mem")
        # 同 id 替换而不是并排：一轮提交失败停机后窗口保留，下一次 run 可能重入同一次
        # 调用、再暂存同一条确定性 id 的结果（`res_...`）。memory 对同 id 是幂等的，
        # 暂存区也得是，否则提交前的装配里同一个 tool_call 出现两份结果。
        entry = (memory, event, provider_ctx)
        for i, (_m, staged, _p) in enumerate(snap.staged_memory):
            if staged.id == event.id:
                snap.staged_memory[i] = entry
                return True
        snap.staged_memory.append(entry)
        return True

    def staged_memory(self, task_id: str) -> "list[Any]":
        """这一轮暂存着、还没落盘的 memory 写入（`MemoryEvent` 列表，按写入顺序）。

        读侧叠加用：装配 prompt 时必须看得见它们——那正是「LLM 开口之前」这一轮要用到的
        内容（人刚给的答复、刚发的消息）。
        """
        snap = self._rounds.get(task_id) if task_id else None
        return [ev for _mem, ev, _pctx in snap.staged_memory] if snap is not None else []

    def is_round_open(self, task_id: str) -> bool:
        return task_id in self._rounds

    def any_round_open(self) -> bool:
        """本 TM 名下还有没有开着的未提交窗口。

        供快照写入的前置条件用（`SnapshotWriter._is_safe_to_write`）：窗口开着就意味着有
        暂存的 memory 写入还没落盘，而缓冲里的事件已经**先于**它们被补投给 rest 订阅者。
        那一刻写快照会写出一张领先于 memory 的——恢复「以快照为准」之后那是静默的数据丢失。
        """
        return bool(self._rounds)

    def round_hitl_id(self, task_id: str) -> str:
        """这一轮收口了哪个旧气泡（没有则空串）。供 `revert_round` 钩子 release 用。"""
        snap = self._rounds.get(task_id)
        return snap.hitl_id if snap is not None else ""

    @property
    def open_round_task_ids(self) -> "frozenset[str]":
        """当前开着窗口的 task id。

        两个读者：`Runtime._load_agents_of` 用它保住路由判据（这些 task 的
        `TASK_CREATED` 还没落盘，折不进 `AgentView`，热重装若照 view 覆盖
        `current_task_id` 就会把它倒回上一个已终态的 task）；`Runtime.pause_session`
        用它决定要不要跳过排队弃子（见那里）。
        """
        return frozenset(self._rounds)

    async def commit_round(self, task_id: str) -> None:
        """这一轮算数了：按序补投缓冲里的事件，关窗，丢掉快照。

        触发点有两个，**都不在本类**：act 收到本轮第一个 chunk（正常路径），以及
        `_run_task` 的收尾兜底（见那里——失败、outage 耗尽、装配炸掉一律提交，
        只有用户主动中止才丢弃）。

        补投之后调 `commit_round` 钩子：先把这一轮里待终局的 HITL 答复终局
        （`HitlResolved`），再把暂存的 memory 写入按序落盘——**先有终局事实，答复才进
        memory**。每一个提交点都经过这里，所以谁提交都不会漏掉那两步。
        """
        snap = self._rounds.get(task_id)
        if snap is None:
            return
        # **快照在全部成功之后才弹出**。任何一步抛出（典型：存储不可用）都原样上抛、快照
        # 留在原地，下一个提交点从断处重试——曾经一开头就 pop：补投失败时暂存的 memory
        # 写入随快照丢失，待终局的 HITL 既没终局也没退回（进程内再也答不了），总线把窗口
        # 放回去了、TM 却已不认这一轮，此后该 task 的事件全进一个没人关的缓冲。
        #
        # 重试的幂等性：事件补投用 `events_flushed` 标记跳过；钩子里的两步本身幂等——
        # 已终局的 HITL 再 commit 是 no-op，暂存记录都带 id、重复 ingest 是 no-op。
        if not snap.events_flushed:
            # `ROUND_COMMITTED` **先于**补投，且**不带 task_id**（那道闸按 task_id 定，带上
            # 就会被自己挡住）。顺序是给 host 看的：它据此把攒着的用户消息帧 flush 出去，
            # 那一帧必须排在这一轮的 task/run 帧**之前**——用户先说话，agent 才开跑。
            await self._emit(
                EventType.ROUND_COMMITTED, task_id=None, payload={"task_id": task_id},
            )
            bus = self._event_bus
            if bus is not None:
                await bus.commit_provisional(task_id)
            snap.events_flushed = True
        # **排在补投之后**：`HitlResolved` 不能落在一个下游还没建键的 task 上（窗口里
        # 攒着的 `TASK_*` / `RUN_STARTED` 得先出去）。**不吞异常**：钩子失败意味着终局
        # 事实或 memory 没落定，这一轮不算提交完。
        if self._hooks.commit_round is not None:
            await self._hooks.commit_round(task_id, list(snap.staged_memory))
        self._rounds.pop(task_id, None)

    async def drop_round(self, task_id: str) -> None:
        """撤掉一扇**什么都还没收下**的窗：调用方开了窗，随后那件事没成（应答校验失败 /
        幂等 no-op）。

        与 `abandon_round` / `discard_round` 的分界：那两个撤的是「收下过东西的一轮」，要经
        `revert_round` 钩子退回答复、清暂停闩锁；这里窗里什么都没有，调钩子反而会误清别人
        的闩锁。只做三件事：弹快照、丢缓冲、发 `ROUND_DISCARDED`（host 据此丢掉为这次应答
        攒的消息帧；它若已自行回滚，那条帧是 no-op）。
        """
        if self._rounds.pop(task_id, None) is None:
            return
        bus = self._event_bus
        if bus is not None:
            bus.discard_provisional(task_id)
        await self._emit(
            EventType.ROUND_DISCARDED, task_id=None,
            payload={"task_id": task_id, "reason": "nothing_accepted"},
        )

    def drop_round_buffer(self, task_id: str) -> None:
        """**同步**丢掉一扇窗的两半（快照 + 总线缓冲），不发任何事件、不调任何钩子。

        `drop_round` 的最小版本，只给逐出路径用（`Runtime._evict_session_memory`）：那里
        这条会话的运行时内存正在被整体摘掉，发事件既没有意义也没有消费者。正常路径不该
        走到这里——`cancel_session` 会先把窗收干净，`forget_session` 的准入判据也不放行。
        """
        if self._rounds.pop(task_id, None) is None:
            return
        bus = self._event_bus
        if bus is not None:
            bus.discard_provisional(task_id)

    async def abandon_round(self, task_id: str) -> None:
        """只撤「这一轮」本身，**不动 task 与 run**：热应答被暂停时用（spec 2026-09-09）。

        与 `discard_round` 的分界：那个是整轮连 run 一起当没发生过（run 自己也在窗口里
        出生）；这个的 run 早在窗口之前就开跑了，它还要照常 park、照常发结局事件——
        所以这里不还原快照、不发 task 状态事件、不动队列与登记，只做三件事：

        1. `revert_round` 钩子：答复退回 pending、清暂停闩锁。
        2. 关窗，丢掉缓冲的事件（那条答复带出来的 `CapabilityFinished` 之类）与暂存的
           memory 写入（回灌出来的工具结果）——memory 里本来就没有它们，无需补偿。
        3. `ROUND_DISCARDED`（不带 task_id）：host 据此丢掉攒着的用户消息帧。

        之后调用方抛 `HitlPark`，task 的状态由 `apply_run_outcome` 照常落盘。
        """
        if task_id not in self._rounds:
            return
        if self._hooks.revert_round is not None:
            try:
                await self._hooks.revert_round(task_id)
            except Exception:
                logger.exception(
                    "abandon_round: revert_round hook failed for task %s", task_id)
        self._rounds.pop(task_id, None)
        bus = self._event_bus
        if bus is not None:
            bus.discard_provisional(task_id)
        await self._emit(
            EventType.ROUND_DISCARDED, task_id=None,
            payload={"task_id": task_id, "reason": "discarded_before_first_chunk"},
        )

    async def discard_round(self, task_id: str) -> None:
        """这一轮当作没发生过。**只用于用户主动中止**（首 chunk 之前按暂停 / 取消）。

        五步，顺序就是这段代码的全部要点——**每一步都必须发生在关窗之前**，窗口一关，
        此后发的任何事件都直接落盘：

        1. `revert_round` 钩子：被收口的旧气泡 `release` 回 pending（HITL 在 orchestrator
           之下，TM 够不到，见 hooks.py）。这一轮的用户消息 / 答复只在快照的暂存区里，
           随第 5 步关窗一起扔掉，memory 从来没见过它们。
        2. 还原快照：`status` / `retry_count` / `outputs` / `process_report*`。后三样
           是 `_inject_user_turn` 就地清掉的，没有别处存过旧值。`user_prompt_in_memory`
           同理：暂存的提问被扔掉了，标志要回到开窗之前。

           **`title` / `description` 不在其列**：唯一会就地改它们的是
           `update_task_metadata`（`recognize_intent` 调），而旁路的起飞点在
           `act._commit_round` —— 那时窗口已经关了，它跑在窗口之外，改不到窗口里的状态。
        3. 发一条把 agent 拨回原位的 task 事件——**这是关键的一步**，且必须先于第 4 步
           （事件的 agent_id 从 `_running_agents` 反查，先清就发不出去了）：
           · owns_task → `TASK_CANCELED`（ALM 映射成 `SETTLED` → `idle`）
           · 否则     → `TASK_AWAITING_HUMAN{hitl_id}`（ALM → `waiting_human`）
           两者都是 ALM 本来就有的转移，不需要给它引入任何可回滚状态；而它们自己也在
           窗口里，随缓冲一起被丢掉 —— **内存状态正确，日志干净**。
        4. 队列与登记复位：新建的连 task 一起摘；既有的把重排塞进去的那个条目撤掉。
        5. 关窗丢缓冲。
        """
        snap = self._rounds.get(task_id)
        if snap is None:
            return

        # ① TM 够不到的那部分
        if self._hooks.revert_round is not None:
            try:
                await self._hooks.revert_round(task_id)
            except Exception:
                logger.exception(
                    "discard_round: revert_round hook failed for task %s "
                    "(继续丢弃——一次撤销失败不该把暂停变成 run 崩溃)", task_id)

        task = self._tasks.get(task_id)
        if task is not None and not snap.owns_task:
            # ② 还原被这条消息就地改掉的字段
            task.status = snap.task_status
            task.retry_count = snap.retry_count
            task.outputs = snap.outputs
            task.process_report = snap.process_report
            task.process_report_at = snap.process_report_at
            # 这一轮暂存的若是 task 自己的提问（`_persist_user_prompt`），它随快照一起被扔掉了，
            # 落库标志得跟着回落，否则下一轮以为写过了、不再写。
            task.user_prompt_in_memory = snap.user_prompt_in_memory

        # ③ 把 agent 拨回原位（事件在窗口里，只到 ALM，不落盘）。
        #
        #    **必须先于下面的清理**：`_emit` 的 agent_id 是从 `_running_agents` /
        #    `Task.assigned_agent_id` 反查的，先清后发会发出一条没有 agent_id 的事件，
        #    而 ALM 的 `handle_event` 对没有 agent_id 的事件直接返回——agent 就永远停在
        #    `running` 了。（这条踩过一次，别再把顺序调回去。）
        if snap.owns_task:
            await self._emit(
                EventType.TASK_CANCELED, task_id=task_id,
                payload={"reason": "discarded_before_first_chunk"},
            )
        else:
            await self._emit(
                EventType.TASK_AWAITING_HUMAN, task_id=task_id,
                payload={"hitl_id": snap.hitl_id,
                         "reason": "discarded_before_first_chunk"},
            )

        # ④ 队列与登记复位。用 `cancel` + `unmark_running` 而**不是** `mark_complete`：
        #    后者会把 task_id 塞进 `_succeeded`（给依赖解阻塞用的「已成功」集合）。
        #    一轮没发生过的 task 不该出现在那里。
        self._queue.cancel(task_id)
        self._queue.unmark_running(task_id)
        self._running_agents.pop(task_id, None)
        self._running_tasks.discard(task_id)
        self._staged.pop(task_id, None)
        if snap.owns_task:
            self._tasks.pop(task_id, None)

        # ⑤ 关窗——必须最后
        self._rounds.pop(task_id, None)
        bus = self._event_bus
        if bus is not None:
            bus.discard_provisional(task_id)

        # 关窗之后才发这条：它**不带 task_id**（绕开那道闸），host 据此把攒着的用户
        # 消息帧丢掉、把文本退回输入框当草稿。放在最后是为了「host 看到它的时候，这一轮
        # 已经确定不会再有任何事件出来了」。
        await self._emit(
            EventType.ROUND_DISCARDED, task_id=None,
            payload={"task_id": task_id, "reason": "discarded_before_first_chunk"},
        )

    def stage_task(
        self,
        task: Task,
        blocked_by: list[str] | None = None,
        parent_task_id: str | None = None,
    ) -> None:
        """把子任务投入当前 run 的缓冲区，延迟到 _flush_staged 才真正入队。

        与 push_task 同签名；调度顺序问题（同批次 FIFO）由 _flush_staged 统一处理。
        缓冲按 parent_task_id 分桶，避免 max_concurrent>1 时并发 run 的 spawn 串台。
        """
        key = parent_task_id or ""
        self._staged.setdefault(key, []).append((task, blocked_by, parent_task_id))
        logger.debug("TaskManager.stage_task: %s under parent=%s", task.id, key)

    def detach_staged(self, from_parent_id: str, to_parent_id: str | None) -> None:
        """把 from_parent_id 本轮 staged 的子任务改投到 to_parent_id 名下（独立后续）。

        用于 finish_task 与 delegate_* 同批出现时：当前 task 收尾，被派发任务
        不再做它的阻塞子任务，而改挂到它的 parent（当前是 root 则为 None→顶层）独立调度。

        只改写 tuple 的 parent_task_id（驱动 _flush_staged 时 push_task 的归属）与
        task.parent_task_id，**不搬桶**——桶 key 仍是当前运行 task 的 id，_flush_staged
        才能在本 run 结束时正常弹桶入队。plan 内部兄弟间的 blocked_by 顺序链原样保留。
        """
        bucket = self._staged.get(from_parent_id)
        if not bucket:
            return
        rewritten: list[tuple[Task, list[str] | None, str | None]] = []
        for task, blocked_by, _ in bucket:
            task.parent_task_id = to_parent_id
            rewritten.append((task, blocked_by, to_parent_id))
        self._staged[from_parent_id] = rewritten
        logger.debug(
            "TaskManager.detach_staged: %d staged task(s) re-parented %s → %s",
            len(rewritten), from_parent_id, to_parent_id,
        )

    async def _flush_staged(self, task_id: str) -> None:
        """run 正常结束时把缓冲区的子任务入队。

        队列是 LIFO（pop 取末尾），因此按投入顺序 reversed 后再 push，
        使同一批次内的子任务 pop 出来是 FIFO（投入顺序）；整批位于栈顶，
        相对更早批次仍是 LIFO（深度优先）。
        """
        staged = self._staged.pop(task_id, None)
        if not staged:
            return
        if self._pause_abandon or self._cancelled:
            # pause 弃子窗口 / 硬取消（含熔断 trip 封闸）：本轮 staged 的子任务直接丢弃
            # （push 时才发 TASK_CREATED，无投影残留），防止清队后又有漏网新任务入队被派发——
            # _cancelled 分支专堵在途 run 收尾迟到的 staged 子任务变成新的搁浅任务。
            logger.info("staged tasks of %s dropped (pause_abandon=%s cancelled=%s): %d",
                        task_id, self._pause_abandon, self._cancelled, len(staged))
            return
        for task, blocked_by, parent_task_id in reversed(staged):
            await self.push_task(task, blocked_by=blocked_by, parent_task_id=parent_task_id)

    def _effective_agent(self, task: "Task | None") -> str:
        """任务实际执行所在的 agent id——同 agent 串行判定的键（单一真相见 effective_agent_id）。"""
        root = self._session.root_agent_id if self._session else ""
        return effective_agent_id(task, root)

    def _session_unhealthy(self, task_id: str) -> bool:
        """该 task 所属会话是否处于存储隔离（spec: event-commit）。无健康面时恒 False。"""
        if self._unhealthy_check is None:
            return False
        task = self._tasks.get(task_id)
        if task is None:
            return False
        return bool(self._unhealthy_check(task.session_id))

    def _task_unattended(self, task_id: str) -> bool:
        """该 task 是不是无人值守作业。查不到的按「有人」算——保守压向受闸门约束那边。"""
        task = self._tasks.get(task_id)
        return bool(task.unattended) if task is not None else False

    def _port_of(self, task_id: str) -> str:
        """该 task 接在哪个交互口上。查不到的按主口算——同 `_task_unattended`，
        保守压向受闸门约束那边（宁可多串行一次，不可让两条往返撞进同一个口）。"""
        task = self._tasks.get(task_id)
        return task.port_key if task is not None else PORT_MAIN

    def _held_ports(self) -> "set[str]":
        """此刻被占着的交互口（port 闸门的判据；单值版本始于 2026-09-22）。

        占着 = 存在一个 task **接在这个口上**（`port_key` 非空）、**要等对端**
        （非 unattended）、且处于「在跑 **或** `AWAITING_HUMAN`」。

        互斥的理由是一次往返的应答必须能路由回发起方：同一个口上两条并行的问话，
        回来的答复无从归属。所以**必须算上 `AWAITING_HUMAN`，只看在跑是不够的**——
        park 一发生槽位就还回来了（`_settle` 对 `_PARKED_STATUSES` 先 `_release_slot`
        再立刻 `drain()`），于是下一个同口 task 被派发、它也 park，那个口上就有两条
        同时等应答的往返。

        另外两个 park 态**不算**，它们等的不是对端：
        - `SUSPENDED`（等子任务）——父等子时子正该跑，算进来整个 DAG 立刻死锁；
        - `INTERRUPTED`（等 `/resume` 的故障态）——故障不该把正常工作一并按住。

        「接口但自治」（`port_key` 非空 + `unattended=True`）**不占口**：它不参与往返，
        只是跑完把产出推到那个口。它既不被别人拦，也不拦别人——见 `drain` 的谓词。
        """
        return {
            task.port_key
            for task_id, task in self._tasks.items()
            if task.port_key and not task.unattended
            and (task_id in self._running_tasks or task.status == "AWAITING_HUMAN")
        }

    async def drain(self) -> None:
        """Pop and run unblocked tasks until nothing more can be dispatched.

        三道闸门，形状同构（都是「按某个键去重」），只是键不同：

        1. **同 agent 不并发**（键 = `effective_agent_id`）：跳过"目标 agent 正忙"的
           条目，它们留在队列里，等该 agent 空闲（某任务完成 → on_task_finished →
           再 drain）时被选中。
        2. **交互口闸门**（键 = `port_key`，2026-09-22 起为单值，现按 port 分桶）：
           同一个交互口上同时至多一个 task 占着往返，见 `_held_ports`。两类 task 不受
           此限——自治作业（含「接口但自治」：只把产出推过去，不参与往返），以及接在
           别的口上的 task（btw 之类的旁支交互线与主线并行）。
        3. **并发槽**（无键，计数）：一个池，**所有 task 都计入**；但交互 task 可以
           临时把池顶上去（2026-09-28）。

        第 3 条的形状值得说清。池只有一个（`max_concurrent`），在跑的自治作业和交互
        task 一起算在里面。差别在于**谁会被它挡住**：

        - **自治作业**严格受限：在跑总数够了就排队等。
        - **交互 task 不看这个数**，等于自带一份临时额度。额度不会失控，因为第 2 条
          已经保证每个口至多一个往返——**一条口最多让池涨 1**。

        于是有效上限 = ``max_concurrent + 当前要派发的交互口数``，而自治作业会被交互
        挤压（交互占了槽，自治的可用空间就少了）。这正是想要的优先级：一个有人在等的
        往返被压在队列里干等、而队列对那个人不可见，是最糟的结果；自治作业多等一会儿
        没人在意。

        **`max_concurrent <= 0` 仍是总闸**：一个都不派，交互 task 也不例外。这条既有
        语义刻意保留——它是「这个 session 暂不派发」的表达方式，被多处依赖。
        """
        if self._runner is None:
            raise RuntimeError("No task runner registered")

        if self._cancelled:
            return

        while True:
            # 被同一 session 上更新的 TM 顶替（recover_agent 覆盖了 _task_managers 映射）→
            # 立即停止派发，无声（不发事件、不改状态）。避免重叠 resume 下两套 drain 并行派发
            # 同一批任务；在跑协程照旧靠 _fire_session_done 处的 _is_current 收敛（spec/07 §9）。
            if self._hooks.is_current is not None and not self._hooks.is_current():
                return
            async with self._lock:
                if self._max_concurrent <= 0:
                    break       # 总闸（见 docstring）：一个都不派，交互 task 也不例外
                busy_agents = {
                    self._running_agents.get(tid) or self._effective_agent(self._tasks.get(tid))
                    for tid in self._running_tasks
                }
                # 每次循环重算：上一轮 pop 出的 task 已进 `_running_tasks`（同在本锁内），
                # 它若占着某个口，这一轮就该把那个口关上。
                held_ports = self._held_ports()
                # 一个池，所有 task 都计入（见 docstring 第 3 条）。同样每轮重算——
                # 上一轮 pop 出的 task 已进 `_running_tasks`（同在本锁内）。
                running_count = len(self._running_tasks)
                entry = self._queue.pop(
                    skip=lambda e: (
                        self._effective_agent(self._tasks.get(e.task_id)) in busy_agents
                        or self._session_unhealthy(e.task_id)
                        or (not self._task_unattended(e.task_id)
                            and self._port_of(e.task_id) in held_ports)
                        # 自治作业严格受池约束；交互 task 不看这个数（自带临时额度，
                        # 上限由第 2 条的「一口一往返」兜住）。
                        or (self._task_unattended(e.task_id)
                            and running_count >= self._max_concurrent)
                    )
                )
                if entry is None:
                    break
                self._running_tasks.add(entry.task_id)

            asyncio.create_task(self._run_task(entry.task_id))

    async def _run_task(self, task_id: str) -> None:
        assert self._runner is not None
        task = self._tasks.get(task_id)
        if task is not None:
            task.status = "ACTIVE"
            task.actor_done = False
            # 上一轮的挂起意图不得带进这一轮：父任务被子任务唤醒后重跑，若 delegate 置的
            # suspend_requested 还留着，ActStep 会再路由去 SuspendStep、父永远醒不过来。
            task.suspend_requested = False

        # ── 阶段 1：装配（assemble）────────────────────────────────────────
        # 失败发生在 TASK_STARTED 之前 → 投影不会出现幽灵 ACTIVE；与执行失败
        # 共用重试路径，但 TASK_REQUEUED.reason=assembly_failure 可区分。
        try:
            binding = await self._runner.assemble(task_id)
        except PersistenceUnavailableError:
            # spec: event-commit——存储不可用：不 commit_round（会再撞同一故障）、
            # 不走失败处置（不发终态事件——emit 同样要过提交门）、不重排。会话已被
            # CommitGate 先标记 storage_unavailable；本 run 就此停住。
            logger.error("Task %s halted: session storage unavailable", task_id)
            return
        except Exception as e:
            logger.exception("Task %s assembly failed: %s", task_id, e)
            # 装配就炸了也算数（同下方收尾兜底的理由）：先提交，再发失败事件。
            if not await self._commit_round_or_halt(task_id):
                return
            await self._handle_task_failure(
                task_id, error=str(e), exc=e, reason=InterruptReason.ASSEMBLY_FAILURE,
            )
            return
        if binding is None:
            # task 已不存在（装配空转）：镜像旧行为（runner 首行 get_task None 即返回）
            await self.on_task_finished(task_id, status="FINISHED")
            return

        # 回填「真正用于执行的 agent id」+ 真实启动时刻；TASK_STARTED 由 TM 发，
        # 每次派发（含 retry/resume）恰好一条（reducer 只落非空 id；见 spec/07）。
        if task is not None:
            task.assigned_agent_id = binding.agent_id
            task.started_at = now_utc()
        self._running_agents[task_id] = binding.agent_id
        await self._emit(EventType.TASK_STARTED, task_id=task_id,
                         payload={"assigned_agent_id": binding.agent_id})

        # ── 阶段 2：执行（execute）────────────────────────────────────────
        # 消费 RunOutcome 有**两条**入口：正常返回（completed / awaiting_human /
        # suspended_on_children / canceled 四种，run 自己 return 回来）与崩溃
        # （`_run_loop` 重抛 → 下面的 `except Exception` 就地构造）。两条喂**同一张**
        # 处置表，「还能不能原地重试」的判断因此只剩 disposition_for 一处。
        try:
            try:
                outcome = await self._runner.execute(binding, task_id)
            except RoundDiscarded:
                # 这一轮在 LLM 开口之前被用户中止（spec 2026-09-09）：整轮抹掉。
                #
                # **不走 `apply_run_outcome`**——那会发一条 task 状态事件，而窗口这时
                # 已经要关了，那条事件会直接落盘，等于白丢。agent 回 `idle` 由抛出点
                # 在窗口内发的那条 `TASK_CANCELED` 负责（ALM 映射成 `SETTLED`），到这里
                # 内存状态已经摆平，只剩「关窗 + 抹痕迹」。
                #
                # `discard_round` 必须是本分支的**最后一步**，理由同上：窗口一关，
                # 此后任何事件都直接落盘。
                self._staged.pop(task_id, None)
                await self.discard_round(task_id)
                return
            except BaseException:
                # cancel(CancelledError) / 异常退出：丢弃未 flush 的缓冲，
                # 防止泄漏或日后 resume 时被误入队。再交回外层原有处理。
                self._staged.pop(task_id, None)
                raise
            if outcome is None:
                # 这次执行没留下结局（task 已不存在等）→ 按「没炸、没挂起、没取消」兜底。
                # **与旧路径并不完全同形**：旧路径走 `on_task_finished(FINISHED)`，一条 task
                # 事件都不发；这里会经处置表发一条
                # `TaskFinished{"outcome": "success", "summary": "", "outputs": None}`。
                # 当前接线下本分支不可达（`_tasks` 从不删条目；`_SessionTaskRunner.execute`
                # 只在 `get_task(task_id) is None` 时返回 None，而 `_execute_task` 恒返回
                # 非 None 的 state），故不是行为差异；留着是给未来的 runner 实现兜底。
                outcome = RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict=VERDICT_SUCCESS)
            # 处置**先于** _flush_staged：子任务一入队就可能跑完、回头唤醒父亲，而
            # `_try_resume_parent` 的门是 `parent.status == "SUSPENDED"`——父亲必须在
            # 子任务入队前落到 SUSPENDED（旧路径由 control tool 在 run 内写，同一时序）。
            # 未提交窗口的收尾兜底：这一轮跑到了有结局的地步（正常收尾 / park /
            # 挂起 / 失败都算），那它就**算数**——哪怕一个 chunk 都没吐出来（outage
            # 重试耗尽、provider 永久错、装配炸掉）。补一次提交，让那条用户消息和这个
            # task 如实进日志，`/resume` 才有东西可续。
            #
            # 只有用户主动中止走 `discard_round`，那条路由 act 抛 `RoundDiscarded`
            # 触发、发生在这之前，到这里 `_rounds` 已经没有它了 → 本句 no-op。
            #
            # 位置要在 `apply_run_outcome` **之前**：那一句会发 task 状态事件，得先让
            # 窗口里攒的 `TASK_CREATED` 出去，不然状态事件会先于创建事件落盘。
            if not await self._commit_round_or_halt(task_id):
                return
            status = await self.apply_run_outcome(task_id, outcome)
            # runner 正常跑完才把本轮 spawn 的子任务入队（“一轮跑完之后 push”）
            await self._flush_staged(task_id)
            await self._settle(task_id, status)
            # **收尾的最后一步**：这次停下来等的问题若已经有答复，说明那条应答落在了本段
            # 收尾里、它的重排被「还在跑」挡掉了。槽位此刻已经释放，补做即可。
            if outcome.kind is RunOutcomeKind.AWAITING_HUMAN:
                await self._wake_if_already_answered(task_id, outcome.hitl_id)
        except PersistenceUnavailableError:
            # spec: event-commit——同 assemble 分支：不提交、不处置、不重排。
            # 会话隔离已由 CommitGate 标记；drain 的健康检查挡住后续派发。
            logger.error("Task %s halted: session storage unavailable", task_id)
            return
        except Exception as e:
            if getattr(e, "retriable", False):
                logger.warning("Task %s failed (retriable): %s", task_id, e)
            else:
                logger.exception("Task %s failed: %s", task_id, e)
            # 同上：崩溃也算数，先提交再发终态事件。
            if not await self._commit_round_or_halt(task_id):
                return
            # 崩溃入口：结局的构造在 `crash_run_outcome` 一处（retriable 的取法是崩溃
            # 专用的，与 outage 支硬编码的 False 不同源——契约见那个工厂的 docstring）。
            status = await self.apply_run_outcome(task_id, crash_run_outcome(e))
            await self._settle(task_id, status)

    async def _wake_if_already_answered(self, task_id: str, hitl_id: str) -> None:
        """run 停在「等人」并收尾完毕 → 自检：它等的那个问题是不是已经有答复了。

        有 → 补一次重排。那条应答一定是落在本次收尾里的：它到达时 `resume_task` 看见
        task 还在 `_running_tasks`（槽位要到 `_settle` 才释放，早放会让同一个 task 被派发
        两次），于是早退、既不入队也不 drain，而这次唤醒没有任何地方记得。应答侧照旧推
        一把（快路径），本方法是那条推送落空时的兜底——它不看「有没有人推过」，只看状态。

        判据只认**这次 park 所等的那个 hitl_id**（`RunOutcome.hitl_id`，也就是
        `TaskAwaitingHuman` 那条事件的 payload），所以不会变成无条件重排：重排后的 run
        若再停下来，等的是另一个问题，那个问题还没答复。

        配对事件不会重复：工具类应答那条路的 `resume_task` 是在发 `TaskHumanResolved`
        **之前**早退的，这里补发正好一次；wait 气泡那条路已经由 `mark_human_resolved`
        发过并把状态置成 PENDING，`resume_task` 的 `was_blocked` 判据因此不成立，只入队。

        best-effort：判据读不到（钩子未注入 / 抛异常）就当作没答复——退回本方法存在之前
        的行为，`/resume` 仍然救得回来。
        """
        ready = self._hooks.human_answer_ready
        if not hitl_id or ready is None:
            return
        try:
            answered = ready(task_id, hitl_id)
        except Exception:
            logger.exception(
                "_wake_if_already_answered: predicate failed for task %s hitl %s",
                task_id, hitl_id)
            return
        if not answered:
            return
        logger.info(
            "Task %s parked on %s, but the answer had already arrived during the run's "
            "settle window — re-queuing it here (the reply's own wake was dropped)",
            task_id, hitl_id)
        await self.resume_task(task_id, hitl_id=hitl_id)
        await self.drain()

    async def _commit_round_or_halt(self, task_id: str) -> bool:
        """`_run_task` 的收尾兜底提交。失败 → 记日志、返回 False，调用方就地停住。

        与 `PersistenceUnavailableError` 两个既有分支同一处置：不处置 task（状态事件
        同样要过提交门）、不重排。窗口与暂存留在快照里（`commit_round` 失败不弹快照），
        下一个提交点重试；进程重启则按日志与 memory 各自的真相恢复——没补投成功的这一轮
        在两边都不存在。

        曾经直接 `await self.commit_round(...)`：它抛出的异常若在 `except Exception` 块
        里发生，同级的 `except PersistenceUnavailableError` 接不住，异常逃出 `_run_task`，
        `_running_tasks` 不清理，agent 在进程内一直显示忙。
        """
        try:
            await self.commit_round(task_id)
        except Exception:
            logger.exception(
                "Task %s halted: committing its round failed (window and staged memory "
                "are kept for a retry)", task_id)
            return False
        return True

    async def apply_run_outcome(self, task_id: str, outcome: RunOutcome) -> str:
        """run 的结局 → task 的处置：写状态 + 发那一条 task 状态事件。**唯一入口**。

        loop 只报「发生了什么」（RunOutcome），这里回答「那么 task 变成什么」——判据
        全在 `disposition_for` 那张纯函数表里，本方法只负责执行：写 status、写回
        retry_count、发事件。返回落定的状态，供调用方选出口。公开是因为`run_single_task`
        那条 compat 路径（没有队列、不经 `_run_task`）也要消费同一份结局。
        """
        task = self._tasks.get(task_id)
        if (task is not None and outcome.kind is not RunOutcomeKind.COMPLETED
                and task.status in _TERMINAL_STATUSES):
            # 终态守卫（见 _TERMINAL_STATUSES）：熔断 trip 先把 root 判 FAILED、再对在跑
            # 的 root 发协作取消，那次取消的 CANCELED 不得盖回已写定的 FAILED。run 侧的
            # 同一守卫是 `_run_loop` 的 cancel_takes_effect（它据此决定发不发 RUN_CANCELED）。
            return task.status
        disp = self._decide_and_write(task_id, outcome)
        await self._emit(EventType(disp.event_type), task_id=task_id, payload=disp.payload)
        return disp.status

    def _decide_and_write(self, task_id: str, outcome: RunOutcome) -> "Disposition":
        """结局 → 处置，并就地把状态写进 task。**纯同步、不发事件**。

        抽出来是为了让带外判决（`apply_out_of_band_verdict`）能在**同一个锁内**完成
        「仲裁 + 转移」——事件发射带 await，留在锁里会把临界区撑开，那正是竞态的来源。
        """
        task = self._tasks.get(task_id)
        disp = disposition_for(
            outcome,
            retry_count=task.retry_count if task is not None else 0,
            max_retries=task.max_retries if task is not None else self._task_max_retries,
        )
        if task is not None:
            task.status = disp.status
            if disp.event_type == "TaskRequeued":
                # **处置表不 mutate**：新的 retry_count 只在 payload 里，必须在这里写回，
                # 否则重试预算永不消耗、同一个 task 无限重排（旧路径是先 `+= 1` 再写 payload）。
                task.retry_count = disp.payload["retry_count"]
            if outcome.kind is RunOutcomeKind.INTERRUPTED:
                # 成因随 task 走：`disp.payload` 里的 error_code 只随这一条事件走一次，
                # `get_task(task_id)` 等后续内存态查询要看的是 task 对象上的这份存档
                # （注意 TaskView 不带该字段、跨重启不还原，见总账 A9）。
                task.error = outcome.error or task.error
                if outcome.error_code:
                    task.error_code = outcome.error_code
        return disp

    async def apply_out_of_band_verdict(
        self,
        task_id: str,
        outcome: RunOutcome,
        *,
        process_report: str = "",
        task_summary: str = "",
        next_step_hint: str = "",
        finalize: "Callable[[], Awaitable[None]] | None" = None,
    ) -> bool:
        """**带外判决**：一条不属于任何活跃 run 的结局（2026-09-22）。

        唯一调用方是后台 observe：park 之后 run 就地结束了（`_cold_park` 抛 `HitlPark`
        释放协程），它判完时这个 task 没有任何 run 在跑，`_close_report` 那条「同 run 内
        由 finalize 取用」的路走不通。

        返回 **True = 判决被接受**（observer 字段已落地），**False = 被仲裁拒绝、task 一个
        字段都没动**。注意 True 不等于「task 终结了」：只有 `success` 才走处置表，
        `retry` / `fail` 维持 park——「没做完」在有人在场时该由人拍板，不该由系统自动
        重排或判死（`max_retries` 是给无人值守的自动重跑设的失控护栏）。

        仲裁判据：只接受仍停在 `AWAITING_HUMAN` 的 task。人可能已经开口重排了它——
        `_inject_user_turn` 写 USER_PROMPT 时**不等**后台 observe，所以「人先开口」是
        常态而非边角。那时这份判决已经过时：人还有话说，这个 task 显然没完。

        **检查与转移在同一个锁内**，中间没有 await。否则「检查时还是 AWAITING_HUMAN、
        转移时已被重排」的窗口依然在，仲裁就成了摆设。事件发射与队列收尾留在锁外：
        那时 `task.status` 已是终态，人的消息会在 `send_message` 层走
        「current_task 已终态 → `_start_task_for_agent` 建新 task」，不会丢。

        observer 的三个文本字段由调用方传进来在锁内一并写：后台 observe 跑在
        `readonly` 的 ControlContext 上（见 `ControlContext.readonly`），工具自己没写，
        判决被拒时它们也就不该落地。

        ``finalize``：**带外收尾**——在「状态已写定」与「转移已宣布」之间那个窗口里跑
        （2026-09-27）。三条约束各钉一侧，顺序不能动：

        - **在仲裁之后**：判决被拒时一个字都不能写。收气泡、bubble 给 parent、软删自身
          对话，对一个还要继续跑的 task 全是破坏。
        - **在 `_settle` 之前**：`_settle` → `on_task_finished` → `_try_resume_parent`
          会唤醒 parent，parent 不能在子任务的产出落地之前醒过来装配。
        - **状态写留在锁内**，这是关门点：`task.status` 一落终态，人的消息在
          `send_message` 层就按 `_task_is_terminal` 走新建分支，不会再 `requeue` 这个
          正在收尾的 task。所以带 IO 的这一段跑在锁外是安全的。

        只在真发生了转移（即 `success`）时调用——`retry`/`fail` 维持 park，没有收尾可言。
        **失败不吞掉转移**：抛异常只记日志，照常往下发 `TaskFinished` + `_settle`。状态
        已经是终态了，不发事件宿主永远不知道、不 `_settle` 则槽位永不释放、交互线永久
        占着；降级成「这个 task 少了收尾副作用」远好过卡住整个会话。
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != "AWAITING_HUMAN":
                logger.info(
                    "out-of-band verdict rejected for task %s (status=%s): it is no longer "
                    "parked — the user spoke first",
                    task_id, getattr(task, "status", "<missing>"),
                )
                return False
            if process_report:
                task.process_report = process_report
                task.process_report_at = now_utc()
            if task_summary:
                task.task_summary = task_summary
            task.next_step_hint = next_step_hint or None
            # 归一后再落地与比较（`retry` 是别名，认不出的归 continue——决不归 fail）：
            # 这条入口的入参直接来自 LLM 的工具回传，不能假定它已经是规范值。
            verdict = normalize_verdict(outcome.verdict) if outcome.verdict else ""
            if verdict:
                task.observer_outcome = verdict
            if verdict != VERDICT_SUCCESS:
                # retry / fail → 维持 park。字段已写（hint 供下一轮 act 用），但不转移
                # 状态：观察者说「没做完」时，在有人在场的会话里该由人决定下一步。
                logger.info(
                    "out-of-band verdict '%s' for task %s: fields recorded, staying parked "
                    "for the user",
                    verdict, task_id,
                )
                return True
            disp = self._decide_and_write(task_id, outcome)

        if finalize is not None:
            try:
                await finalize()
            except Exception:
                logger.exception(
                    "out-of-band finalize failed for task %s; announcing the transition "
                    "anyway (the task is already terminal; skipping the announcement would "
                    "strand the slot and hold the interaction line forever)", task_id)
        await self._emit(EventType(disp.event_type), task_id=task_id, payload=disp.payload)
        await self._settle(task_id, disp.status)
        return True

    async def _settle(self, task_id: str, status: str) -> None:
        """处置落定之后的队列收尾：挂起出口 / 重排出口 / 终态收尾，三选一。

        出口逻辑与旧路径逐字相同，只是判据从「loop 写进 task.status 的值」换成了
        「处置表算出来的状态」。
        """
        if status in _PARKED_STATUSES:
            # 子任务已由 control tool 推入队列；parent 等待所有子任务完成后
            # 由 _try_resume_parent 重新入队，此处只需移出 running set 并 drain。
            # 三种非终态停顿共用这一条出口：SUSPENDED（等子任务）、AWAITING_HUMAN
            # （HitlPark，等人应答）、INTERRUPTED（LLM 故障 / run 崩溃，待 /resume 由 restore 重排）。
            async with self._lock:
                self._release_slot(task_id)
            # 自治作业停在 INTERRUPTED 时没有对端会来救它（见 `_schedule_autonomous_requeue`）
            # ——SDK 自己安排恢复。返回 True 表示耗尽支已落 FAILED 并自行收完，就地 return。
            if status == "INTERRUPTED" and await self._schedule_autonomous_requeue(task_id):
                return
            await self.drain()
            # 整个会话因 park/suspend 进入空闲（无在跑任务、无待派子任务）→ 通知 runtime 回收
            # 按 run 计的控制信号。注意是 is_done（而非"本 task 挂起"）：父等子时子仍在跑，
            # is_done 为 False、不触发，待子完成 resume 父；只有全会话静止才算空闲挂起。
            if self.is_done():
                await self._fire_session_idle()
            return
        if status == "PENDING":
            # observer 判 retry（本轮未完成，含机械退出）或崩溃后可重试：重新入队。
            async with self._lock:
                self._release_slot(task_id)
                self._queue.push(QueueEntry(task_id=task_id, session_id=self._session_id))
            await self.drain()
            return
        # 终态：用处置算出的那个，避免把 FAILED/CANCELED 覆盖成 FINISHED
        final_status: TaskStatus = (  # type: ignore[assignment]
            status if status in _TERMINAL_STATUSES else "FINISHED"
        )
        await self.on_task_finished(task_id, status=final_status)

    async def _handle_task_failure(
        self, task_id: str, *, reason: str, error: str = "",
        exc: BaseException | None = None,
    ) -> None:
        """**装配失败**专用：先尝试自动 retry，耗尽或不可重试 → 挂起等恢复（绝不落终态 FAILED）。

        Task 4 起，**执行**（execute）崩溃不再走这里：它由 `_run_task` 的 `except
        Exception` 就地构造 `RunOutcome(INTERRUPTED, reason=InterruptReason.RUN_CRASH)`，喂
        `disposition_for` 那张表（重试判断因此只剩一处）。本方法保留是因为装配阶段
        （assemble）没有 run、也就没有 RunOutcome，且它的 TASK_REQUEUED
        `reason=InterruptReason.ASSEMBLY_FAILURE` 是对外可区分的契约。`reason` 因此改为
        必传：旧的默认值 `"run_failure_retry"` 随执行崩溃那条路径一起退场，留着只会是个
        再也不会出现的字面量。


        运行层崩溃（异常退出，未经 observer/FinalizeStep）是**可恢复中断**，不是任务失败：
        真失败只有 observer 判 fail 一条路（FinalizeStep 闭合胶囊、发 TaskFailed、回传父亲）。
        exc.retriable=False（如 LLMCallError 401 认证失败、CONTEXT_OVERFLOW）时跳过重试直接挂起，
        避免对确定性错误做无效重试。
        """
        task = self._tasks.get(task_id)
        if task is not None and task.status in _TERMINAL_STATUSES:
            # 终态不复活（见 apply_run_outcome 的同源守卫）：熔断 trip 可能已把这个
            # task 判 FAILED，此后的装配失败属内部清场，不该把写定的终态盖回非终态、
            # 也不再发任何事件。
            #
            # 但队列簿记的清场不能省：本方法只在 `_run_task` 的 assemble 异常分支被
            # 调用，而 `_run_task` 是 `drain()` 把 task_id 加进 `_running_tasks`（并
            # mark_running 进队列）之后才 `asyncio.create_task` 出来的，所以此刻
            # task_id 必然还占着一个 running 槽位——不摘掉它会话就会永久少一个可派
            # 发的并发槽位（tests/unit/test_terminal_guard_on_assembly_failure.py 的
            # test_terminal_guard_still_clears_running_bookkeeping 测的正是这个）。
            # drain() 之后让别的排队任务照常派发；trip 场景下 `_cancelled` 已置位，
            # drain() 本就空转，调用是安全的。
            #
            # 到此为止——不再跟 `_suspend_task_interrupted` 尾段一样调 is_done() /
            # `_fire_session_idle()`。这个 task 走到本分支时已经是终态：它的终态要么
            # 是熔断 trip 判的（`_trip_failure_threshold` 第 8 步已经同步跑完
            # `_fire_session_done()`，会话早已收尾），要么是 observer 判 fail 经
            # `on_task_finished` 判的（那条路自己就会在同一次调用里做完 is_done() /
            # 收尾判断）。两条路都已经把「会话是不是完了」回答过一次；此处再报一次
            # "idle"（暗示"挂起等恢复"）对一个已经终态收场的会话是错的信号，且
            # `_on_session_idle` 回调不是幂等收尾专用的 `_on_session_done`，不该在
            # 这里替它多按一次。
            async with self._lock:
                self._release_slot(task_id)
            await self.drain()
            return
        # 不可重试的错误（如认证失败 / 上下文溢出），不重试、直接挂起等恢复
        if exc is not None and not getattr(exc, "retriable", True):
            logger.warning(
                "Task %s non-retriable error (%s), suspending for recovery: %s",
                task_id, type(exc).__name__, error,
            )
            await self._suspend_task_interrupted(task_id, error, exc, reason=reason)
            return

        if task is not None and task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = "PENDING"
            task.error = error
            logger.info(
                "Task %s retrying (%d/%d): %s",
                task_id, task.retry_count, task.max_retries, error,
            )
            async with self._lock:
                self._release_slot(task_id)
                entry = QueueEntry(task_id=task_id, session_id=self._session_id)
                self._queue.push(entry)
            # 重排落事件：使投影从 ACTIVE 回到 PENDING；进程在重试间隙崩溃时
            # restore 据 PENDING 重排（而非把停留 ACTIVE 的任务误当成可恢复后重跑）。
            await self._emit(EventType.TASK_REQUEUED, task_id=task_id, payload={
                "reason": reason,
                "retry_count": task.retry_count,
            })
            await self.drain()
        else:
            await self._suspend_task_interrupted(task_id, error, exc, reason=reason)

    # ── 自治作业的 INTERRUPTED 自动恢复（2026-09-28）────────────────────────
    #
    # `INTERRUPTED` 的定义是「等 `/resume`」，而 `/resume` 预设了一个操作者：它的唯一
    # 运行期入口是 `requeue_resumable` ← `_resume_in_existing_tm` ← host 主动调用。
    # 自治作业（`Task.unattended`）没有对端，没有人会对一个后台作业按「继续跑」——于是
    # 它只能靠下一次 `recover_session` 被 `restore` 顺带捡回来，也就是说**恢复依赖一个
    # SDK 无法保证的外部事件**，期间会话既不 idle 收尾也不 done
    # （`_blocked_or_interrupted()` 恒为 True）。
    #
    # 修法是让 SDK 自己发起那个恢复动作。**追加式**：既有的 INTERRUPTED 出口一个字节
    # 不改（`TASK_INTERRUPTED` 照发、槽位照放、投影照旧），只在末尾多安排一次退避重排。
    #
    # 判据刻意**不按 `reason` 分类**：`LLM_OUTAGE` 是最典型的暂时故障，它的
    # `RunOutcome.retriable` 却被刻意设成 False（那个字段答的是「允不允许原地立即重
    # 试」，不是「故障是不是暂时的」，见 `disposition_for` 的契约），而 `RUN_CRASH` /
    # `ASSEMBLY_FAILURE` 两边都可能。统一退避 + 有限预算，让「退几次都不成」自己回答
    # 「有没有希望」，比维护一张暂时/永久对照表可靠。

    async def _schedule_autonomous_requeue(self, task_id: str) -> bool:
        """自治作业停在 INTERRUPTED → 安排退避重排；预算耗尽则落 FAILED。

        返回 **True = 本方法已接管这个 task 的收尾**（耗尽支自己跑完了
        `on_task_finished`，含 drain / is_done / 会话信号），调用方应当就地 return，
        别再发一次 idle——对一个已经终态收场的会话报 "idle" 是错的信号。
        返回 False = 没接管（不是自治作业 / 状态已被别的路径改过 / 只是起了个定时器），
        调用方按原路径继续。
        """
        task = self._tasks.get(task_id)
        if task is None or not task.unattended:
            return False            # 有对端的 task：那个 /resume 有人会按
        if task.status != "INTERRUPTED":
            return False            # 已被别的路径改写（重排 / 取消 / 终态）→ 不插手
        if self._cancelled:
            return False
        if task.interrupt_requeue_count >= self._autonomous_requeue_max:
            # 预算耗尽 → 响亮失败。**不能**停在 INTERRUPTED：那是个没有对端会来碰的
            # 非终态，会话因此永不收敛。落 FAILED 则 failure_counter 递增、可能触发
            # 熔断——这是刻意的（熔断的意义就是「环境不对了，别硬撑」），阈值已随之
            # 从 3 提到 5，见 `Session.failure_threshold`。
            logger.warning(
                "Autonomous task %s exhausted its requeue budget (%d) — failing loudly",
                task_id, self._autonomous_requeue_max,
            )
            task.error_code = TaskErrorCode.AUTONOMOUS_REQUEUE_EXHAUSTED
            await self._emit(EventType.TASK_FAILED, task_id=task_id, payload={
                "error_code": TaskErrorCode.AUTONOMOUS_REQUEUE_EXHAUSTED,
                "error_message": task.error or "",
                "retry_count": task.retry_count,
            })
            await self.on_task_finished(task_id, status="FAILED")
            return True
        delay = self._autonomous_requeue_backoff_base_sec * (2 ** task.interrupt_requeue_count)
        prev = self._autonomous_requeue_timers.get(task_id)
        if prev is not None and not prev.done():
            return False            # 已经有一个在飞，不叠第二个
        timer = asyncio.create_task(self._requeue_autonomous_after(task_id, delay))
        self._autonomous_requeue_timers[task_id] = timer
        logger.info(
            "Autonomous task %s interrupted — auto-requeue in %.0fs (%d/%d)",
            task_id, delay, task.interrupt_requeue_count + 1, self._autonomous_requeue_max,
        )
        return False

    async def _requeue_autonomous_after(self, task_id: str, delay: float) -> None:
        """退避 `delay` 秒后把自治作业放回队列。取消即静默退出。

        醒来后三重自检：会话已硬取消 / 本 TM 已被顶替 / task 状态已不是 INTERRUPTED
        （被 `/resume`、`restore`、取消或别的路径处理过）——任一成立就什么都不做。
        退避期间世界会变，定时器不该拿着一份过期的判断去改状态。
        """
        try:
            await asyncio.sleep(delay)
            if self._cancelled:
                return
            if self._hooks.is_current is not None and not self._hooks.is_current():
                return
            task = self._tasks.get(task_id)
            if task is None or task.status != "INTERRUPTED":
                return
            async with self._lock:
                if task_id in self._running_tasks:
                    return          # 已经被别的路径派发了
                if any(e.task_id == task_id for e in self._queue.peek_all()):
                    return          # 已经在队列里
                task.interrupt_requeue_count += 1
                task.status = "PENDING"
                # retry_count 归零，与 `restore` / `requeue_resumable` 同口径：这是
                # 「重新起一次」，不是「接着上次的重试预算跑」。
                task.retry_count = 0
                self._queue.push(QueueEntry(
                    task_id=task_id, session_id=self._session_id, priority=task.priority,
                ))
            await self._emit(EventType.TASK_REQUEUED, task_id=task_id, payload={
                "reason": InterruptReason.AUTONOMOUS_REQUEUE,
                "retry_count": task.retry_count,
            })
            await self.drain()
        except asyncio.CancelledError:
            return
        finally:
            self._autonomous_requeue_timers.pop(task_id, None)

    async def _suspend_task_interrupted(
        self, task_id: str, error: str, exc: BaseException | None, *, reason: str,
    ) -> None:
        """运行层崩溃的终局：挂起等 /resume，**不是失败**。

        置 INTERRUPTED（非终态）+ 发 TASK_INTERRUPTED——投影只认事件类型
        （TASK_STATUS_BY_EVENT），不发则任务停留 ACTIVE、restore 语义错位。这是一条
        **task 级事实**：TM 在 run 外面、不拥有 run（连 run_id 都拿不到），run 域的
        `RunInterrupted` 由 `runtime._run_loop` 自己发。会话/agent 怎么了也不在这里
        宣布——本方法发的这条 TASK_INTERRUPTED 由 `AgentLifecycleManager` 的五态机
        （`_INPUT_BY_EVENT`）接手，折成对应 agent 的 AGENT_INTERRUPTED（判据是事件
        类型，不是 reason 字面量）。2026-09-04（Task 12）前这里还会额外经
        `announce_queue_state` 聚合出一条会话级队列信号，现已停发、其事件类型也已于
        2026-09-05 删除（那条信号唯一的消费者早已降格）。
        **发射时机是重试判定之后**：`_handle_task_failure` 决定原地重试的那一支发的是
        TASK_REQUEUED（→ PENDING），只有落到本方法才是「停在 INTERRUPTED 等 /resume」。
        不发 TASK_FAILED、不增 failure_counter、不闭合胶囊：真失败只有 observer 判 fail
        一条路。恢复由 /resume → restore() 据非终态重排（重排时 retry_count 归零）。
        """
        task = self._tasks.get(task_id)
        error_code = crash_error_code(exc)
        if task is not None:
            task.status = "INTERRUPTED"
            task.error = error
            task.error_code = error_code
        async with self._lock:
            self._release_slot(task_id)
        # reason 只作**溯源**，不作路由——判据是 TASK_INTERRUPTED 这个类型本身。
        # 值本身是对外契约的一部分（host 升级须知的映射表按 reason 分流展示文案），
        # 故按调用方透传的 `reason` 原样发出，不再在这里硬编码单一来源——
        # `_handle_task_failure` 现在服务装配失败（ASSEMBLY_FAILURE）一条路，
        # 执行崩溃改走 `_run_task` 的 `except Exception`（RunOutcome 路径），两者的
        # reason 必须同源透传，不得在挂起出口各写各的
        # （tests/unit/test_assembly_failure_reason.py 检测的正是这一点；
        # tests/unit/test_layered_signals.py 检测的是「分流」而非「出现」）。
        await self._emit(EventType.TASK_INTERRUPTED, task_id=task_id, payload={
            "reason": reason,
            "error_code": error_code,
            "error_message": error,
            "retry_count": task.retry_count if task else 0,
        })
        # 同 `_settle` 的 park 出口：自治作业没有对端会按 /resume，SDK 自己安排恢复。
        if await self._schedule_autonomous_requeue(task_id):
            return
        # 其它 agent 的排队任务照常派发；全会话静止则通知 runtime 回收 per-run 控制信号
        await self.drain()
        if self.is_done():
            await self._fire_session_idle()

    async def on_task_finished(self, task_id: str, status: TaskStatus) -> None:
        # task 落终态 → 通知外部回收挂在 task 上的运行期状态（当前是 CapabilityCache 的 pin）。
        # **这里就是「终态」那条路**：`_settle` 判 retry（status == "PENDING"）时先 return，
        # 根本走不到这个函数，所以重试自动保住 pin，不需要额外判据。best-effort：清理失败
        # 不该拦住任务收尾。
        if self._hooks.on_task_terminal is not None:
            try:
                await self._hooks.on_task_terminal(task_id)
            except Exception:
                logger.exception("TaskManager: on_task_terminal callback failed for %s", task_id)

        async with self._lock:
            self._clear_running(task_id)
            if status in ("FAILED", "CANCELED"):
                # spec: task-handoff——终态但非成功，**不解锁任何后继**：它们依赖的是
                # 这个任务的产出，而产出不存在。后继的善后由永久阻塞扫描处置。
                self._queue.mark_failed(task_id)
            else:
                self._queue.mark_complete(task_id)

            # Update task object
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                task.finished_at = now_utc()

        # ── 取消胶囊闭合 funnel（Task 14）───────────────────────────────────────
        # 在途协作取消的任务：发信号时（cancel_all 的 CancelToken / 熔断的 cancel_inflight）
        # 只做了 ack 替换（幂等自愈），finish 对要等这里——终态真正坐实（`apply_run_outcome`
        # 据处置表把 task.status 写成 CANCELED；Task 4 之前是 _run_loop 自己写）——才补写。
        # 正常收尾的任务走 FinalizeStep，永不落到这个分支
        # （status 只会是 FINISHED/FAILED，与本 if 互斥）。未真正 start 过的任务（started_at 为
        # 空）不会有派发框/own scope 可闭，交由 synthesize_cancel_closure 的 find-only 兜底判定
        # 即可，这里额外用 started_at 提前短路只是省一次无意义调用。
        if status == "CANCELED" and task is not None and task.started_at \
                and self._hooks.cancel_finalizer is not None:
            try:
                await self._hooks.cancel_finalizer([task], task.error or "cancelled")
            except Exception:
                logger.exception("TaskManager: cancel_finalizer callback failed for %s", task_id)

        # ── failure_counter 维护 ──────────────────────────────────────────────
        if self._session is not None:
            if status == "FAILED":
                self._session.failure_counter += 1
                self._recent_failures.append((
                    # 标题 + id：这份清单会被渲进 root 的 finish 对（runtime 的阈值收尾），
                    # 同名任务只印标题时读者分不清是哪一个失败了几次。
                    task_ref(task) if task else task_ref_parts(task_id, ""),
                    ((task.error or task.process_report or "") if task else "")[:200],
                ))
                threshold = self._session.failure_threshold
                if (
                    threshold > 0
                    and self._session.failure_counter >= threshold
                    and not self._threshold_tripped
                ):
                    await self._trip_failure_threshold()
                    return
            elif status == "FINISHED":
                self._session.failure_counter = 0  # 成功时重置
                self._recent_failures.clear()  # 随 counter 同步清空
            elif status == "CANCELED":
                # 用户主动中断：标记 session 为 CANCELED，防止 is_done() 误判为 SUCCEEDED。
                # pause 弃子（_pause_abandon）除外：连带取消不定会话去向，由 root park 决定。
                # _threshold_tripped 除外：熔断已判会话 FAILED，在途任务迟到的协作取消收尾
                # 绝不能把终态盖回 CANCELED（trip 序列已先手，见 _trip_failure_threshold）。
                # spec: task-handoff——依赖阻塞取消除外：那是计划内部的失败传播，不是
                # 用户叫停；子任务被阻塞取消后父任务照常唤醒、观察并决定补救/收尾，
                # 会话终态交给正常收敛路径（同 _pause_abandon 的豁免形态）。
                if (
                    not self._pause_abandon
                    and not self._threshold_tripped
                    and not (task is not None and task.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP)
                ):
                    self._session.status = "CANCELED"

        # spec: task-handoff——永久阻塞善后的运行期入口：前序 FAILED/CANCELED 落定后，
        # on_success 后继若已判明永不可满足 → 立即处置（级联到不动点）。
        await self.dispose_blocked_dependents()

        # Try to resume parent
        await self._try_resume_parent(task_id)
        # Drain next
        await self.drain()

        # 若 queue 已空且无任务在运行，通知 session 真正结束
        # （有重试时 drain() 会把重试任务入队，is_done() 为 False，不触发）
        if self.is_done():
            # 被顶替旧 TM 的迟到收尾不得代表会话发信号——新 owner 的状态才是真相。
            # runtime 侧回调本就 compare-and-check，这里连事件也一并静默，避免污染
            # 事件流的 host 显示与重放。
            if self._hooks.is_current is not None and not self._hooks.is_current():
                return
            # 「还有人在等」不再靠注入的 pending-HITL 谓词判断，而是由 AWAITING_HUMAN 的
            # 任务表达；「断了」由 INTERRUPTED 的任务表达。这里只区分「要不要走终结
            # 收尾」——后者要 gather 后台协程 + 回调（2026-09-04 Task 12 起两者都不再
            # 额外经 `announce_queue_state` 聚合成会话级信号，那条信号已停发）。
            if self._blocked_or_interrupted():
                await self._fire_session_idle()
            else:
                if self._session is not None and self._session.status not in ("FAILED", "CANCELED"):
                    # failure_counter > 0 表示本轮有任务失败（成功时会被重置为 0）
                    self._session.status = self._final_status()
                await self._fire_session_done()

    def _find_blocked_forever(self) -> "tuple[str, str] | None":
        """定位一个依赖已判明永不可满足的排队任务 → (task_id, 阻塞源 dep_id)。

        判据是**当前状态**而非回调：任何时刻某依赖已 FAILED/CANCELED，该条目就永不
        可能被释放（没有再让它成功的路径）。幂等：已终态任务跳过。
        """
        for entry in self._queue.peek_all():
            task = self._tasks.get(entry.task_id)
            if task is None or task.status in TERMINAL_TASK_STATUSES:
                continue
            for dep in entry.blocked_by:
                dep_task = self._tasks.get(dep)
                if dep_task is not None and dep_task.status in ("FAILED", "CANCELED"):
                    return entry.task_id, dep
        return None

    async def dispose_blocked_dependents(self) -> None:
        """spec: task-handoff——永久阻塞善后（运行期 + 恢复期共用，幂等）。

        定点迭代至不动点：处置一个被阻塞取消的任务后它自己也成了 CANCELED 终态，
        依赖它的 on_success 后继随之满足处置条件，重扫直到无新受害者。每个受害者：
        落 CANCELED + ``BLOCKED_BY_FAILED_DEP`` + 指明阻塞源，发 TASK_CANCELED
        （payload 带 error_code 与 blocked_by_task_id，投影据此可恢复解释），出队，
        然后走 on_task_finished 的正规收尾（队列记账 / 父任务唤醒 / drain / 会话判定
        ——其中会话终态改写被 error_code 豁免，见该分支注释）。

        崩溃窗口兜底：A 的 FAILED 已落盘、B 的级联取消未落盘时进程崩溃，恢复重建
        依赖后由 runtime 在首次 drain 前调本方法补扫（同一实现、同一幂等）。
        """
        if self._in_blocked_dispose:
            return  # 处置内部的 on_task_finished 再入；外层循环已覆盖其级联
        self._in_blocked_dispose = True
        try:
            while True:
                found = self._find_blocked_forever()
                if found is None:
                    return
                task_id, dep_id = found
                task = self._tasks.get(task_id)
                if task is None:
                    # `_find_blocked_forever` 已过滤过；真到这里说明登记表被并发改动，
                    # 必须 break 而非 continue——重扫会原样再返回同一条，成死循环。
                    break
                task.status = "CANCELED"
                task.error_code = TaskErrorCode.BLOCKED_BY_FAILED_DEP
                task.error = f"blocked by failed/canceled predecessor {dep_id}"
                task.finished_at = now_utc()
                self._queue.cancel(task_id)
                await self._emit(
                    EventType.TASK_CANCELED, task_id=task_id,
                    payload={
                        "error_code": TaskErrorCode.BLOCKED_BY_FAILED_DEP,
                        "blocked_by_task_id": dep_id,
                        "reason": f"dependency_failed:{dep_id}",
                    },
                )
                await self.on_task_finished(task_id, status="CANCELED")
        finally:
            self._in_blocked_dispose = False

    async def _trip_failure_threshold(self) -> None:
        """连败达阈值：真终结——清场 + root 判死 + 会话终态，而非裸终结。

        从 on_task_finished 的 FAILED 分支进入（其 `not _threshold_tripped` 前置保证只进一次）。
        步骤对应机制设计「trip 序列」1-8；顺序是定案，改动前请对照 task-9-brief.md。
        """
        # 1) 幂等闩置位；_cancelled 封闸——drain 守卫白拿，_flush_staged 的
        #    `_pause_abandon or _cancelled` 丢弃条件也据此堵住在途 run 迟到的 staged 子任务。
        self._threshold_tripped = True
        self._cancelled = True
        threshold = self._session.failure_threshold if self._session else 0
        counter = self._session.failure_counter if self._session else 0
        logger.warning(
            "Session %s failure_counter=%d reached threshold=%d → 熔断真终结",
            self._session_id, counter, threshold,
        )

        # 2) FAILURE_THRESHOLD_HIT，payload 带本轮已知连败清单
        await self._emit(EventType.FAILURE_THRESHOLD_HIT, payload={
            "failure_counter": counter,
            "threshold": threshold,
            "failures": [{"title": title, "reason": reason}
                         for title, reason in self._recent_failures],
        })

        # 3) 取消该 session 所有未决 pending HITL（best-effort）：HitlCancelled 须全部
        #    先于会话终态发出，防止 host 投影翻态早于 hitl 侧收尾。
        if self._hooks.cancel_pending_hitl is not None:
            try:
                await self._hooks.cancel_pending_hitl()
            except Exception:
                logger.exception("TaskManager: cancel_pending_hitl callback failed")

        # 清场**分类**交纯函数（见 failure_threshold.py）；**顺序**留在这里。
        async with self._lock:
            pending = self._queue.drain_pending()
        plan = plan_threshold_trip(
            self._tasks, pending_ids=pending, running_ids=self._running_tasks,
        )

        # 4) 清队 + 5) 取消挂起：非 root 直接标 CANCELED + 发事件。
        #    发事件的顺序仍是「先清队条目、后挂起条目」，与改造前一致。
        for tid in (*plan.cancel_queued, *plan.cancel_suspended):
            t = self._tasks[tid]
            t.status = "CANCELED"
            t.finished_at = now_utc()
            await self._emit(
                EventType.TASK_CANCELED, task_id=tid,
                payload={"reason": CancelReason.FAILURE_THRESHOLD},
            )

        # 5b) 在途非 root：只发协作取消信号，不发事件（其 TASK_CANCELED 由 run 结束后的
        #     `apply_run_outcome` 发；finish 对交由 on_task_finished 的取消胶囊闭合 funnel；
        #     已启动带框者进 plan.ack_task_ids，供 threshold_finalizer 做 eager ack 替换）。
        for tid in plan.signal_inflight:
            self._signal_cancel(tid)

        # 5.5) 立即整对闭合已终态的取消任务（清队 + 挂起，均已启动）——它已经是终态，
        #      没有后续 on_task_finished 会来补 finish 对。在途任务保持 eager ack + funnel。
        if plan.cancel_now_ids and self._hooks.cancel_finalizer is not None:
            try:
                await self._hooks.cancel_finalizer(
                    [self._tasks[tid] for tid in plan.cancel_now_ids],
                    CancelReason.FAILURE_THRESHOLD,
                )
            except Exception:
                logger.exception(
                    "TaskManager: cancel_finalizer callback failed (threshold cleanup)")

        # 6) root 判 FAILED。**先标 FAILED 再**对在跑的 root 发 cancel_inflight——顺序
        #    保证两道守卫都接得住：task 侧是 `apply_run_outcome` 的终态守卫（不把 FAILED
        #    盖回 CANCELED）；run 侧是 `_run_loop` 的 cancel_takes_effect（不发 RUN_CANCELED）。
        root_we_failed_and_started: Task | None = None
        for tid in plan.fail_roots:
            t = self._tasks[tid]
            t.status = "FAILED"
            t.error_code = TaskErrorCode.BY_THRESHOLD
            t.error = (f"Session failure threshold reached "
                       f"({counter} consecutive sub-task failures).")
            t.finished_at = now_utc()
            await self._emit(EventType.TASK_FAILED, task_id=tid, payload={
                "error_code": TaskErrorCode.BY_THRESHOLD,
                "error_message": t.error,
            })
            if t.started_at is not None:
                root_we_failed_and_started = t
        for tid in plan.signal_roots:
            self._signal_cancel(tid)

        # 7) finalizer：内联 await（不是后台甩），保证 memory 落盘先于 SESSION_FINISHED
        #    （SSE 关闭）；异常只记日志不阻断终结。
        if self._hooks.threshold_finalizer is not None:
            try:
                await self._hooks.threshold_finalizer(
                    root_we_failed_and_started,
                    [self._tasks[tid] for tid in plan.ack_task_ids],
                    list(self._recent_failures),
                )
            except Exception:
                logger.exception("TaskManager: threshold_finalizer callback failed")

        # 8) 会话终态 + 收尾事件。终态直接写定为 FAILED（2026-09-04 Task 12 起不再
        #    经由队列聚合信号/会话状态机——熔断的前提就是 failure_counter 已达阈值，
        #    `_final_status()` 此刻本就恒为 FAILED，无需绕一圈再落定）。
        if self._session is not None:
            self._session.status = "FAILED"
        await self._fire_session_done()

    async def cancel_all(self, *, reason: str = "") -> None:
        """硬取消整条 session 链：清空 pending 队列并标 CANCELED，会话置 CANCELED。

        在途 task 不在此处理——由 CancelToken → act checkpoint → CancelledError →
        `_run_loop` 交回 `RunOutcome(CANCELED)` → `apply_run_outcome` 置该 task CANCELED
        → on_task_finished（其 drain() 被 _cancelled 守卫挡住，
        finish 对经 on_task_finished 的取消胶囊闭合 funnel 补写，见 Task 14）。

        标态后立即对已启动的任务（`started_at` 非空——含 root：root 没有 origin_tool_call_id，
        但 synthesize_cancel_closure 的 own-root 形态不需要框，仍要闭合自己的 scope）经
        `_cancel_finalizer` 闭合胶囊（ack 终态化 + `[outcome=cancelled]` finish 对）。未启动
        的任务从未铸框/写过任何 memory，跳过——零 memory 写。
        """
        self._cancelled = True
        # 在飞的自治退避定时器一并取消：它们醒来后本来也会自检 `_cancelled` 而静默退出，
        # 但那意味着最多再挂 120s。硬取消让 session 立刻关得干净。
        for timer in list(self._autonomous_requeue_timers.values()):
            timer.cancel()
        self._autonomous_requeue_timers.clear()
        async with self._lock:
            pending = self._queue.drain_pending()
        to_close: list[Task] = []
        for tid in pending:
            t = self._tasks.get(tid)
            if t is not None:
                t.status = "CANCELED"
                t.finished_at = now_utc()
                if t.started_at:
                    to_close.append(t)
            await self._emit(EventType.TASK_CANCELED, task_id=tid, payload={"reason": reason})
        if to_close and self._hooks.cancel_finalizer is not None:
            try:
                await self._hooks.cancel_finalizer(to_close, reason or CancelReason.USER_CANCEL)
            except Exception:
                logger.exception("TaskManager: cancel_finalizer callback failed (cancel_all)")
        if self._session is not None:
            self._session.status = "CANCELED"

    def _signal_cancel(self, task_id: str) -> None:
        """对在途 task 发协作取消信号（best-effort：缺注入或异常都不阻断 trip 序列）。"""
        if self._hooks.cancel_inflight is None:
            return
        try:
            self._hooks.cancel_inflight(task_id)
        except Exception:
            logger.exception("TaskManager: cancel_inflight callback failed for %s", task_id)

    def _clear_running(self, task_id: str) -> None:
        """从「在跑」登记里摘掉这个 task。**调用方须已持 `self._lock`。**"""
        self._running_tasks.discard(task_id)
        self._running_agents.pop(task_id, None)

    def _release_slot(self, task_id: str) -> None:
        """归还一个派发槽位：清在跑登记 + 解除队列的 running 标记。

        **调用方须已持 `self._lock`。** 五个非终态出口（park / 重排 / 装配失败的终态
        守卫 / 装配失败的重试 / INTERRUPTED 挂起）逐字共用这三行——不摘干净会话就永久
        少一个并发槽位。终态出口（`on_task_finished`）不走这里：它的队列侧动作是
        `mark_complete` / `mark_failed`（两者内部已 `_running.discard`），只需
        `_clear_running`。
        """
        self._clear_running(task_id)
        self._queue.unmark_running(task_id)

    def _agent_id_of(self, task_id: str) -> str | None:
        """先看正在跑的登记，再回落到 task 自己的 assigned_agent_id。"""
        running = self._running_agents.get(task_id)
        if running:
            return running
        task = self._tasks.get(task_id)
        return task.assigned_agent_id if task is not None else None

    async def _emit(
        self,
        event_type: EventType,
        task_id: str | None = None,
        payload: dict | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        """构造并发出一个 session/task 级别的事件（无 LoopState）。

        V2 §0：envelope 管身份。``agent_id`` 未显式传且有 ``task_id`` 时自动解析
        （先查在跑登记 ``_running_agents``，查不到再回落到 ``Task.assigned_agent_id``）；
        无 ``task_id``（如 ``FAILURE_THRESHOLD_HIT`` 这类会话级聚合信号）则 agent_id
        保持 None，不乱填。
        """
        await emit_event(
            self._event_bus,
            event_type,
            session_id=self._session_id,
            tenant_id=self._session.tenant_id if self._session else "default",
            origin=_ORIGIN,
            task_id=task_id,
            agent_id=(
                agent_id if agent_id is not None
                else (self._agent_id_of(task_id) if task_id is not None else None)
            ),
            payload=payload,
        )

    def _final_status(self) -> str:
        """全部终态时的会话结论。failure_counter > 0 表示本轮有任务失败。

        2026-09-04（Task 12）起 `announce_queue_state` 及其队列聚合事件已停发（事件
        类型于 2026-09-05 删除），本方法不再是谁的数据源——唯一读者是本文件
        `_settle` 里的 ``self._session.status = self._final_status()``（终态收尾
        直接写 session 状态，不再经由已退役的会话状态机）。**只返回终态值**：
        空串或别的东西会把 `_session.status` 写成一个不存在的状态。
        """
        if self._session is not None and self._session.failure_counter > 0:
            return "FAILED"
        return "SUCCEEDED"

    def _blocked_or_interrupted(self) -> bool:
        """还有非终态停顿的任务在等人/等 /resume——会话不算走完。"""
        return any(t.status in _PARKED_STATUSES for t in self._tasks.values())

    async def _fire_session_idle(self) -> None:
        """会话空闲挂起（park/suspend，非终结）：通知 runtime 回收 per-run 控制信号。

        2026-09-04（Task 12）起不再先报队列状态——`announce_queue_state` 已停发
        （其消费者会话状态机早已降格，见 events-v2 §5），本方法只剩通知这一件事。
        """
        if self._hooks.on_session_idle is not None:
            try:
                await self._hooks.on_session_idle()
            except Exception:
                logger.exception("TaskManager: session_idle callback failed")

    async def _fire_session_done(self) -> None:
        """会话可能已经走完：先等后台协程，再走可选回调。

        2026-09-04（Task 12）起不再报队列状态——`announce_queue_state` 已停发
        （events-v2 §5），幂等因此全靠 `_on_session_done` 侧的回收本身幂等。
        """
        # 等待所有后台协程完成，确保 RecognizeIntent 等事件全部 emit 后再关闭 SSE 流
        if self._background_asyncio_tasks:
            await asyncio.gather(*list(self._background_asyncio_tasks), return_exceptions=True)
        # 归属权判定放在 gather **之后**：本 TM 可能在等待后台任务期间被逐出
        # （forget/purge 之后又有人为这个 session 建了新 TM）。此时本 TM 已非 owner →
        # 收尾变 no-op，绝不发 SessionFinished、绝不触发 on_session_done。
        if self._hooks.is_current is not None and not self._hooks.is_current():
            logger.info("TaskManager(%s): superseded during session-done; skip callback",
                        self._session_id)
            return
        # 同一个 TM 在 gather 期间接了下一轮（单例：`send_message` /
        # `start_session(resume=True)` / `/resume` 都复用活 owner，不再另建 TM）。
        # 上一轮这次迟到的收尾不代表会话走完了——新一轮正在跑或正停在等人，此时回调
        # 会冲掉新一轮的控制信号（`_release_round` 清 run token 与 pause 闩锁）。
        if not self.is_done() or self._blocked_or_interrupted():
            logger.info("TaskManager(%s): next round started during session-done; skip callback",
                        self._session_id)
            return
        if self._hooks.on_session_done is not None:
            try:
                await self._hooks.on_session_done()
            except Exception:
                logger.exception("TaskManager: session_done callback failed")

    async def _try_resume_parent(self, finished_task_id: str) -> None:
        parent_id = self._parent_map.get(finished_task_id)
        if parent_id is None:
            return

        # 判定 all_done → 翻转 SUSPENDED→PENDING → push 三步必须在同一临界区内完成：
        # 否则两个（同 agent）子任务并发完成时会各自读到 all_done=True + status==SUSPENDED，
        # 双双 push/resume 父任务（父在同一 scope 上并发跑两遍，污染 memory）。drain 留到锁外。
        #
        # PENDING 不是 ACTIVE（D5）：ACTIVE 的正主是 TASK_STARTED（TM 派发时发、回填
        # assigned_agent_id）。这里只是「解除阻塞、入队」，真正开始跑要等 drain 派发到它。
        # 此前置 ACTIVE 是入队前的抢跑，投影因此在派发前的窗口就误报「在跑」。
        resumed = False
        async with self._lock:
            siblings = self._children_of.get(parent_id, set())
            # 空集守卫：无已登记子任务时绝不 resume（all([]) 恒为 True 的 vacuous-truth 防御）。
            all_done = bool(siblings) and all(
                (self._tasks[tid].status if tid in self._tasks else "PENDING")
                in TERMINAL_TASK_STATUSES
                for tid in siblings
            )
            if all_done:
                parent_task = self._tasks.get(parent_id)
                if parent_task and parent_task.status == "SUSPENDED":
                    parent_task.status = "PENDING"
                    self._queue.push(QueueEntry(
                        task_id=parent_id,
                        session_id=self._session_id,
                    ))
                    resumed = True

        if resumed:
            logger.info("All children of %s done, resuming parent", parent_id)
            # 先发事件、再 drain（评审 Important）：drain() 派发走 asyncio.create_task，
            # 不等子协程跑完就把控制权交还——旧顺序「drain 在前」曾经无害，是因为
            # TaskResumed/TaskStarted 都折叠成 ACTIVE，谁先谁后结果一样。D5 把
            # TaskResumed 改成 PENDING 后，若 TaskStarted（→ACTIVE）先落进事件流，
            # 重放会把一个真正在跑的任务钉成 PENDING。「解除阻塞」先于「开始执行」
            # ——这正是本次改动的核心语义，事件顺序也要跟着这条排。
            await self._emit(EventType.TASK_RESUMED, task_id=parent_id)
            await self.drain()

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def children_of(self, task_id: str) -> set[str]:
        """该 task 已派生的子任务 id 集合（用于 blackboard 订阅）。"""
        return set(self._children_of.get(task_id, set()))

    def all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def is_done(self) -> bool:
        """True when queue is empty and nothing is running."""
        return not self._queue.has_pending() and not self._running_tasks

    async def finalize_idle_session(self, status: str) -> None:
        """恢复专用：会话所有 task 已终态但 session 因崩溃未落终态 —— 设终态并复用
        _fire_session_done（先 gather 重跑的后台 recap，再发 SESSION_FINISHED + 回调）。

        镜像 on_task_finished 的会话收尾：走同一条 `_fire_session_done` 链，终态由调用方
        传入的 `status` 直接写定（2026-09-04 Task 12 起不再经由队列聚合信号/会话状态机）。幂等由 `_fire_session_done` 内的归属权判定 + `_on_session_done`
        自身幂等承担。
        """
        if self._session is not None:
            self._session.status = status
        await self._fire_session_done()

    async def resume_task(self, task_id: str, *, hitl_id: str) -> None:
        """重排一个被 HITL 应答唤醒的 task：置 PENDING 并入队，供**复用活 owner**的就地续跑路径。

        不重建 TM——直接把该 task 塞回本 owner 的队列。已终结/在跑/已在队列的任务不重复入队。
        wait_for_user 冷应答已由 `_inject_user_reply` 置 PENDING（并自行发 `TaskHumanResolved`，
        见其 docstring）；approval 走此路径重排后由 reconcile 重放 dangling tool_call——这里
        才是 approval 分支唯一发 `TaskHumanResolved` 的地方，与 `TaskAwaitingHuman{hitl_id}`
        配对（D4）。

        **只在真的解除了「被人挡住」时才发那条事件**（`was_blocked` 判据）：`_inject_user_reply`
        走 wait_for_user 分支时已经委托 `mark_human_resolved` 把状态改成 PENDING 并发过一次；若这里不做这个判断，
        紧随其后的这次调用会对同一个 hitl_id 重复发 `TaskHumanResolved`——配对就不再是一对一。
        """
        t = self._tasks.get(task_id)
        if t is None or t.status in TERMINAL_TASK_STATUSES:
            return
        if task_id in self._running_tasks:
            return
        if any(e.task_id == task_id for e in self._queue.peek_all()):
            return
        was_blocked = t.status in ("AWAITING_HUMAN", "SUSPENDED")
        t.status = "PENDING"
        t.retry_count = 0  # 挂起期间的旧计数不带入新一轮 attempt
        self._queue.push(QueueEntry(
            task_id=task_id, session_id=self._session_id, priority=t.priority,
        ))
        if was_blocked:
            await self._emit(EventType.TASK_HUMAN_RESOLVED, task_id=task_id,
                              payload={"hitl_id": hitl_id})

    def requeue_resumable(self, parked_task_ids: "set[str] | None" = None) -> list[str]:
        """`/resume` 撞上**活的 owner**：就地把可续跑的 task 重新入队，不重建 TM。

        session 的 TaskManager 是单例（`CtxWeftRuntime._bind_task_manager`），活 owner
        在世时不允许再建一个去顶替它，于是「崩溃后从事件日志 `restore()`」的那套判据
        在这里改读**本 TM 内存里**的状态——owner 活着，内存才是真相：

        - 终态 / 已废弃的辅助 task（compact / metadata）/ 有未决 HITL 的：不动
          （与 `restore()` 第二趟逐条相同）；
        - SUSPENDED：只在子任务全终态时重排——尚有活子任务的要留给
          `_try_resume_parent` 唤醒，提前改状态会把那次唤醒吞掉（见 `restore()`）；
        - 其余非终态（INTERRUPTED / 被应答后没人驱动的 AWAITING_HUMAN / PENDING 却不在
          队列里……）：置 PENDING 入队，`dag_deps` 里尚未终态的仍作阻塞。

        另加两道只有活 TM 才需要的闸（与 `resume_task` 同口径）：正在跑的、已在队列
        里的不重复入队——同一个 task 被派发两次就是同一 agent 上的两个并发 run。

        与 `restore()` 一样不发事件：真正开跑时 drain 发的 `TASK_STARTED` 才是事实。
        不 drain，drain 交给调用方。返回真正重排了的 task id。
        """
        parked = parked_task_ids or set()
        queued = {e.task_id for e in self._queue.peek_all()}
        terminal = {tid for tid, t in self._tasks.items() if t.status in TERMINAL_TASK_STATUSES}
        requeued: list[str] = []
        for t in self._tasks.values():
            if (t.id in terminal or t.id in self._running_tasks or t.id in queued
                    or t.id in parked
                    or isinstance(t.settings, (CompactTaskSettings, MetadataFillerTaskSettings))):
                continue
            if t.status == "SUSPENDED":
                if not all(cid in terminal for cid in self._children_of.get(t.id, set())):
                    continue
                blocked: set[str] = set()
            else:
                blocked = {dep for dep in (t.dag_deps or []) if dep not in terminal}
            t.status = "PENDING"
            t.retry_count = 0  # 挂起期间的旧计数不带入新一轮 attempt
            self._queue.push(QueueEntry(
                task_id=t.id, session_id=self._session_id,
                priority=t.priority, blocked_by=blocked,
            ))
            requeued.append(t.id)
        return requeued

    async def requeue_for_message(self, task_id: str, *, reason: str = "send_message") -> bool:
        """把一个被**外部消息**（`CtxWeftRuntime.send_message`，Task 18）重新激活的
        task 放回队列——发 `TaskRequeued`，**不是** `TaskHumanResolved`。

        `TaskHumanResolved` 是 `TaskAwaitingHuman{hitl_id}` 的**一对一配对解除事件**
        （见 `resume_task` docstring："唯一发 TaskHumanResolved 的地方…配对就不再是
        一对一"；`reducers.py` 同一断言）——它只属于 HITL 应答路径。外部消息注入
        没有对应的 `TaskAwaitingHuman`，硬发它会在事件流里留一个配不上对的孤儿，还
        会经 ALM（`_INPUT_BY_EVENT` 把它译成 `AgentInput.HUMAN_RESOLVED`）把 agent
        状态**提前**翻成 `running`——task 这时只是入了队，真正的 `TASK_STARTED`
        要等 `drain()` 真派发才发，这个窗口里 agent 报 running 但其实没在跑，会让
        紧接着的下一次 `send_message` 被 `assert_can_receive` 误判成 `AgentBusyError`
        拒收。

        `TASK_REQUEUED` 才是语义对的事实：`reducers.py` 自己的注释说它和
        `TaskHumanResolved` "效果相同（判据是类型不是 payload）"，区别只在后者多背
        了一层 HITL 配对；ALM 把它译成 `AgentInput.SETTLED` → agent 回 `idle`，
        正确反映"已入队、尚未开跑"，`drain()` 真正派发时 `TASK_STARTED` 才会把它
        翻成 `running`——时序对得上。

        与 `resume_task` 同样的三道幂等闸（已终态 / 已在跑 / 已在队列 -> no-op，
        不重复入队不重发事实）；不看 `was_blocked`——不管当前是哪种非终态挡着，
        只要真的把它塞回了队列就发一次。返回是否真的发生了重排（供调用方决定要不要
        补一次 `drain()`——本方法本身不 drain，与 `resume_task` 同一分工，drain
        交给调用方，见 `_resume_in_existing_tm` / `_try_resume_parent` 的既有先例）。
        """
        t = self._tasks.get(task_id)
        if t is None or t.status in _TERMINAL_STATUSES:
            return False
        if task_id in self._running_tasks:
            return False
        if any(e.task_id == task_id for e in self._queue.peek_all()):
            return False
        t.status = "PENDING"
        t.retry_count = 0  # 挂起期间的旧计数不带入新一轮 attempt
        self._queue.push(QueueEntry(
            task_id=task_id, session_id=self._session_id, priority=t.priority,
        ))
        await self._emit(EventType.TASK_REQUEUED, task_id=task_id, payload={"reason": reason})
        return True

    async def mark_human_resolved(self, task_id: str, *, hitl_id: str) -> None:
        """HITL 应答落地后把 task 置回 PENDING 并发事实——**不看当前状态**。

        与 `resume_task` 的区别：后者有 `was_blocked` 门（只对
        AWAITING_HUMAN/SUSPENDED 生效），而 wait_for_user 冷应答的重建路径上
        `restore()` 先跑、已把该 task 的状态从 AWAITING_HUMAN 翻成了 PENDING，
        那道门必然落空——`_inject_user_reply` 走的就是这条入口。

        不入队（与旧行为一致）：wait_for_user 场景下重排交给调用方后续的驱动，
        本方法只落状态 + 发事实。

        终态不复活：已 FINISHED/FAILED/CANCELED 的 task 原样返回、不发事件。
        """
        task = self._tasks.get(task_id)
        if task is None or task.status in _TERMINAL_STATUSES:
            return
        task.status = "PENDING"
        await self._emit(EventType.TASK_HUMAN_RESOLVED, task_id=task_id,
                          payload={"hitl_id": hitl_id})

    def is_cancelled(self) -> bool:
        """本 TM 是否已被硬取消（cancel_all 置 _cancelled）——供派发点补投 born-cancel 判定。"""
        return self._cancelled

    def is_alive(self) -> bool:
        """本 TM 是否仍是该 session 的当前 owner。

        供 recover_agent 判断"是否有活 TM"，以决定复用还是重建。session 的 TM 是单例，
        活 owner 只会被复用、不会被顶替；它返回 False 只有一种情形：本 TM 已被逐出
        （forget/purge）。

        ⚠ **它只表达「有没有被顶替」，不表达「这一轮跑完没有」**（2026-09-08 生命周期
        改造后的口径收窄）。从前 `_fire_session_done` -> `_release_session` 会把终结的
        TM 从 `_task_managers` 摘掉，`is_current` 随即为 False，于是这个方法顺带兼了
        「未终结」的意思；现在 TM 常驻到显式 `forget_session`，一个已经发过
        `SessionFinished` 的 TM 在这里照样是 alive——**这是有意的**：它确实还能接新
        task（`drain` 只被 `_cancelled` 与 `is_current` 挡，`_fire_session_done` 不设
        任何闩），`_start_task_for_agent` 因此可以直接复用它，不必重建一份。
        要判断「这一轮是不是走完了」请用 `is_done()`。
        """
        return self._hooks.is_current is None or self._hooks.is_current()

    def running_task_ids(self) -> set[str]:
        """当前正在执行（已派发、_run_task 未返回）的 task id 集合。"""
        return set(self._running_tasks)

    def running_agent_of(self, task_id: str) -> str | None:
        """在跑 run 的真实执行 agent id（无此在跑任务 → None）。pause 弃子划分的真相源。"""
        return self._running_agents.get(task_id)

    def set_pause_abandon(self, flag: bool) -> None:
        """置位/复位 pause 弃子窗口标志。

        置位期间产生两个效果：① CANCELED 任务不再连带把 session 状态改成 CANCELED（会话去向留给
        root park 决定，见 _on_task_finished 里的判断）；② run 结束时本轮新 staged 出的子任务直接丢弃、
        不入队派发（见 _flush_staged），防止弃子清队后又漏网新任务被派发。
        """
        self._pause_abandon = flag

    async def abandon_pending(
        self, *, reason: str = CancelReason.PAUSE_ABANDON, keep_agent: str | None = None,
    ) -> list[str]:
        """放弃排队中任务（标 CANCELED、发 TASK_CANCELED），不触碰 session 状态。

        ``keep_agent`` 非 None **且该 agent 当前没有在途 run** 时：effective agent 等于它的
        排队条目保留在队列（相对顺序不变）、不放弃——pause 弃子须保留 root agent 已入队未派发
        的那一条（root run 尚未派发、无可 pause 的对象），交由调用方补 drain 派发成为唯一续跑点；
        否则它随全清被误取消，会话既无在途 run 又无排队条目、无气泡 → 永久滞留 RUNNING。
        keep_agent 已有在途 run（那一轮已被 pause→park 成续跑点）或为 None 时，root-scope 排队
        条目照旧全清——保证「一次暂停恰一个续跑点」，与全清行为一致。

        与 cancel_all 的差异：不置 _cancelled（弃子后仍需 drain 派发保留/重排的任务）、
        不把 session 置 CANCELED（pause 弃子不是用户取消）。
        """
        async with self._lock:
            entries = self._queue.peek_all()
            self._queue.drain_pending()
            busy_agents = set(self._running_agents.values())
            keep = keep_agent is not None and keep_agent not in busy_agents
            cancelled: list[str] = []
            for e in entries:
                if keep and self._effective_agent(self._tasks.get(e.task_id)) == keep_agent:
                    self._queue.push(e)  # 保留：相对顺序按原队列顺序回填
                else:
                    cancelled.append(e.task_id)
        for tid in cancelled:
            t = self._tasks.get(tid)
            if t is not None:
                t.status = "CANCELED"
                t.finished_at = now_utc()
            await self._emit(EventType.TASK_CANCELED, task_id=tid, payload={"reason": reason})
        # 队列弃子不经 on_task_finished，不会自动触发父任务重排：若某 SUSPENDED 父任务的
        # 子任务此刻**全部**还在排队（无一在途），无人调用 _try_resume_parent → 父任务永不
        # 重排、会话滞留 RUNNING 且无续跑点。此处对每个被弃子任务补触发重排检查（幂等：
        # 兄弟仍在途时 all_done 不成立、由其 on_task_finished 接力；父已 ACTIVE 不二次入队）。
        for tid in cancelled:
            await self._try_resume_parent(tid)
        return cancelled


@dataclass
class RoundSnapshot:
    """开窗那一刻的可回滚状态（spec 2026-09-09）。

    只装 **TaskManager 自己拥有** 的东西，外加这一轮暂存的 memory 写入（不透明地保管，
    落盘归 `commit_round` 钩子）。HITL 与 ALM 归 runtime —— 那两样经 `revert_round` 钩子回退。

    后三个字段是这份快照存在的主要理由：`_inject_user_turn` 会就地把 `outputs` /
    `process_report` / `process_report_at` 清空（"新消息意味着有新工作要做"），而旧值
    **没有任何别处存过**。不拍这一张，撤销就只能撤一半：消息没了，上一轮的进展也没了。
    """

    owns_task: bool
    hitl_id: str = ""
    task_status: str = ""
    retry_count: int = 0
    outputs: Any | None = None
    process_report: str | None = None
    process_report_at: "datetime | None" = None
    user_prompt_in_memory: bool = False
    #: `commit_round` 已把缓冲的事件补投出去（`RoundCommitted` + `commit_provisional`）。
    #: 提交在钩子那一步失败、快照保留时，重试据此跳过补投。
    events_flushed: bool = False
    #: 这一轮暂存的 memory 写入：`(memory provider, MemoryEvent, ProviderContext)`，按写入
    #: 顺序。见 `stage_memory`。
    staged_memory: "list[tuple[Any, Any, Any]]" = field(default_factory=list)


def task_payload(task: Task, *, user_prompt_jsonable: "str | list[dict] | None") -> dict:
    """TaskCreated 事件的 payload，供 sessions.py translate_event 构建前端 task 对象。

    ``user_prompt_jsonable``：`push_task` 在 `await self._emit(...)` 之前备好的
    event-jsonable 载荷（保 ref、绝不落字节；源头是入口从原始 content 算出的那一份，
    见 `push_task` 的 docstring）——**不**在本函数内部算，因为本函数是同步的、还负责
    十余个与内容无关的字段，async 化会让所有调用方等一次 IO（spec §6）。

    刻意做成**必传的 keyword-only 参数**（无默认值）：曾经有一条「默认 None 时退回
    同步 `content_to_jsonable`」的分支，效果是把 inline base64 直接塞回 TaskCreated
    payload——今天零调用方用它（`push_task` 是唯一调用点，且必传），留着只是把「事件
    库恒不含字节」这条不变量的破口焊死在函数签名的默认参数上，静静等下一个复用本函数
    的人在没有 await 的地方顺手调用。去掉默认值后，下一个复用者会在签名处就被挡住，
    而不是无声地把字节写回事件。
    """
    import dataclasses
    settings_d = dataclasses.asdict(task.settings)
    settings_d["_type"] = type(task.settings).__name__
    ts = task.created_at.isoformat() if task.created_at else ""
    prompt = user_prompt_jsonable
    return {
        "task": {
            "id": task.id,
            "session_id": task.session_id,
            "status": task.status,
            "title": task.title or "",
            "description": task.description or "",
            "creator_agent_id": task.creator_agent_id or "",
            "assigned_agent_id": task.assigned_agent_id or "",
            "parent_task_id": task.parent_task_id or "",
            # 空 part 列表被 `or ""` 降级成 ""：今日安全，因为本仓处处把 [] 与 "" 当等价的
            # "无内容"（没有生成合法的空/短 part 列表的路径）。Phase 3 若出现这样的合法列表
            # （例如一张裁掉了文字的纯图片 prompt 被上游误判为"空"），这里就会把它错误吞掉。
            "user_prompt": prompt or "",
            "priority": task.priority,
            "max_retries": task.max_retries,
            "dag_deps": task.dag_deps,
            "unattended": task.unattended,
            "port_key": task.port_key,
            "origin_tool_call_id": task.origin_tool_call_id or "",
            "origin_tool_name": task.origin_tool_name or "",
            "settings": settings_d,
            "result": None,
            "outputs": {},
            "error": None,
            "created_at": ts,
            "updated_at": ts,
        }
    }
