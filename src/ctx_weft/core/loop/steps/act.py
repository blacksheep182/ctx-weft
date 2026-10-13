"""ActStep：多 turn LLM 子循环 + capability 调用。

工具调用通过 CapabilityGateway 统一执行（授权 + 审计 + 内存），
ActStep 只处理 LLM 流 + turn 循环逻辑。
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ctx_weft.protocols import LLMMessage, LLMRequest, LLMUsage, ToolCall
from ctx_weft.providers.llm.stream import ensure_collected_stream
from ctx_weft.core.loop.driver import (
    ACT_TURNS_USED_KEY, COMPACT_NOOP_KEY, CONTEXT_RECOVERY_COUNT_KEY,
    HOT_REPLY_ROUND_KEY, RECOGNIZE_INTENT_PENDING_KEY, ROUND_COMMITTED_KEY,
    LoopContext, LoopState, Step, StepOutcome, make_event,
)
from ctx_weft.core.loop.llm_gateway import (
    PROMPT_EST_BASE_KEY, PROMPT_EST_SEG_KEY, request_prompt_estimate, resolve_llm_identity,
    stream_llm_resilient,
)
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
from ctx_weft.core.loop.park import HitlPark, RoundDiscarded
from ctx_weft.core.capabilities.control_tools import FINISH_TASK_NAME
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.core.loop.observing import has_observe_role
from ctx_weft.core.utils.estimate import effective_limit
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import MintedCall, generate_id, mint_turn_call_ids
from ctx_weft.protocols import MemoryEvent, MemoryKind, MemoryScope
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    HitlAsk,
    UserTurnDelivery,
)

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """Single LLM turn record (for Observer + memory)."""

    turn: int
    messages_sent: list[LLMMessage]
    assistant_text: str
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    tool_results: list[Any] = dataclasses.field(default_factory=list)
    usage: LLMUsage = dataclasses.field(default_factory=LLMUsage)


class ActStep(Step):
    """ActStep: multi-turn LLM loop + capability invocation via CapabilityGateway."""

    name = "act"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        prompt = state.assembled_prompt
        if prompt is None:
            raise RuntimeError("ActStep requires assembled_prompt — PrepareStep must run first")

        agent = state.agent
        # 轮数预算**跨「上下文恢复」累计**：越过停机线后本步会经 PrepareStep 压缩再重入
        # （见 `_can_recover_context`），若每次重入都从 1 重计，等于每恢复一次白送一整份
        # `max_turns_per_act`——而这个配置的语义是「一次执行最多几轮」，不是「每段最多几轮」。
        # `turn_num` 同样用累计值，使 ACT_TURN_STARTED 的轮号在一个 run 内单调递增。
        max_turns = agent.loop_config.max_turns_per_act
        turns_used = state.extra.get(ACT_TURNS_USED_KEY, 0)
        # transcript 同样累加而非覆盖：state_patch 是整字段覆盖，重入时本地变量若从空起头，
        # 压缩前那些轮次就从 state.transcript 消失了（消费方：observe 的
        # `_mechanical_verdict` 空 transcript → fail、background_observe 的 total_turns）。
        transcript: list[TurnRecord] = list(state.transcript)
        exit_reason = "normal"

        # 运行时态势 guidance（任务树/已完成子任务/收尾提醒）已随装配管线注入
        # （PrepareStep → GuidanceSource → composer 末条 user 尾部），此处不再修饰。
        current_messages = list(prompt.messages)
        # 动态 max_tokens 的增量基线：上一轮实际发送的 message 条数（= 本轮 usage 对应的真实
        # prompt 基线）。None=本轮无循环内真实基线（首轮）→ request_prompt_estimate 走整份估算。
        baseline_msg_count: int | None = None

        for turn_num in range(turns_used + 1, max_turns + 1):
            state.extra[ACT_TURNS_USED_KEY] = turn_num
            await _interrupt_checkpoint(state, ctx)
            await ctx.event_bus.emit(make_event(
                state, EventType.ACT_TURN_STARTED, payload={"turn": turn_num}))

            # 1) 单轮 LLM：流式累积文本 / reasoning / tool_calls / usage（软打断在内部 park）
            sent_msg_count = len(current_messages)  # 本轮发送条数（append 前）→ 下轮增量基线
            turn = await _run_llm_turn(state, ctx, prompt, current_messages, turn_num, baseline_msg_count)

            # 2) token 记账 + context_limit 判定
            context_limit_hit = await _account_tokens(state, ctx, turn.usage)
            # 仅当本轮真实刷新了 context_tokens（usage>0）才前移基线；否则保持旧值/None，
            # 让下一轮退回整份估算（更保守），避免基线与陈旧 context_tokens 错配。
            if turn.usage.prompt_tokens > 0:
                baseline_msg_count = sent_msg_count

            # 3) assistant 回合落 memory + 接回 message 历史 + 记 transcript
            persisted_turn = await _ingest_assistant_turn(
                state, ctx, turn.text, turn.tool_calls, turn.usage, turn_num,
                anchor=turn.anchor, minted=turn.minted)
            asst_tool_dicts = persisted_turn.tool_calls
            state.extra["assistant_record_id"] = persisted_turn.record_id
            current_messages.append(LLMMessage(
                role="assistant", content=turn.text, tool_calls=asst_tool_dicts,
                reasoning_content=turn.reasoning or None, provider_state=turn.provider_state))
            turn_record = TurnRecord(
                turn=turn_num, messages_sent=list(current_messages),
                assistant_text=turn.text, tool_calls=turn.tool_calls, usage=turn.usage)

            # 4) 纯文本（无 tool call）：interactive 让位用户 / 否则即任务产出 → observe
            #
            # **这一支刻意排在 context_limit 之前**（改造前相反）：纯文本就是本 run 的收尾，
            # 模型已经把答复交出来了，此刻压缩没有任何收益——而让 context_limit 抢先的代价是
            # 把一次正常收尾误报成机械退出，经 observe 强制改判 retry、白吃一次重试预算。
            if not turn.tool_calls:
                transcript.append(turn_record)
                await _finish_plain_text_turn(state, ctx, turn_num, transcript)
                break

            # 5) 派发前压缩（含派发调用且越阈值时；在 dispatch 执行 / 子 spawn-inherit 之前）
            await _maybe_predispatch_compact(state, ctx, turn.tool_calls, turn.usage)

            # 6) 执行 tool calls（同批次延后退出信号，保证 tool_call ↔ result 一一对应）
            #
            # **越过停机线也照执行完**（改造前在 4) 就 break 了，工具一个都不跑）。三条理由：
            # ① 工具调用已经发生过，结果是净收益，丢掉等于白烧一轮 LLM；
            # ② 不执行就没有 TOOL_RESULT，memory 里留下一条**悬挂的** assistant tool_call。
            #    改造前它被 observe 的 `_fold_retry_segment`（keep_last=0）连整段 raw 一起
            #    supersede 掉，坑被掩埋；一旦改成压缩后原地续跑，`collapse_keep_last` 会把它
            #    留在保留区，重装配出的 messages 就带一个无 result 的 tool_call，撞上
            #    `llm_gateway.drop_dangling_tool_calls`——不 400（有兜底），但那条 assistant
            #    会被剥成空消息再被 `remove_empty_messages` 整条删掉，模型看不到自己调过这些
            #    工具，且每次恢复都打一条 ERROR（那个兜底的语义是「上游对账漏了」）。
            # ③ 执行完可能进一步超出窗口，但这是安全的：装配预算层把 assistant(tool_calls)
            #    与其 tool 结果聚成同生共死单元再整单元丢弃（`budget._coalesce_tool_pairs`），
            #    永不产孤儿；且紧接着就是 compact，折的正是这批新料。
            tool_results = await _execute_tool_calls(state, ctx, turn.tool_calls)
            for tr in tool_results:
                current_messages.append(LLMMessage(
                    role="tool", content=tr["result"], tool_call_id=tr["tool_call_id"]))
            turn_record.tool_results = tool_results
            transcript.append(turn_record)
            await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
                "turn": turn_num,
                "reason": "context_limit" if context_limit_hit else "tool_calls_processed"}))

            # finish_task 与 delegate 同批：finish 胜出（派发改投为独立后继）。
            _reconcile_finish_vs_dispatch(state, ctx, turn.tool_calls)

            # 7) actor 收尾 —— 同样**优先于** context_limit：任务已经做完了，没有可续的跑。
            if state.task.actor_done:
                exit_reason = "actor_done"
                break

            # 8) 越过上下文停机线：工具结果已全部落定、配对完整 → 结束本段。
            #    去向由 `_can_recover_context` 决定（回 prepare 压缩续跑 / 退回 observe），
            #    见下方路由。**不清 pin**：`_pinned` 按 task_id 分区（新 task 看不到旧 pin）、
            #    有 max_pins 的 LRU 上界、且 task 落终态时由 `on_task_finished` 清——在这里
            #    清等于让 agent 续跑后从零重新发现工具，正是 `CapabilityCache.evict` 的注释
            #    明确拒绝在 run 边界做的那件事，而续跑连 run 边界都不跨。
            if context_limit_hit:
                exit_reason = "context_limit"
                logger.warning(
                    "ActStep context_limit_hit: prompt_tokens=%d >= %.2f * eff(limit=%d) "
                    "for agent %s", turn.usage.prompt_tokens,
                    getattr(agent.loop_config, "context_limit_stop_ratio", 0.8),
                    agent.loop_guard.context_limit, agent.id)
                break

        else:
            exit_reason = "max_turns"
            logger.warning("ActStep hit max_turns=%d for agent %s", max_turns, agent.id)
            await ctx.event_bus.emit(make_event(state, EventType.MAX_TURNS_REACHED, payload={
                "max_turns": max_turns}))

        # 离开 act 的循环 = 这一轮已经推进到别的步骤（suspend / prepare 压缩续跑 / observe），
        # 那些步骤直接读 memory、不认暂存区，所以先提交：暂存的答复与消息此刻落盘。
        # 热应答醒来后一个 chunk 都没等到就走到这里（例如同批的 finish_task），也算数。
        await _commit_round(state, ctx)

        # 路由优先级：suspend > 上下文恢复 > observe。
        #
        # - suspend 最先：suspend_requested 表示 actor 本轮派了活、本 run 要停在等子任务 →
        #   SuspendStep（判据从 `task.status == "SUSPENDED"` 换成意图位：状态归 TaskManager，
        #   Task 4）。越线也不例外——父的上下文由 `_maybe_predispatch_compact` 那条路负责，
        #   而这个 run 本来就要停了，没有「续跑」可言。
        # - 恢复次之：越线但还救得回来 → 回 PrepareStep。为什么是 prepare 而不是已注册的
        #   "compact" 步：① `CompactStep.execute` 返回 next_step=None，会直接终结 driver 的
        #   循环（既无 observe 也无 finalize，run_outcome 是 None）；② `escalating_compact`
        #   **从不回写** loop_guard.context_tokens（est 是局部量），所以 compact → prepare
        #   会让 prepare 用压缩前的陈旧基线再开一次门、白跑第二次压缩；③ prepare 顺带做了
        #   续跑必须做的两件事——resolve_and_bind 重解能力面 + `_assemble()` 重建
        #   assembled_prompt（本步的 current_messages 正是从它来的）。
        if state.task.suspend_requested:
            next_step = "suspend"
        elif exit_reason == "context_limit" and _can_recover_context(state):
            state.extra[CONTEXT_RECOVERY_COUNT_KEY] = (
                state.extra.get(CONTEXT_RECOVERY_COUNT_KEY, 0) + 1)
            next_step = "prepare"
        else:
            next_step = "observe"

        # ── task.outputs：收尾时合成最终交付物（spec 2026-07-01 反转契约）──
        # 收尾路径 = 纯文本收尾(normal) 或 finish_task 收尾(actor_done 且未挂起)；答复即消息正文，
        # finish_task 的 deliverables_summary 为可选产出小结。max_turns / context_limit /
        # delegate-suspend 不在此列（不产最终输出，维持现状）。
        if exit_reason in ("normal", "actor_done"):
            _synthesize_final_outputs(state, transcript)

        # ── finish_task 收尾也让位（S-b，2026-09-27）────────────────────────────
        # `actor_done` 走到这里只有一个来源：`finish_task`（delegate 那条置了
        # `suspend_requested`，上面已经路由去 suspend 了）。它此前在 root 上是**零复核**的
        # ——`observe._should_use_llm` 对 `parent_task_id is None` 降级走机械判决，而
        # `_mechanical_verdict` 把 `actor_done` 无条件映射成 success，那道
        # success-without-outputs 护栏又长在 `report_task_outcome` 里、不在机械判决的路上。
        # 于是「说了段话要被判、明确宣布做完了反而不被判」，等于给 LLM 留了一个绕开复核的
        # 开关。改成与纯文本同一条归宿：park + 后台判定。
        #
        # **先只对 root 生效**（`parent_task_id is None`），与 S5 同一个收窄办法：子任务的
        # `finish_task` 今天走前台 LLM observe，那条路有真复核、`FinalizeStep` 也照常跑，
        # 先别动。代价是同一个子任务两种收尾走两条路，观察一段之后再决定要不要推广。
        #
        # 三条判据与 `_finish_plain_text_turn` 逐字相同（`NormalTaskSettings` / 非 unattended
        # / hitl 在），只多两条：root，以及**这个 agent 得真有 observer**。park 会抛
        # `HitlPark`，下面那个 StepOutcome 到不了；不让位则零副作用返回，照旧走 observe。
        #
        # `has_observe_role` 那条（2026-09-28，随 default ROLE 兜底一起加）：这个 park 的
        # 全部意义就是「先别收尾，等 observer 复核」，而 park 之后**前台 ObserveStep 跑不到**
        # ——`HitlPark` 在 `runtime._run_loop` 那里 unwind 的是 `driver.run` 整个循环。于是
        # park 之后系统里唯一的判决来源就是后台那次带外判决；模板没有 ROLE 时它降成只摘要
        # 档（`background.recap` 里 `judges` 与 `has_observe_role` 相与），没人再终结这个
        # task，root 就永远停在 AWAITING_HUMAN。
        #
        # 所以没有 observer 时**压根不让位**：走下面那个 `StepOutcome` 回 observe，机械判决
        # 拿 `actor_done` 判 success，`FinalizeStep` 照常跑，后台照常产 recap
        # （`boundary="finish"`）——即 S-b 之前那条本来就在的路。
        #
        # 位置在 `_synthesize_final_outputs` **之后**：park 的 task 恒 `outputs=None` 会让
        # 护栏把 success 改判 retry（2026-09-24 那个坑，见该函数 docstring）。`_commit_round`
        # 也已经在本轮第一个 chunk 就跑过了，与纯文本 park 同样是「已提交之后才 park」。
        if (
            next_step == "observe"
            and exit_reason == "actor_done"
            and state.task.parent_task_id is None
            and has_observe_role(state)
            and isinstance(state.task.settings, NormalTaskSettings)
            and not state.task.unattended
            and ctx.hitl is not None
        ):
            # 这一回合会发出**第二条** `ACT_TURN_COMPLETED`（先 `tool_calls_processed`、
            # 再 `await_user`）。刻意的：两者不矛盾——工具确实处理完了，然后这一回合以让位
            # 收尾。别为了「一回合一条」把哪一条删掉，那条纪律来自纯文本路径（那里
            # `await_user` 与 `stop` 是互斥的两个结局），不适用于这里。
            # 轮号取 `state.extra` 而不是循环变量：这里在 for 之外，靠 `turn_num` 泄漏出来
            # 虽然眼下不会 unbound（循环一次都没跑意味着 `exit_reason == "max_turns"`，走不到
            # 这条分支），但那是两处代码的巧合，不是这一行自己成立。
            await _park_await_user(
                state, ctx, state.extra.get(ACT_TURNS_USED_KEY, 0), boundary="finish_park")

        return StepOutcome(
            next_step=next_step,
            state_patch={
                "transcript": transcript,
                "act_exit_reason": exit_reason,
            },
        )


def _compose_final_outputs(transcript: list[TurnRecord]) -> tuple[str, str]:
    """拆出收尾交付物的两段 `(body, summary)`（spec 2026-07-01）。

    body = 收尾回合(transcript[-1])正文；空则回溯本段最近一段非空 assistant_text（兼容模型把
    答复写在上一回合、收尾回合只调 finish_task 的情况）。summary = 收尾回合 finish_task 调用的
    deliverables_summary（可空）。两段都空 → `("", "")`，交给 observer 护栏。

    **只拆不拼**：拼接是调用方的事。两段语义不同（body 是答复本身，summary 是给 reviewer 的
    交付物清单），下游有需要分开的消费者（TASK_FINALIZED 事件），在这里拼死就再也分不开了。
    """
    last = transcript[-1]
    body = (last.assistant_text or "").strip()
    if not body:
        body = next(
            (t.assistant_text.strip() for t in reversed(transcript)
             if (t.assistant_text or "").strip()),
            "",
        )
    summary = ""
    for tc in last.tool_calls:
        if tc.name == FINISH_TASK_NAME:
            val = (tc.arguments or {}).get("deliverables_summary", "")
            summary = val.strip() if isinstance(val, str) else ""
            break
    return body, summary


async def _stream_until_stop(agen: Any, ctx: LoopContext) -> "AsyncIterator[Any]":
    """按 chunk 迭代 LLM 流；**等待下一个 chunk 的过程中**也能被软打断/硬取消掐断。

    改造前这里是裸 `async for`，暂停检查写在循环体里——只有 chunk 到达才执行得到。
    于是请求发出到首个 chunk 之间（TTFT；思考模型、provider 排队、冷路由都能到几十秒）
    协程挂在 `__anext__` 上，`pause_token` 被置位也没有任何人读它：用户按下暂停毫无反应。
    退避重试期间更长——gateway 的自愈最多 300s，adapter 自己的 HTTP 重试还能各睡 60s，
    那两段同样一个 chunk 都不产出。

    这里改成与 `_await_tool_or_stop` **同一形状**（那条路早就是对的：在跑的工具能被当场
    掐掉）：把 `__anext__` 和两个停止信号赛跑。

    停止信号胜出 → 取消挂起的 `__anext__`。那个 `CancelledError` 会打进生成器内部当前
    挂着的 await —— 包括 adapter 退避里的 `asyncio.sleep` 与 httpx 的
    `async with client.stream(...)`，后者退出上下文即**真正掐断这次请求**，不再空烧
    token。`CancelledError` 是 `BaseException`，gateway 的 `except LLMCallError` 与
    adapter 的各处 `except Exception` 都拦不住它（已核实两处均无裸 `except`）。

    本函数**不判断是 pause 还是 cancel**，只负责停下来并收干净——分流由调用方在循环
    之后做，判据仍是那两个 token 本身，不引入第二处真相。
    """
    waiters: list[asyncio.Future] = []
    if ctx.cancel_token is not None:
        waiters.append(asyncio.ensure_future(ctx.cancel_token.wait()))
    if ctx.pause_token is not None:
        waiters.append(asyncio.ensure_future(ctx.pause_token.wait_paused()))
    try:
        while True:
            nxt = asyncio.ensure_future(agen.__anext__())
            if waiters:
                await asyncio.wait({nxt, *waiters}, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.wait({nxt})
            if not nxt.done():
                # 停止信号先到：掐断挂起的取流，等它真的收完尾再走。
                nxt.cancel()
                await asyncio.wait({nxt})
                return
            try:
                chunk = nxt.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        for w in waiters:
            w.cancel()
        if waiters:
            await asyncio.wait(set(waiters))
        # 生成器可能仍挂在某个 yield 上（调用方 break 出去时）：关掉它，让 httpx 的
        # 流上下文退出。已经被上面的 cancel 拆完的生成器对此是 no-op。
        await agen.aclose()


async def _commit_round(state: LoopState, ctx: LoopContext) -> None:
    """提交本轮：LLM 开口了，这一轮算数（spec 2026-09-09）。**幂等**，每个 chunk 都调。

    在此之前，这一轮的两样东西还没落地：

    1. `TASK_CREATED` / `TASK_STARTED` / `RUN_STARTED` / `LLM_PROMPT_SENT` 攒在总线的
       未提交窗口里（只到达了进程内状态机，见 `EventBus` 类 docstring）；
    2. 把这一轮唤醒的那条 HITL 答复还停在**待终局**（`pending_decision`），
       `HitlResolved` 一直没发——日志里那个气泡仍是 pending。
    3. 这一轮的 memory 写入（用户消息、HITL 答复、由答复回灌的工具结果）暂存在窗口里
       （`ingest_or_stage`），memory 里还没有它们。提交时钩子先发 `HitlResolved`、再按序
       落盘——**终局事实先于答复进 memory**。
    4. `recognize_intent` 还没起飞（判定在 `PrepareStep`，起飞在这里）——旁路只该为
       **真的发生过**的那一轮花一次 LLM 调用。它跑在 root task 上，而 root task 从第二句
       起每一轮都开窗，放在 prepare 起飞就会在窗口里跑、被撤销时整串事件白丢。

    暂存的记录压缩够不着（L0.5 图片降级 / L1 / L3 只折 memory）。所以 PrepareStep 判定要
    压缩时**先提交**再压缩——代价是那一轮不再可撤，换来带图的第一条消息照样能被降级
    （`test_media_fold_replay_e2e` 钉住）。act 离开循环去别的步骤之前同样先提交：那些步骤
    直接读 memory，而这一轮已经推进到了别处，算数。

    顺序不可换：`TASK_CREATED` 必须最先出闸，它是下游 reducer 与 host 建 task 键的那一条；
    `recognize_intent` 必须在它之后起飞，它会发自己的一串事件，且要写 `task.title/description`
    ——那两个字段正是窗口里 `TASK_CREATED` 的 payload 在出闸那一刻现算的来源。
    """
    if state.extra.get(ROUND_COMMITTED_KEY):
        return

    # 两阶段终局的第二阶段（那条把这一轮唤醒的答复——HITL 应答 / `send_message` 对旧
    # 气泡的收口——现在才算数、`HitlResolved` 在此刻才发）由 `commit_round` 的钩子完成，
    # 见 `TaskManagerHooks.commit_round`。
    tm = ctx.task_manager
    if tm is not None:
        await tm.commit_round(state.task.id)
    # **成功之后才置位**：提交失败（存储不可用）时异常上抛、标志保持 False，下一个提交点
    # （或 `_run_task` 的收尾兜底）会重试，而不是以为已经提交过了。
    state.extra[ROUND_COMMITTED_KEY] = True
    # 这一轮算数了，它就不再是「可热撤销的一轮」。
    state.extra.pop(HOT_REPLY_ROUND_KEY, None)

    if state.extra.pop(RECOGNIZE_INTENT_PENDING_KEY, False):
        from ctx_weft.core.loop.steps.recognize_intent import launch_recognize_intent
        launch_recognize_intent(state, ctx)



async def _discard_round_if_uncommitted(state: LoopState, ctx: LoopContext) -> None:
    """用户在 LLM 开口之前按了停 → 整轮丢弃（spec 2026-09-09）。已提交则原样返回。

    只对**由一条用户消息新开出来的 task**（`send_message` 的新建分支，`TaskManager`
    给它开了未提交窗口）成立。注入既有 task 的那两条路径（agent 本就在
    `AWAITING_HUMAN` / 挂起等子任务）不在此列——那个 task 早就提交过、是一段正在进行的
    对话，丢不得；它们照常 park 出续跑气泡，只是那条刚注入的用户消息同样还没落 memory，
    所以照样零残留。

    本函数只抛信号：`RoundDiscarded` 一路 unwind 到 `_run_task`，由 `discard_round` 收尾。
    memory 这边无事可做——这一轮的用户消息 / 答复只在窗口的暂存区里（`ingest_or_stage`），
    随窗口一起扔掉。

    **一条事件都不发**：task 状态事件的发射点只有 TaskManager 一处（Task 4 的不变式，
    `test_task_manager_owns_status` 有静态守卫盯着）。把 agent 送回 `idle` 的那条
    `TASK_CANCELED`、以及关窗丢弃，都在 `TaskManager.discard_provisional` 里按正确顺序完成。
    """
    if state.extra.get(ROUND_COMMITTED_KEY):
        return
    tm = ctx.task_manager
    if tm is None or not tm.is_round_open(state.task.id):
        return

    hot_hitl_id = state.extra.pop(HOT_REPLY_ROUND_KEY, "")
    if hot_hitl_id:
        # **热撤销**：这一轮是一条热应答开出来的——同一个 run 醒来接着跑，还没等到 LLM
        # 开口就被暂停了。不能整轮 `RoundDiscarded`：这个 run 的 `RUN_STARTED` 早已落盘，
        # 连它的结尾一起丢掉，日志里就是一个永不结束的 run。
        #
        # 于是分两半：答复这一半撤掉（答复退回待答；缓冲的事件与暂存的工具结果随窗口丢弃，
        # memory 里本来就没有），run 这一半照常 park 在那个问题上——正是「人还没答」的世界，
        # `RUN_FINISHED` / `TaskAwaitingHuman` 如实落盘，重答走冷路径。
        await tm.abandon_round(state.task.id)
        req = ctx.hitl.registry.get(hot_hitl_id) if ctx.hitl is not None else None
        raise HitlPark(hitl_id=hot_hitl_id,
                       tool_call_id=req.tool_call_id if req is not None else "")

    raise RoundDiscarded(state.task.id)


@dataclass
class _LLMTurnOutput:
    """单轮 LLM 流式产出。"""

    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    usage: LLMUsage
    # 摄入锚（预铸的 assistant 记录 id，ingest 按 MemoryEvent.id 采纳）与铸造伴随。
    # tool_calls 内的 id 已是内部标识；minted 保留原始 wire id / ordinal 供落库伴随字段。
    anchor: str = ""
    minted: list[MintedCall] = dataclasses.field(default_factory=list)
    provider_state: dict[str, Any] = dataclasses.field(default_factory=dict)


async def _run_llm_turn(
    state: LoopState, ctx: LoopContext, prompt: Any, current_messages: list[LLMMessage],
    turn_num: int, baseline_msg_count: int | None = None,
) -> _LLMTurnOutput:
    """发请求事件 → 流式累积 token/reasoning/tool_calls/usage → 处理软打断 → 发 RESPONSE_FINISHED。

    软打断（pause）时提交半截 assistant 文本并 park（raises HitlPark），不返回；硬取消 →
    CancelledError。正常结束返回 _LLMTurnOutput。
    """
    agent = state.agent
    model, llm_account = resolve_llm_identity(state)
    # req_id: 与 llm_gateway.stream_llm_resilient 内部用同一个确定性公式独立算出——两边都在
    # 「本次 LLM 调用的任何事件被发射之前」求值（此处在调用 stream_llm_resilient 之前；gateway
    # 侧在其 while 重试循环、也就是第一次 emit 之前），故 state.sequence_counter 两处读到同一个
    # 值，无需新增参数或跨函数传值即可对齐（详见 llm_gateway.stream_llm_resilient 文档字符串）。
    req_id = f"req_{agent.id}_{state.sequence_counter}"

    llm_request = LLMRequest(
        model=model, system=prompt.system, messages=list(current_messages), tools=prompt.tools)
    llm_request.prompt_token_estimate = request_prompt_estimate(
        ctx.llm.tokenizer, llm_request, getattr(agent, "loop_guard", None), baseline_msg_count)
    # turn：LLM_REQUEST_STARTED / LLM_PROMPT_SENT 现由 gateway 发射，但 turn_num 是本函数的局部
    # 循环变量、state 上没有对应字段——经 metadata 这个既有的瞬态透传通道带给 gateway（同
    # PROMPT_EST_BASE_KEY/PROMPT_EST_SEG_KEY 的做法），不新增函数参数。
    llm_request.metadata["turn"] = turn_num

    interrupted = False
    stream = ensure_collected_stream(stream_llm_resilient(ctx, state, llm_request))

    try:
        async for chunk in _stream_until_stop(stream, ctx):
            tok = ctx.cancel_token
            if _interrupt_pending(ctx):
                interrupted = True
                break
            if tok is not None and tok.is_cancelled:
                tok.raise_if_cancelled()
            # Preserve the original commit/accept boundary, including tool heartbeats.
            await _commit_round(state, ctx)
            if chunk.kind == "token":
                ctx.run_phase.produced = True
            stream.accept_current()
    finally:
        await stream.aclose()

    result = stream.partial_result()
    text, reasoning, usage = result.text, result.reasoning, result.usage
    tool_calls = list(result.tool_calls)

    # 流在**首个 chunk 到达之前**被停止信号掐断时，上面的循环体一次都没执行到——
    # 判定必须补在循环之后，否则一次 TTFT 窗口里的暂停会被当成「流正常结束」。
    if not interrupted and _interrupt_pending(ctx):
        interrupted = True
    tok = ctx.cancel_token
    if tok is not None and tok.is_cancelled:
        tok.raise_if_cancelled()

    if interrupted:
        # ⓪ 一个 chunk 都没到就被中止 → 整轮丢弃，不 park、不留任何痕迹。
        await _discard_round_if_uncommitted(state, ctx)   # 命中则 raises RoundDiscarded
        # ② 已吐 token：把半截 assistant 文本入 memory 并标注「被用户打断」；
        # ① 已开口但本回合没吐正文（纯工具调用后被打断）：不留记录。随后 park 待用户续接。
        has_partial = bool(text.strip() or reasoning.strip())
        await _commit_interrupted_partial(state, ctx, text, reasoning, turn_num)
        if _is_own_root(state.task):
            from ctx_weft.core.loop.background import launch_recap
            launch_recap(state, ctx, boundary="interrupt")
        await _park_for_interrupt(state, ctx, edit=not has_partial)

    # 摄入前铸造内部调用标识（spec: conversation-integrity）：LLM 复用 wire id（call_1
    # 类短值）是常态，跨轮次/跨任务重建后按裸 id 配对会错配。本函数是唯一供值点——
    # 事件 payload、memory 记录、current_messages、gateway/HITL 全部消费同一次铸造的
    # 值。锚 = 预铸的 assistant 记录 id（ingest 经 MemoryEvent.id 采纳，见
    # _ingest_assistant_turn），跨平面一致且与 tool_call_id 派生同源。observe 的
    # run_observe_react 对自己的回合同口径（live 平面，不入 memory）。
    anchor = generate_id("asst")
    minted: list[MintedCall] = []
    if tool_calls:
        minted = mint_turn_call_ids(tool_calls, anchor=anchor, turn_seq=turn_num)
        tool_calls = [m.call for m in minted]

    # token 自校准回喂：真实 usage 与发送前估算段作比（基线不参与），喂给该模型 tokenizer。
    # 估算段取 PROMPT_EST_SEG_KEY（tokenizer.count 直接产出）而非
    # prompt_token_estimate − base——整份路径上返回值可能被 max(full, ctx_tokens) floor 成
    # 上一轮真实值，拿 floor 后的值回喂会产出 ratio≈1 的假样本，把伺服系统性拖向 1。
    base = llm_request.metadata.get(PROMPT_EST_BASE_KEY)
    est_seg = llm_request.metadata.get(PROMPT_EST_SEG_KEY)
    if usage.prompt_tokens > 0 and est_seg is not None and base is not None:
        ctx.llm.tokenizer.observe(est_seg, usage.prompt_tokens - base)
        # spec: tool-schema-budget——偏差留痕（SHALL）：工具面分项（指纹/估算）与实测同现，
        # 供校准与回归分析（R4「偏差可归因」）。
        guard = getattr(agent, "loop_guard", None)
        logger.info(
            "token estimate feedback: est_seg=%s base=%s actual_prompt=%d "
            "tools_signature=%s tools_est=%d",
            est_seg, base, usage.prompt_tokens,
            getattr(guard, "last_tools_signature", ""),
            getattr(guard, "last_tools_est", 0))

    await ctx.event_bus.emit(make_event(
        state, EventType.LLM_RESPONSE_FINISHED,
        payload={
            "request_id": req_id, "content": text, "reasoning": reasoning,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ],
            "usage": dataclasses.asdict(usage),
            # 本次调用实际使用的模型/账号（host 云端上报按此计账，不再受切换竞态影响）
            "llm_model": model, "llm_account": llm_account,
            "finish_reason": "tool_use" if tool_calls else "stop", "turn": turn_num}))

    return _LLMTurnOutput(
        text=text, reasoning=reasoning, tool_calls=tool_calls, usage=usage,
        anchor=anchor, minted=minted, provider_state=result.provider_state)


def _can_recover_context(state: LoopState) -> bool:
    """越过上下文停机线之后，该不该回 PrepareStep 压缩续跑（而非退回 observe 吃一次 retry）。

    两个否决条件都必须有，各管一种失败模式：

    - **配额**（`max_context_recoveries`，0 = 关闭恢复）：compact 压得动一点、但离 target
      总是差一口气时，prepare↔act 会来回 ping-pong，每圈烧一次真实 act LLM 调用 + 一到两次
      摘要调用。配额是这条的硬止损。
    - **compact 已压不动**（`COMPACT_NOOP_KEY`）：`escalating_compact` 的每一级都有可折性
      guard，全 noop 时它返回的事件里一条 `MEMORY_COMPACTED` 都没有——说明这个 scope 再压
      也挤不出东西（例如 task 层只剩 ≤ collapse_keep_last 条、agent 层没有已完成胶囊、
      单轮塞进来一条超大 tool result）。此时再恢复一次纯属白烧，立即放弃，不等配额耗尽。

    ⚠️ 这个否决只在 `used > 0`（即已经恢复过至少一次）时生效。**本 run 开头那次 prepare 的
    compact 也会写这个标志**，而那一次几乎必然是 noop——run 刚起，task 层往往只有一条
    USER_PROMPT，各级 guard 全不满足。若不区分，那面「压不动」的旗会在 act 还没发出第一个
    请求时就立起来，把整个 run 的恢复资格一笔勾销（实测：`prepare→act→observe`，恢复从不
    发生）。所以第一次恢复只受配额约束——想知道压得动压不动，总得先真压一次。
    """
    limit = getattr(state.agent.loop_config, "max_context_recoveries", 0)
    if limit <= 0:
        return False
    used = state.extra.get(CONTEXT_RECOVERY_COUNT_KEY, 0)
    if used >= limit:
        return False
    return not (used > 0 and state.extra.get(COMPACT_NOOP_KEY))


async def _account_tokens(state: LoopState, ctx: LoopContext, usage: LLMUsage) -> bool:
    """更新 LoopGuard 基线 + session.token_used；返回是否越过上下文停机线。

    阈值取 `loop_config.context_limit_stop_ratio`（默认 0.9），**曾硬编码 0.8**——那个值
    与 `compact_token_ratio` 的默认值重合，于是 prepare 刚把估算压到「刚低于 0.8」，本步
    第一轮就又越线，一次恢复配额当场蒸发。两条线现在分工：compact_token_ratio 管 run 开头
    的常规压缩（估算口径），本比率管 mid-act 兜底（真实 usage 口径），故设得更高。

    对齐 miniAgents actor.py context_limit_hit 逻辑。
    """
    agent = state.agent
    if usage.prompt_tokens > 0:
        agent.loop_guard.context_tokens = usage.prompt_tokens
        try:
            from ctx_weft.protocols import MemoryKind, MemoryScope

            # v2 P3a：len(视图谓词) 取代 count_recent（= 旧 USER_PROMPT+LLM_RESPONSE 口径，
            # 与 prepare._estimate_tokens 对齐）。
            view = await ctx.memory.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
            agent.loop_guard.context_message_count = sum(
                1 for r in view
                if r.kind is MemoryKind.CONVERSATION_TURN and r.role in ("user", "assistant")
            )
        except Exception as exc:
            # best-effort 计数：失败不影响主流程，但记 debug 便于排查（不静默吞）。
            logger.debug("load_view count for loop_guard failed: %s", exc)

    # 累加 session.token_used（供 token_budget 检查使用）
    state.session.token_used += usage.prompt_tokens + usage.completion_tokens

    context_limit = agent.loop_guard.context_limit
    reserve = getattr(agent.loop_guard, "reserved_output_tokens", 0)
    eff = effective_limit(context_limit, reserve)
    ratio = getattr(agent.loop_config, "context_limit_stop_ratio", 0.8)
    return (
        eff > 0
        and usage.prompt_tokens > 0
        and usage.prompt_tokens >= int(eff * ratio)
    )


@dataclasses.dataclass
class PersistedAssistantTurn:
    """`_ingest_assistant_turn` 的结构化返回（spec: tool-operations，wp5）。

    ``record_id`` 是 LLM_RESPONSE 回合的 memory 记录 id——tool_call_id 的派生输入之一
    （确定性派生 → 同一逻辑调用跨重启同 id）。``tool_calls`` 直通旧形态（message
    重建消费点零语义变化）。
    """

    record_id: str
    tool_calls: list[dict]


async def _ingest_assistant_turn(
    state: LoopState, ctx: LoopContext, text: str, tool_calls: list[ToolCall],
    usage: LLMUsage, turn_num: int,
    anchor: str = "", minted: list[MintedCall] | None = None,
) -> PersistedAssistantTurn:
    """把本轮 assistant 回合入 task 层 memory；返回结构化（record_id + tool_calls）。

    派发(submit_*)/silent 工具的 tool_call 排除出 LLM_RESPONSE.metadata（派发落 agent 层
    delegate conversation turn、silent 结果不入对话），避免无配对 TOOL_RESULT 的悬挂调用破坏
    无损重建（spec 2026-06-28 §2.3）。

    anchor（spec: conversation-integrity）：预铸的记录 id，经 MemoryEvent.id 交给 provider
    采纳——它同时是内部调用标识与 tool_call_id 的派生锚，三者由此同源。minted 携带每个
    调用的 raw wire id 与 ordinal，随 metadata 落库（raw_tool_call_id 供追溯）；
    两者缺省（旧测试直调 / 无工具回合）时退化为无伴随字段的旧行为。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS, SILENT_TOOLS
    by_ordinal = {m.ordinal: m for m in (minted or [])}
    asst_tool_dicts: list[dict] = []
    for i, tc in enumerate(tool_calls):
        d: dict = {"id": tc.id, "name": tc.name, "input": tc.arguments}
        m = by_ordinal.get(i)
        if m is not None:
            d["raw_tool_call_id"] = m.raw_id
        asst_tool_dicts.append(d)
    _excluded = DISPATCH_TOOLS | SILENT_TOOLS
    non_dispatch_tool_dicts = [d for d in asst_tool_dicts if d["name"] not in _excluded]
    # 经 `ingest_or_stage`：正常路径上这一轮早已提交（首 chunk 就是提交点），分流直写；
    # 唯一走暂存的是「LLM 流一个 chunk 都没吐就正常结束」——那时窗口还开着，直写会让
    # memory 里出现一条属于「还没算数的一轮」的 assistant 回合（崩溃后它甚至可能属于一个
    # 日志里不存在的 task）。id 是预铸的锚，暂存安全。
    from ctx_weft.core.loop.driver import ingest_or_stage
    # 锚缺省（旧测试直调）时自己铸一个：暂存需要一个稳定 id，而落库返回值这条路上
    # 已经没人再给了。生产路径恒有锚（`_run_llm_turn` 预铸）。
    record_id = anchor or generate_id("asst")
    await ingest_or_stage(
        ctx.memory,
        MemoryEvent(
            id=record_id,
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            content=text,
            timestamp=now_utc(),
            role="assistant",
            metadata={
                "turn": turn_num,
                "tool_call_count": len(tool_calls),
                "tool_calls": non_dispatch_tool_dicts,
                "usage": dataclasses.asdict(usage),
            },
        ),
        ctx.provider_ctx,
        task_manager=getattr(ctx, "task_manager", None),
        # 鸭子类型的 state 替身（单测直调本函数）可能没有 task。
        task_id=getattr(getattr(state, "task", None), "id", ""),
    )
    return PersistedAssistantTurn(record_id=record_id or "", tool_calls=asst_tool_dicts)


async def _maybe_predispatch_compact(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall], usage: LLMUsage,
) -> None:
    """本轮含派发调用且越过 predispatch 阈值 → 派发执行前先走一遍 compact（同 CompactStep）。

    在 gateway 写 delegate conversation turn / 子 spawn-inherit 之前完成，使子继承到压缩后的记忆；每轮至多一次。
    随后 dispatch 执行（父转 SUSPENDED）→ 本 Act 末尾路由到 SuspendStep。usage.prompt_tokens 是本轮真实计数。

    task-4 复审修复：driver 只在**步骤边界**切换 state.origin，此刻仍是 Act 步骤执行期间、
    origin 还是 LOOP_ACT——但接下来经 maybe_compact_before_dispatch → escalating_compact →
    summarize_for_compact 发起的这次 LLM 调用实际是 compact 摘要，不是 act 的一次 LLM turn。
    不临时切换的话，llm_gateway.stream_llm_resilient 的 origin 门禁会把它错当成 act 调用发出
    LLM_REQUEST_STARTED/LLM_PROMPT_SENT（turn 还取不到值恒为 0），escalating_compact 内部
    make_event(state, ...) 发的 MEMORY_COMPACT_* 也会被错标成 LOOP_ACT。这里临时切到
    LOOP_COMPACT、调用结束（含异常路径）后用 finally 还原，使这次调用的所有事件来源标注
    正确，也走上 summarize_for_compact 里新补的收尾事件。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS
    if not any(tc.name in DISPATCH_TOOLS for tc in tool_calls):
        return
    from ctx_weft.core.loop.steps.compact import maybe_compact_before_dispatch
    prev_origin = state.origin
    state.origin = EventOrigin.LOOP_COMPACT
    try:
        events = await maybe_compact_before_dispatch(state, ctx, prompt_tokens=usage.prompt_tokens)
    finally:
        state.origin = prev_origin
    for ev in events:
        await ctx.event_bus.emit(ev)


async def _execute_tool_calls(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall],
) -> list[dict]:
    """逐个经 CapabilityGateway 执行 tool calls，返回 tool_results。

    工具间 / 执行中命中软打断 → 给本 tc 及之后补「已取消/被打断」result + park（raises HitlPark）；
    硬取消 → CancelledError；被 park 的工具 → task 转 SUSPENDED 并上抛（spec/07 §7）。
    """
    tool_results: list[dict] = []
    ctx.run_phase.in_tool_loop = True

    for i, tc in enumerate(tool_calls):
        # 账本键就是 tc.id（摄入点铸造的内部标识），gateway 自己从 tool_call_id 取——
        # 调用侧不再铸第二条身份、也不再经共享 provider_ctx 转移所有权。
        # 工具间命中：软打断 → 本 tc 及之后全部「未开始」→ 补「已取消」+ park；硬取消 → CancelledError。
        if _interrupt_pending(ctx):
            # 这一轮若还没提交（热应答醒来、到下一个提交点之前），暂停就是撤回那条答复：
            # 走 `_discard_round_if_uncommitted`（热撤销 → park 在原问题上，抛出）。剩下的
            # 工具调用**不补**结果——人重答之后 reconcile 会把它们当作从未开始的调用执行。
            await _discard_round_if_uncommitted(state, ctx)
            for rest in tool_calls[i:]:
                await _ingest_synthetic_tool_result(state, ctx, rest, CANCELLED_MARK, cancelled=True)
            ctx.run_phase.in_tool_loop = False
            if _is_own_root(state.task):
                from ctx_weft.core.loop.background import launch_recap
                launch_recap(state, ctx, boundary="interrupt")
            await _park_for_interrupt(state, ctx)
        if ctx.cancel_token is not None and ctx.cancel_token.is_cancelled:
            ctx.cancel_token.raise_if_cancelled()

        # 可取消地执行在途工具：执行中被打断 → 取消该工具、补「被打断」result。
        invoke_task = asyncio.ensure_future(_invoke_tool(tc, state, ctx))
        completed = await _await_tool_or_stop(invoke_task, ctx)
        if not completed:
            if _interrupt_pending(ctx):
                # 同上。被打断的这个工具若已过 gateway step 5，这一轮已提交，下面这句 no-op；
                # 没过 step 5 就意味着 provider 从未被调用，当作没开始即可。
                await _discard_round_if_uncommitted(state, ctx)
                await _ingest_synthetic_tool_result(state, ctx, tc, INTERRUPTED_MARK, interrupted=True)
                for rest in tool_calls[i + 1:]:
                    await _ingest_synthetic_tool_result(state, ctx, rest, CANCELLED_MARK, cancelled=True)
                ctx.run_phase.in_tool_loop = False
                if _is_own_root(state.task):
                    from ctx_weft.core.loop.background import launch_recap
                    launch_recap(state, ctx, boundary="interrupt")
                await _park_for_interrupt(state, ctx)
            if ctx.cancel_token is not None:
                ctx.cancel_token.raise_if_cancelled()  # 硬取消
        try:
            result = invoke_task.result()
        except HitlPark:
            # 被 park 的工具未执行，直接 unwind（spec/07 §7）；task 落 AWAITING_HUMAN
            # 由 TaskManager 据 RunOutcome 定（Task 4：这里原先那句 `task.status =
            # "SUSPENDED"` 是写给自己看的死值，_run_loop 的 HitlPark 支随即覆盖）。
            raise

        tool_results.append({
            "tool_call_id": tc.id,
            "name": tc.name,
            "result": result.content,
            "is_error": result.is_error,
        })

    ctx.run_phase.in_tool_loop = False
    return tool_results


def _reconcile_finish_vs_dispatch(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall],
) -> None:
    """finish_task 与 delegate_task / delegate_plan 同批出现时仲裁：finish 胜出。

    两类工具语义互斥——一个要当前 task 收尾(→observe)，一个要它挂起等子任务(→suspend)。
    用户意图是「我做完了，顺手派生独立后续」：故 finish 胜出，被派发任务从「当前 task 的
    阻塞子任务」改投为「当前 task 的 parent 名下的独立后继」(当前是 root 则为顶层)，自行调度。

    - detach_staged：把本轮 staged 子任务改挂到 parent，切断与收尾 task 的阻塞链。
    - 清 suspend_requested：撤销 delegate 置的挂起意图，使路由走 observe(task.outputs
      已由 finish_task 写好)。
    - 清 spawn_titles：避免 SuspendStep 误报(虽已不路由到 suspend，仍清掉防脏状态)。

    顺序无关：只看本批最终是否两类工具都出现。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS
    if ctx.task_manager is None:
        return
    names = {tc.name for tc in tool_calls}
    if FINISH_TASK_NAME not in names or not any(n in DISPATCH_TOOLS for n in names):
        return
    ctx.task_manager.detach_staged(state.task.id, state.task.parent_task_id)
    state.task.suspend_requested = False
    if isinstance(state.task.settings, NormalTaskSettings):
        state.task.settings.spawn_titles = []
    logger.info(
        "act: finish_task + dispatch in same batch on task %s — finishing it; "
        "detaching delegated work to parent %s",
        state.task.id, state.task.parent_task_id,
    )


def _synthesize_final_outputs(state: LoopState, transcript: list) -> None:
    """把收尾回合的正文合成 `task.outputs`（幂等，可重复调用）。

    收尾路径 = 纯文本收尾(normal) 或 finish_task 收尾(actor_done 且未挂起)；答复即消息
    正文，finish_task 的 deliverables_summary 为可选产出小结。max_turns / context_limit /
    delegate-suspend 不在此列（不产最终输出）。

    **必须能在 park 之前被调到**（2026-09-24 修）：park 走 `_cold_park` 抛 `HitlPark`
    释放协程，`ActStep.execute` 就地中断——循环之后那次合成永远执行不到。于是 park 的
    task 恒 `outputs=None`，而 `report_task_outcome` 的 success-without-outputs 护栏据它
    把 success 改判 retry，**park 的 task 因此永远不会被判成功终结、DAG 后继永远放不行**。
    日志里的形态是 `arguments={'task_status':'success'}` 而
    `result="Assessment recorded: outcome=retry."`。
    """
    # getattr 防御：手构的替身 task 未必带这个字段（本函数现在也从 park 路径被调，
    # 而那条路上的单测惯用 SimpleNamespace）。
    if getattr(state.task, "suspend_requested", False) or not transcript:
        return
    body, summary = _compose_final_outputs(transcript)
    # 拼接留在这里：`task.outputs` 的既有契约（**拼好的单串**）一个字不变——它有 6 处
    # 读取方（_build_memory_content / background_observe / finalize 的 final_reply 等），
    # 改形态会把这些一并掀翻。
    outputs = f"{body}\n\n{summary}" if (body and summary) else (body or summary)
    if outputs:
        state.task.outputs = outputs
    # 两段另存一份：TASK_FINALIZED 需要它们**分开**出核——output 是交付物本身，summary
    # 是 agent 给 reviewer 的自评清单。混在一起正是 host 侧打印「最终答复」时把自评清单
    # 当答案一起打出来的成因。
    state.extra["final_body"] = body
    state.extra["final_summary"] = summary


async def _finish_plain_text_turn(
    state: LoopState, ctx: LoopContext, turn_num: int, transcript: list,
) -> None:
    """纯文本回合（无 tool call）收尾。

    **有人在场**就让位 → `_park_await_user`（HITL input 冷 park、raises HitlPark；park
    与后台 observe 并行，observer 判 success 则经带外入口终结本 task、判 retry/fail 则
    维持 park 等人）。unattended / 非普通任务 / 无 hitl：纯文本即任务产出，发 stop 事件
    路由前台 observe。

    判据 2026-09-22 从 `interaction_mode == "interactive"` 换成 `not unattended`：
    「这段话是想问人还是交付完了」是**逐回合**的语义，不该由 task 上的一个静态字段
    统一回答——那样 interactive 的 task 里 agent 交付完了也得 park，auto 的 task 里
    agent 想停下来说句话也停不了。现在交给 observer 逐次判定，这里只问「有没有人」。

    下面这条 `stop` 是「没让位」的收尾，与让位严格互斥：真让位了就抛 HitlPark，压根
    走不到这里；`_park_await_user` 正常返回就意味着这一轮不让位（无人值守），它保证
    零副作用返回，所以 `stop` 是这个回合唯一的一条 ACT_TURN_COMPLETED。
    """
    # **先合成 outputs，再决定让不让位**：下面那条 park 会抛 `HitlPark` 就地中断整个
    # ActStep，循环之后那次合成到不了。见 `_synthesize_final_outputs` 的 docstring。
    _synthesize_final_outputs(state, transcript)

    if (
        isinstance(state.task.settings, NormalTaskSettings)
        and not state.task.unattended
        and ctx.hitl is not None
    ):
        await _park_await_user(state, ctx, turn_num)
    await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
        "turn": turn_num, "reason": "stop"}))


async def _invoke_tool(tc: ToolCall, state: LoopState, ctx: LoopContext) -> Any:
    """通过 CapabilityGateway 执行工具调用。gateway 不存在时降级为旧路径。"""
    if ctx.capability_gateway is not None:
        return await ctx.capability_gateway.invoke(
            tool_name=tc.name,
            arguments=tc.arguments,
            state=state,
            ctx=ctx,
            tool_call_id=tc.id,
        )

    # 降级：无 gateway 时的最简路径（保持向后兼容）
    from ctx_weft.core.loop.capability_gateway import InvocationResult
    logger.warning("No CapabilityGateway in LoopContext; tool '%s' cannot be dispatched", tc.name)
    return InvocationResult(
        invocation_id="noop",
        tool_name=tc.name,
        content=f"[Error: CapabilityGateway not configured, tool '{tc.name}' skipped]",
        is_error=True,
    )


INTERRUPTED_MARK = "[interrupted by the user]"
CANCELLED_MARK = "[cancelled]"


async def _ingest_synthetic_tool_result(
    state: LoopState, ctx: LoopContext, tc: ToolCall, content: str, *,
    interrupted: bool = False, cancelled: bool = False,
) -> None:
    """为被打断/未执行的工具补一条 TOOL_RESULT，使 tool_call↔result 一一对应（无 dangling）。

    走 `ingest_tool_result`：记录 id 必须是确定性的那一个，否则这条补位在 dangling 判定
    眼里不存在——「补它就是为了不悬挂」这个目的会落空，恢复时该调用被重新执行一遍。
    """
    from ctx_weft.core.loop.capability_gateway import ingest_tool_result
    await ingest_tool_result(
        ctx.memory, ctx.provider_ctx, state.scope,
        tool_call_id=tc.id, content=content, is_error=True, tool_name=tc.name,
        extra_metadata={"interrupted": interrupted, "cancelled": cancelled},
        task_manager=getattr(ctx, "task_manager", None))


async def _await_tool_or_stop(invoke_task: "asyncio.Future", ctx: LoopContext) -> bool:
    """运行在途工具；任一停止信号（硬取消 cancel_token / 软打断 pause_token）在执行中触发
    → 取消该工具并等其清理跑完，返回 False；正常完成返回 True。
    """
    waiters: list[asyncio.Future] = []
    if ctx.cancel_token is not None:
        waiters.append(asyncio.ensure_future(ctx.cancel_token.wait()))
    if ctx.pause_token is not None:
        waiters.append(asyncio.ensure_future(ctx.pause_token.wait_paused()))
    if not waiters:
        await asyncio.wait({invoke_task})
        return True
    try:
        await asyncio.wait({invoke_task, *waiters}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for w in waiters:
            w.cancel()
        await asyncio.wait(set(waiters))
    if invoke_task.done():
        return True
    invoke_task.cancel()
    await asyncio.wait({invoke_task})
    return False


def _interrupt_pending(ctx: LoopContext) -> bool:
    """软打断挂起中：pause_token 被 pause 且有 hitl 可 park。"""
    tok = ctx.pause_token
    return (
        tok is not None
        and tok.is_paused
        and ctx.hitl is not None
    )


def _is_own_root(task) -> bool:
    """root task 判定（与 observe._is_own_root / finalize._close_one 保持一致）。

    True  → session root（parent_task_id is None）或跨 agent own-root。
    False → 同 agent 子任务（parent 非空且 creator==assigned）。
    """
    same_agent = task.creator_agent_id == task.assigned_agent_id
    cross_agent = bool(task.parent_task_id) and not same_agent
    return task.parent_task_id is None or cross_agent


async def _commit_interrupted_partial(
    state: LoopState, ctx: LoopContext, text: str, reasoning: str, turn_num: int,
) -> None:
    """把被打断的半截 assistant 文本入 memory（标注「被用户打断」）。无内容则不留记录（①）。"""
    if not (text.strip() or (reasoning or "").strip()):
        return
    content = f"{text}\n\n{INTERRUPTED_MARK}" if text.strip() else INTERRUPTED_MARK
    await ctx.memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            content=content,
            timestamp=now_utc(),
            role="assistant",
            metadata={
                "turn": turn_num,
                "interrupted": True,
                "reasoning": reasoning or None,
            },
        ),
        ctx.provider_ctx,
    )


def _interrupt_edit_prefix(prev_request: str) -> str:
    """产出①打断续接的前缀说明；prev 为空则返回空串。

    抽出成纯文本→纯文本的小函数，供多模态调用点用 content_with_prefix 组装
    （new_input 可能是 list[ContentPart]，f-string 直接拼会把图片拍扁）。
    """
    prev = (prev_request or "").strip()
    if not prev:
        return ""
    return f'(I cancelled my previous request: "{prev}" — replacing it with the request below.)\n\n'


def interrupt_edit_note(prev_request: str, new_input: str) -> str:
    """① 打断（未吐 token）续接时的说明：上一条请求被取消、改为新请求。空 prev 时原样返回。"""
    prefix = _interrupt_edit_prefix(prev_request)
    return f"{prefix}{new_input}" if prefix else new_input


async def _cold_park(
    state: LoopState, ctx: LoopContext, preface: str, *, unattended: bool,
) -> None:
    """起 wait_for_user 冷 park：登记 HITL 请求 + 抛 HitlPark（task 落 AWAITING_HUMAN
    由 TaskManager 据 RunOutcome 定）。

    **纯机械动作，不含任何策略判断**：preface 选哪条、这次让不让位，都由上面两个具名
    包装（`_park_for_interrupt` / `_park_await_user`）决定后传进来。此前这些判断和动作
    挤在同一个函数里，靠一个 `source` 字符串同时表达「选哪条 preface」与「要不要豁免
    无人值守守卫」两件毫不相干的事。

    **不写会话状态**：这不是本函数的职责。2026-09-02 那次重构曾把会话状态的唯一写者
    定为 `SessionRegistry`，由一条队列级聚合信号翻译成会话级状态；那整条链路已随会话
    状态机退役——`SessionRegistry` 自 2026-09-03 起降格为纯 agent 登记表，两端的事件
    类型也已于 2026-09-05 一并删除。会话级别的展示状态目前不由 core 预先算好广播，
    由 host 自行按 agent 状态聚合推导（docs/events-v2.md §2.1.1）。

    续跑方式由 **delivery 显式声明**，不再靠 `form == "wait"` + sentinel capability_id
    这组跨三个模块的魔法字符串（spec §5）。
    """
    req = await ctx.hitl.open(
        HitlAsk(
            form=HITL_FORM_WAIT,
            delivery=UserTurnDelivery(task_id=state.task.id, preface=preface),
        ),
        session_id=state.session.id,
        task_id=state.task.id,
        agent_id=state.agent.id,
        stage=HITL_STAGE_TOOL,
        unattended=unattended,
        tenant_id=state.session.tenant_id,
    )
    # 不建等待槽 —— 本调用方随即 park 释放协程而非 await，应答必然走冷续跑。
    raise HitlPark(hitl_id=req.id)


async def _park_for_interrupt(
    state: LoopState, ctx: LoopContext, *, edit: bool = False,
) -> None:
    """人按了暂停键 → park 等他续接。**对无人值守守卫豁免**（`unattended=False`）。

    守卫要挡的是「没有人可问」，不是「没有人在场」：按下暂停键的就是一个人，续接的
    也会是那个人——运维暂停一个后台无人值守作业是完全合法的操作，那时 park 正是对的
    行为。这条路径不属于守卫要挡的场景，所以豁免写死在这个具名函数里，而不是让四个
    调用点各传一个 `allow_unattended=True`（那种写法太容易被后来人顺手改掉）。

    ``edit`` 只决定 preface：已吐过 token / 已进工具循环 → after_interrupt；未吐任何
    token 且未进工具 → after_interrupt_edit（续接时需补一句「上一条请求被取消」）。
    """
    preface = PREFACE_AFTER_INTERRUPT_EDIT if edit else PREFACE_AFTER_INTERRUPT
    await _cold_park(state, ctx, preface, unattended=False)


async def _park_await_user(
    state: LoopState, ctx: LoopContext, turn_num: int, *, boundary: str = "plain_text",
) -> None:
    """agent 想让位给用户 → 发 await_user + 起后台判定 + 冷 park。

    ``boundary`` 区分让位的**方式**，它一路传到装配层决定两件事（见
    `composer._BOUNDARY_FACTS` / `_OUTPUTS_BEARING_BOUNDARIES`）：

    - ``plain_text``：说了段话就停下。那段文本本身在重建的对话里看得见，不注入 outputs。
    - ``finish_park``：调了 `finish_task` 宣布完成（S-b，2026-09-27）。`finish_task` 是
      SILENT 工具、产出不写任务层对话，**必须**把 `task.outputs` 注入，否则观察者会虚构
      一段完成叙述。

    两者都产 verdict，都 park，归宿完全一样——这正是 S-b 要的：root 上「说了段话」与
    「宣布做完了」不该一个被复核、一个不被。

    与 `_park_for_interrupt` 相反，这条**没有人保证会回来**：让位是 agent 自己提的，
    无人值守的 task 里根本没人会发下一条消息，park 即永久挂起。所以守卫在这里生效。

    **判断前置于一切副作用**，这是本函数存在的全部意义：`hitl.open()` 里的守卫抛
    `UnattendedHitl` 时，`await_user` 事件早已发出、background observe 早已起飞，
    两个结局都坏——异常逸出打挂整个 run，或就地 catch 继续走、同一回合发出两条
    `ACT_TURN_COMPLETED`（`await_user` 一条、`stop` 一条），前端看到的是「agent 说
    它在等用户，紧接着又说它停了」。

    这里**刻意不包 `try/except UnattendedHitl` 兜底**：判断读的和守卫查的是同一个
    `Task.unattended`，只有一个真相源，不存在漂移。守卫仍留在 `open()` 里，所以将来
    若有人加了新路径又忘了前置判断，行为是当场抛异常而不是静默挂死——那正是我们要的。

    让位与不让位严格互斥：让位则抛 `HitlPark`，调用方那条 `stop` 根本到不了；不让位
    则零副作用返回，`stop` 是这个回合唯一的收尾事件。
    """
    if state.task.unattended:
        logger.warning(
            "act: task %s is unattended — not yielding to the user after a plain-text turn; "
            "treating the text as this turn's output and stopping normally",
            state.task.id,
        )
        return
    await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
        "turn": turn_num, "reason": "await_user"}))
    # 纯文本暂停 = 软待命(允许但不强制回复) → PAUSED,区别于 ask_user 的 PAUSED_HITL。
    #
    # 2026-09-22 去掉了这里的 `_is_own_root` 闸：子任务的纯文本回合同样 park、同样要
    # 判定。当时那个闸的理由是「非 root 走 LLM observe 向 parent 上报」——但纯文本回合
    # 压根到不了前台 observe（它在这里就 park 了），于是子任务的这一段既没人判也没人折。
    # 并发不会因此失控：单交互线闸门保证一个 session 同时只有一条非 unattended 的线
    # （见 `TaskManager._interactive_line_held`）。
    from ctx_weft.core.loop.background import launch_recap
    launch_recap(state, ctx, boundary=boundary)
    await _cold_park(state, ctx, PREFACE_NORMAL, unattended=state.task.unattended)


async def _interrupt_checkpoint(state: LoopState, ctx: LoopContext) -> None:
    """协作式停止点：软打断（pause）→ park；硬取消（cancel）→ CancelledError。

    提交点之前命中（act 第一轮的循环顶部，LLM 还没被调用过）→ 整轮丢弃而非 park，
    见 `_discard_round_if_uncommitted`。
    """
    if _interrupt_pending(ctx):
        await _discard_round_if_uncommitted(state, ctx)   # 命中则 raises RoundDiscarded
        edit = not ctx.run_phase.produced and not ctx.run_phase.in_tool_loop
        if _is_own_root(state.task):
            from ctx_weft.core.loop.background import launch_recap
            launch_recap(state, ctx, boundary="interrupt")
        await _park_for_interrupt(state, ctx, edit=edit)  # raises HitlPark
    tok = ctx.cancel_token
    if tok is not None and tok.is_cancelled:
        tok.raise_if_cancelled()
