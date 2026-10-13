"""CompactStep：长对话压缩，由 PrepareStep 内联直调（不再以 task 形式调度）。

  - 作用域 = 当前 state.scope（当前 task + agent）。
  - escalating_compact：预算驱动的 L0.5→L1→L2→L3 升级编排（替旧的双阈值并行折 _compact_scope）。
    L0.5 = 图片降级（media.demote_for_budget，无 LLM、可逆），排在所有折叠之前（子设计 §6）。
    传入 token_estimate，未达 compact_target_ratio（回退 compact_token_ratio）× context_limit
    时空跑；否则按序试 L1 fold_root_experience（agent 层折叠成 AGENT_COMPACT_SUMMARY）→
    L2 demote_kept_capsules（L1 保留胶囊里 rich→lean 降级，无 LLM）→ L3 collapse_task_layer
    （当前 task 层坍缩成一条 USER_PROMPT，原始消息 + 执行摘要两节）；级间用 _active_memory_tokens
    的增量从估算里累减，降到 target 以下即停。
  - L1/L2 各有可折性 guard（无可折对象则跳过）；L3 无 guard，只要预算门开就会试（内部
    collapse_task_layer 自己在 ≤keep_last 时 noop）。
  - 注：observe(max_turns) 与 background_observe 仍走 apply_compact 写 TASK_COMPACT_SUMMARY
    形成胶囊，不在此文件改动范围内。
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from datetime import timedelta
from typing import Any

from ctx_weft.core.assembler.assembler import ContextRequest
from ctx_weft.providers.llm.stream import ensure_collected_stream
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome, make_event
from ctx_weft.core.loop.llm_gateway import (
    request_prompt_estimate, resolve_llm_identity, stream_llm_resilient,
)
# 模块级 import（**实测不成环**）：`core.media` 的模块级依赖只到 `core.utils.content` /
# `protocols`，不回指 `core.loop`。若日后 media 模块级引入了 `core.loop` 的东西，
# 这里要退回函数级 import。
from ctx_weft.core.media import demote_all, demote_for_budget, placeholder_refs
from ctx_weft.core.utils.content import content_to_text, image_tokens
from ctx_weft.core.utils.estimate import effective_limit
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import (
    LLMRequest, MemoryAddress, MemoryEvent, MemoryKind, MemoryScope,
)

logger = logging.getLogger(__name__)

# v2 P3a：类型清单改 kind 视图。task 层全量视图（对话 + 段摘要 + audit）= 旧
# _TASK_LAYER_TYPES/_TASK_BODY_TYPES 五类型；跨 task 聚合用半址（task_id=None）。
_TASK_VIEW_KINDS = [MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY, MemoryKind.TOOL_AUDIT]

# ── 结构化 digest（spec: compact-fidelity）───────────────────────────────────
# 五小节契约（cue 见 composer 的两域 instruction）；Evidence 允许缺，其余必需。
DIGEST_SECTIONS = ("Goal", "Constraints", "Done", "Remaining", "Evidence")
REQUIRED_DIGEST_SECTIONS = DIGEST_SECTIONS[:4]

_DIGEST_SECTION_RE = {
    name: re.compile(rf"^## {name}[ \t]*\n(.*?)(?=^## |\Z)", re.S | re.M)
    for name in DIGEST_SECTIONS
}


@dataclasses.dataclass(frozen=True)
class ParsedDigest:
    """解析后的 digest：text 为落库形态（结构化时即原文），degraded 标降级。

    sections 供断言/评测消费（name → 小节正文；Evidence 可缺）。降级 = 整文作 digest
    （散文 digest 仍可用，不重试不中断——重试一次 LLM 调用换不回格式保证）。
    """

    text: str
    degraded: bool
    sections: dict[str, str] = dataclasses.field(default_factory=dict)


def parse_structured_digest(text: str) -> ParsedDigest:
    """宽松解析（spec: compact-fidelity）：抓 `## <Name>` 小节；Goal/Constraints/
    Done/Remaining 任一缺失（或空）→ 整文作 digest、degraded=True。旧散文 digest
    （本 change 之前的数据）天然走降级路径，无需迁移。"""
    sections: dict[str, str] = {}
    for name in DIGEST_SECTIONS:
        m = _DIGEST_SECTION_RE[name].search(text)
        if m and m.group(1).strip():
            sections[name] = m.group(1).strip()
    degraded = any(name not in sections for name in REQUIRED_DIGEST_SECTIONS)
    return ParsedDigest(text=text, degraded=degraded, sections=sections)


async def _resolve_digest(value) -> ParsedDigest:
    """把 summarize 的产物（ParsedDigest / 旧测试桩的裸 str / 取值的零参协程）归一。

    兼容裸 str 是刻意的：既有测试大量 monkeypatch summarize_for_compact 返回 str，
    落库点不应对替身形态敏感（str → 无小节 → degraded，语义自洽）。"""
    if dataclasses.is_dataclass(value) and isinstance(value, ParsedDigest):
        return value
    if isinstance(value, str):
        return parse_structured_digest(value)
    if inspect.isawaitable(value):
        return await _resolve_digest(await value)
    if callable(value):
        return await _resolve_digest(value())
    raise TypeError(f"unsupported digest payload: {type(value)!r}")


def _media_enabled(ctx) -> bool:
    """L0.5 与 §6.1 前置降级的总闸：只有真接了**可外部化**的 `MemoryBlobStore` 时才跑。

    未注册（`LoopContext.blob_store is None`）或注册的是 `NullMemoryBlobStore`
    （`can_externalize=False`）时，视图里根本不可能存在 `source_type == "ref"` 的图，
    降级必然返回 0；提前短路省掉整级的读操作，坐实子设计 §10「不接 MemoryBlobStore 时行为
    与改造前完全一致」——包括**一次多余的 memory 读都不发**。

    ⚠️ 这**不是** Task 2 判断题 2 拒绝的「第二处 registry 探询」。那条拒的是拿 registry
    当「哪些 part 该降」的判据（已注册但存量记录仍是 inline base64 时会放行并把图弄丢）。
    选谁降级仍然只由 `policy.demotable_ref` 的 ref 判据决定；本闸只能让降级**少做**，
    不能让它多做。代价是一个边角：曾接过 blob store、现已摘掉的宿主，存量 ref 图不再
    被降级——那些图本来也 rehydrate 不回来了。
    """
    store = getattr(ctx, "blob_store", None)
    return store is not None and bool(getattr(store, "can_externalize", False))


def _agent_half(scope) -> MemoryAddress:
    """agent 层半址（AGENT 视图禁带 task_id）。"""
    return MemoryAddress(session_id=scope.session_id, agent_id=scope.agent_id)


async def summarize_for_compact(
    state: LoopState, ctx: LoopContext, *, scope: str = "task"
) -> ParsedDigest:
    """装配 purpose="compact" 上下文 + 一次 LLM 摘要，返回解析后的 digest。

    scope 选 cue（composer 据 extra["compact_scope"] 分流）："task"=整段执行摘要（默认，
    observe 兜底与坍缩共用）；"agent"=派发经验摘要。两域 cue 均要求五小节结构（spec:
    compact-fidelity）；产出经 parse_structured_digest 宽松解析——缺必需节整文降级
    （degraded=True），不重试、不中断。

    LLM 摘要是 compact 的硬依赖（compact + observe 回退档共用本函数）：瞬时故障由
    stream_llm_resilient 自愈，自愈耗尽抛 LLMOutageError → 走 INTERRUPTED。**不再**在
    LLM 失败时静默退化为纯截断（旧的 except Exception 兜底已移除）。

    task-4 复审修复（第二轮）：``model`` 曾用 ``agent.runtime.get("llm_model", "mock")``
    ——inline compact 路径下 ``agent.runtime`` 从不会被写入 ``llm_model``（正常任务执行
    materialize() 出来的 Agent 就是空 runtime），这行代码因此在那条路径下恒回落
    ``"mock"``。改用同子系统其余调用方（act.py 等）已经在用的 ``resolve_llm_identity``
    ——真值来自 ``state.resolved_model``，是派发时 ``AgentLifecycleManager.resolve_model`` 解出的
    那一个，compact 的三条调用路径（inline PrepareStep 触发 / ``_maybe_predispatch_compact``
    / 独立的 ``compact_agent``）用的都是已经带 ``resolved_model`` 的 ``LoopState``（前两者
    共享 ``_execute_task`` 构造的那份，同一份 state 上 act.py 的 ``resolve_llm_identity``
    早就在成功调用；后者是 ``runtime.py::compact_agent`` 显式 ``resolved_model=rm`` 构造
    的孤儿 run），故不额外加 try/except 兜底——真出现 None 就该炸，不吞。

    真实调查结论（供风险评估）：这次修复前，实际发给 LLM 服务商的模型**并不是**字面上的
    "mock"——``providers/llm/anthropic.py``/``openai.py`` 的 ``_build_payload`` 都已有
    ``model = request.model if request.model and request.model != "mock" else self._model``
    这行兜底替换（``anthropic.py`` 的 ``model`` 属性文档字符串原话：「实际调用一直用它替换
    "mock"……事件账面须与之同源」），而 ``self._model``（client 构造时配置的模型）与
    ``AgentLifecycleManager.resolve_model()`` 在 ``choice.model`` 为空时的回落值（``getattr(client,
    "model", "")``）同源，与 ``state.resolved_model.model`` 基本一致。也就是说**运行时
    行为本身不受影响**（真实 adapter 早已在暗中纠正这个 "mock" 哨兵），本次修复修的是
    「事件 payload 撒谎」而非「调错模型」——`docs/events-v2.md` §3.6 要求 model/llm_account
    是「实际用的那个」，撒谎的是事件，不是调用（`MockLLMAdapter` 例外：它没有这层兜底，
    ``last_request.model`` 会原样收到调用方传入的值——`compact.py` 里那处
    ``agent.runtime={"llm_model": rm.model}`` 桥接就是专给它搭的，见 `runtime.py`
    ``compact_agent``；本次改动后这座桥不再是唯一支撑，`resolve_llm_identity` 本身就
    正确，桥不拆但也不再是必需）。

    task-4 复审修复（第一轮）：compact 走的这次 LLM 调用现在也经 gateway 的门禁（origin==
    LOOP_COMPACT）自动获得 LLM_REQUEST_STARTED/LLM_PROMPT_SENT/LLM_TOKEN_STREAMED/
    LLM_REASONING_STREAMED；本函数补发收尾的 LLM_RESPONSE_FINISHED——两边必须成对，
    否则 host SSE 看到的是一条永远等不到收尾的挂死请求。payload 字段名/取值方式照抄
    act.py 那份（``_run_llm_turn``）。request_id 用同一个确定性公式独立算出（同
    act.py 的手法，见 llm_gateway.stream_llm_resilient 文档字符串）：两边都在本次
    调用任何事件发射前求值，天然一致，不需要新增参数或跨函数传值。

    收尾事件的发射条件与 gateway 那 4 个流式事件**同一个门禁**（``bus is not None`` 且
    ``state.origin == EventOrigin.LOOP_COMPACT``），而不是只看 ``bus``——否则本函数万一
    在别的 origin 下被调用（目前实际不会，但别指望调用方永远守规矩），会发一条没有配对
    REQUEST_STARTED 的孤儿 RESPONSE_FINISHED，比不发更糟。``bus is None``（多数单测的
    ctx 不接总线）时事件收发跳过——`state.sequence_counter` 在这些最小化 state 上不一定
    存在，故 request_id 的计算放在同一个 guard 里，不会在 bus 缺失时因缺字段而崩；
    ``resolve_llm_identity`` 则不受这个 guard 保护（见上，本就该在 bus 缺失时也正确解析、
    正确发送）。
    """
    agent = state.agent
    request = ContextRequest(
        purpose="compact",
        scope=state.scope,
        task=state.task,
        agent=agent,
        session=state.session,
        template=state.extra.get("template"),
        bound_capabilities=state.extra.get("bound_capabilities", []),
        token_counter=ctx.llm.tokenizer.count,
        extra={"compact_scope": scope},
    )
    compact_prompt = await ctx.assembler.assemble(request)

    model, llm_account = resolve_llm_identity(state)
    llm_request = LLMRequest(
        model=model,
        system=compact_prompt.system,
        messages=compact_prompt.messages,
        tools=[],
    )
    # 一次性调用（无循环内基线）→ baseline=None，走 max(整份估算, context_tokens)。
    llm_request.prompt_token_estimate = request_prompt_estimate(
        ctx.llm.tokenizer, llm_request, getattr(agent, "loop_guard", None), None)

    bus = getattr(ctx, "event_bus", None)
    # 同 gateway 那 4 个流式事件一个门禁：bus 存在 且 origin==LOOP_COMPACT（见上方 docstring）。
    should_emit_finished = bus is not None and getattr(state, "origin", None) == EventOrigin.LOOP_COMPACT
    req_id = f"req_{agent.id}_{state.sequence_counter}" if should_emit_finished else None

    stream = ensure_collected_stream(stream_llm_resilient(ctx, state, llm_request))
    async with aclosing(stream):
        async for _ in stream:
            stream.accept_current()
    result = stream.get_final_result()
    summary_text, reasoning_text, usage = result.text, result.reasoning, result.usage

    if should_emit_finished:
        await bus.emit(make_event(
            state, EventType.LLM_RESPONSE_FINISHED,
            payload={
                "request_id": req_id, "content": summary_text, "reasoning": reasoning_text,
                "tool_calls": [],
                "usage": dataclasses.asdict(usage),
                "llm_model": model, "llm_account": llm_account,
                "finish_reason": "stop", "turn": 0}))
    # spec: compact-fidelity——落库前宽松解析（缺必需节整文降级，不中断）。
    return parse_structured_digest(summary_text)


# 坍缩 USER_PROMPT 的两节分隔标记；再坍缩时据此切出「原始消息」节，保持有界。
COLLAPSE_DELIM = "\n\n---\n## Execution Summary (earlier conversation compacted)\n"
# 存量标记：分隔标记进 prompt，改英文后旧库里的已坍缩 USER_PROMPT 仍带中文标记。切不出
# 原文节 = 把「原文 + 旧摘要」整体当原文，再坍缩一次就翻倍，故读侧一并认旧标记。
LEGACY_COLLAPSE_DELIMS = ("\n\n---\n## 执行摘要（先前对话已压缩）\n",)


def _original_section(content: str) -> str:
    """取（可能已坍缩过的）USER_PROMPT 的「原始消息」节：有分隔标记取其前段，否则整体即原文。"""
    for delim in (COLLAPSE_DELIM, *LEGACY_COLLAPSE_DELIMS):
        idx = content.find(delim)
        if idx != -1:
            return content[:idx]
    return content


def _rendered_only(recs: list) -> list:
    """去掉装配时不渲染的记录（TOOL_AUDIT）。

    `AgentRecallSource` 只取对话回合 + 摘要进 prompt；审计记录只在 memory 里留痕。凡是
    「保留多少条给模型看」的计数都该按这个口径数，否则审计会白占名额。`getattr` 容忍测试桩。
    """
    return [r for r in recs if getattr(r, "kind", None) is not MemoryKind.TOOL_AUDIT]


def _split_for_collapse(recs: list, keep_last: int) -> tuple[list, list]:
    """把 task 层视图切成 (折区, 保留区)：保留**至少** keep_last 条会被渲染的记录，并对齐回合边界。

    两条规则，缺一条都会让保留区在模型眼里变空：

    1. **按渲染口径数名额**。TOOL_AUDIT 不进 prompt，不占 keep_last。
    2. **切点不落在一轮中间**。保留区里的每条工具结果，发起它的 assistant 也必须在保留区；
       否则结果成孤儿，发送前被 `drop_orphan_tool_results` 静默丢掉。一轮并行调两个工具时，
       旧的「最后 N 条记录」切法留下的是 `[result, audit, result]`，渲染出来 0 条。

    对齐只会让切点往前挪（保留更多），所以保留区可能超过 keep_last 条——keep_last 是下限。
    按 tool_call_id 找归属而不是「往前跳过连续的 tool 记录」：工具结果与 assistant 之间可能
    夹着别的记录（审计、异步回填），只看相邻关系会漏。
    """
    if keep_last <= 0:
        return list(recs), []
    cut, seen = len(recs), 0
    for i in range(len(recs) - 1, -1, -1):
        if getattr(recs[i], "kind", None) is MemoryKind.TOOL_AUDIT:
            continue
        seen, cut = seen + 1, i
        if seen >= keep_last:
            break
    if seen < keep_last:
        return [], list(recs)
    while True:
        kept_call_ids = {r.metadata.get("tool_call_id") for r in recs[cut:]
                         if r.role == "tool" and r.metadata.get("tool_call_id")}
        owner = next((i for i in range(cut) if recs[i].role == "assistant" and any(
            tc.get("id") in kept_call_ids for tc in (recs[i].metadata.get("tool_calls") or []))),
            None)
        if owner is None:
            return recs[:cut], recs[cut:]
        cut = owner


def _worth_collapsing(fold: list) -> bool:
    """折区里除了**那一条最早的 user 回合**（原始请求 / 旧坍缩物）之外还有会被渲染的东西，才值得坍。

    折区只剩原始请求时，「坍缩」= 把 USER_PROMPT 换成「USER_PROMPT + 摘要」，token 不降反升，
    还白烧一次摘要 LLM。按回合切之后这种情况很常见：整个 task 只有一轮大并行调用时，对齐会把
    切点一路挪到那一轮的 assistant，折区里就只剩原始请求。

    只豁免**第一条** user 回合而不是所有 user 回合：连续多条用户消息（交互任务）合并成「第一条
    原文 + 摘要」仍然可能省 token，不该被挡掉。
    """
    rendered = _rendered_only(fold)
    first_user = next((r for r in rendered
                       if r.role == "user" and r.kind is MemoryKind.CONVERSATION_TURN), None)
    return any(r is not first_user for r in rendered)


async def collapse_task_layer(
    state, ctx, keep_last: int,
    summary_text: "str | ParsedDigest | Callable[[], Awaitable[str | ParsedDigest]]",
) -> int:
    """task compact（二级压缩）：把当前 task 层早期回合（含原始 USER_PROMPT 与 observer 的
    `## Progress So Far`）整体坍缩成一条新 USER_PROMPT，content = 原始消息 + COLLAPSE_DELIM +
    执行摘要；保留最近**至少** keep_last 条会被渲染的记录，切点对齐回合边界（见
    `_split_for_collapse`）。返回 supersede 条数；无可折之物 → 0，且不求值摘要。

    坍缩物是 USER_PROMPT 而非 assistant 摘要：composer 据 mtype=="user_prompt"+task_id 定位当前
    task 贴 `## Current Task/## Current Message` 框，故当前运行 task 坍缩后框架不丢；已结束胶囊被
    跨 task 召回时它就是一条背景 message。degraded 标记随 metadata 落库（spec: compact-fidelity）。

    `summary_text` 可以是算好的文本（兼容裸 str 的旧测试桩），也可以是**取摘要的零参 async
    函数**（`escalating_compact` 传的就是后者）：摘要在切分、no-op 判定、折区图片降级**之后**
    才求值——无可折之物时不花那次 LLM，且摘要输入里不会有即将被降级的真图（同 L1 的顺序）。
    """
    memory = ctx.memory
    recs = await memory.load_view(
        state.scope, MemoryScope.TASK, ctx.provider_ctx, kinds=_TASK_VIEW_KINDS)
    fold, kept = _split_for_collapse(recs, keep_last)
    if not _worth_collapsing(fold):
        return 0

    # §6.1：折叠范围内的残留真图**先无条件降级**（不受 keep_recent 保护）。L0.5 保住了
    # 最近 keep_recent 张，升级到这一级时它们恰好可能落进折区；不降的话下面取 `original`
    # 节走 content_to_text 会把它们静默拍扁——最老的图留下了可取回的占位，最新的反而
    # 痕迹全无，优先级完全颠倒。降级后 ref 随 `original` 节一起活下来。
    if _media_enabled(ctx) and await demote_all(
        memory, [r.id for r in fold], ctx.provider_ctx,
        address=state.scope, scope=MemoryScope.TASK, kinds=_TASK_VIEW_KINDS,
    ):
        # 🔴 必须重新 load_view：降级是 fold(旧 id, 新事件)，**record id 全换了**。
        # 拿降级前采集的 id 去下面那次 fold，被降过的一条都 supersede 不掉 → 同一段
        # 对话在视图里出现两次；且手里这份 recs 已过期，用它取 `original` 节拿到的仍是
        # 真图，照旧被拍扁，本段的目的完全落空。
        recs = await memory.load_view(
            state.scope, MemoryScope.TASK, ctx.provider_ctx, kinds=_TASK_VIEW_KINDS)
        fold, kept = _split_for_collapse(recs, keep_last)
        if not _worth_collapsing(fold):
            return 0

    # 「原始消息」节 = 折区最早一条 user 回合的原文（已坍缩过则取其原始节，保持有界）
    original = ""
    for r in fold:
        if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "user":
            text = r.content if isinstance(r.content, str) else content_to_text(r.content)
            original = _original_section(text)
            break

    # 锚：新 USER_PROMPT 须排在所有保留回合之前
    anchor_src = kept[0] if kept else fold[0]
    anchor_ts = anchor_src.timestamp - timedelta(microseconds=1)

    ids = [r.id for r in fold]
    digest = await _resolve_digest(summary_text)
    collapsed = f"{original}{COLLAPSE_DELIM}{digest.text or '[Context compacted]'}"
    # v2 P3d：遗忘+坍缩物一次原子 fold（旧徒手 supersede+ingest 有崩溃丢摘要窗口）
    await memory.fold(ids, [
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            content=collapsed,
            timestamp=anchor_ts,
            role="user",
            metadata={"task_id": state.scope.task_id, "collapsed": True,
                      "keep_last": keep_last, "folded_count": len(fold),
                      "digest_degraded": digest.degraded},
            # 上面 §6.1 的前置降级把折区的真图换成了 L0.5 占位，`original` 节又把其中
            # 一部分逐字带进了坍缩物——那些 ref 于是**仍在视图里、仍被 get_image 定位得到**，
            # 而承载它们的原记录正要被这一次 fold 全部 supersede。占位是文本，GC 的 mark
            # 判据（`collect_blob_refs`）只看结构化字段，不声明就等于让引用归零：宽限期一过
            # 字节被回收，模型读到「call media:get_image(...) to bring it back」，换回来的却是
            # `[image unavailable]`——L0.5 承诺的「可逆」在这一步失效。
            # 只声明**幸存在坍缩物里**的那些（`placeholder_refs` 的口径），没进来的那些
            # 本就再也定位不到，一并声明只会让字节永远回收不掉。
            blob_refs=placeholder_refs(collapsed),
        ),
    ], ctx.provider_ctx)
    return len(ids)


async def maybe_compact_before_dispatch(
    state: LoopState, ctx: LoopContext, *, prompt_tokens: int
) -> list[Any]:
    """派发执行前的按需压缩：本轮 prompt token 越过 predispatch_compact_token_ratio 时，
    对父 scope 跑一次完整压缩（task 层 fold_task + agent 层 fold_root + 回收 finished 短 task，
    共享单次 summary），使父 resume 更精简、inherit 快照为压缩后版本。

    与 PrepareStep 的 compact 同构（复用 escalating_compact），区别只在触发条件：这里由独立的
    predispatch token 阈值门控，且仅在本轮含派发调用时由 ActStep 调用。门控不满足返回空列表。
    """
    agent = state.agent
    ratio = getattr(agent.loop_config, "predispatch_compact_token_ratio", 0.0)
    if ratio <= 0:
        return []
    context_limit = agent.loop_guard.context_limit
    reserve = getattr(agent.loop_guard, "reserved_output_tokens", 0)
    eff = effective_limit(context_limit, reserve)
    tokens = prompt_tokens or agent.loop_guard.context_tokens
    if eff <= 0 or tokens <= 0:
        return []
    if tokens / eff < ratio:
        return []
    return await escalating_compact(state, ctx, token_estimate=tokens, trigger="pre_dispatch")


def _dispatch_finish_sets(recs) -> tuple[set, set]:
    """扫 conversation turn，按 origin 标记：has_dispatch（有 delegate/delegate_plan 调用回合）、
    has_finish（有 finish_task 调用回合）。`active = has_dispatch − has_finish` = 未结束的
    delegating task（在途 working set，不可折，spec §2.3）。"""
    has_dispatch: set = set()
    has_finish: set = set()
    for r in recs:
        if r.kind is not MemoryKind.CONVERSATION_TURN or r.role != "assistant":
            continue
        oid = r.metadata.get("origin_task_id")
        if oid is None:
            continue
        for tc in (r.metadata.get("tool_calls") or []):
            name = str(tc.get("name", ""))
            if name.endswith("finish_task"):
                has_finish.add(oid)
            elif name.endswith("delegate_task") or name.endswith("delegate_plan"):
                has_dispatch.add(oid)
    return has_dispatch, has_finish


async def _active_memory_tokens(state: LoopState, ctx: LoopContext) -> int:
    """当前 scope 活跃记忆的 token 代理：task 层 body（跨 task 半址聚合）+ agent 层对话/摘要，
    逐条 content 求 ctx.llm.tokenizer.count 之和。用于升级 compact 级间的 before/after 增量粗估（非精确装配）。"""
    total = 0
    body = await ctx.memory.load_view(
        _agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx, kinds=_TASK_VIEW_KINDS)
    agent_recs = await ctx.memory.load_view(
        _agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx)
    for r in [*body, *agent_recs]:
        text = r.content if isinstance(r.content, str) else content_to_text(r.content)
        # 图片另计：不计入则降级图片的 freed_tokens 恒为 0，escalating_compact 的
        # est 不减、误判该级白跑而继续升级（见 spec 2026-08-20-multimodal-design §6.5）。
        total += ctx.llm.tokenizer.count(text) + image_tokens(r.content)
    return total


async def _count_root_residues(state: LoopState, ctx: LoopContext) -> int:
    """本 agent scope 内「结束顶层单元（胶囊）」数（spec 2026-06-29，驱动 fold 触发阈值）。

    顶层单元 = 按 origin_task_id 分组的 conversation turn。一个 root task **无论当前 agent 亲自
    执行（finish 对 → has_finish）、还是派给别的 agent 执行且结果已回（dispatch + result/tool 回合
    → has_result）**，都是「已完成顶层单元」，计入 keep_last。真·在途（active，不计）= 派出但结果
    未回：has_dispatch 且既无 has_finish 也无 has_result。当前正在跑的 task 永不当完成单元。"""
    recs = await ctx.memory.load_view(
        _agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN],
    )
    # parent prefer-non-None：dispatch result 回合不带 parent（None），不得覆盖权威 parent。
    parent_of: dict[str, Any] = {}
    has_dispatch, has_finish = _dispatch_finish_sets(recs)
    # has_result：该 origin 有返回结果（result/tool 回合）——派发型 root task 的「完成」标志。
    has_result = {r.metadata.get("origin_task_id") for r in recs
                  if r.role == "tool" and r.metadata.get("origin_task_id") is not None}
    for r in recs:
        oid = r.metadata.get("origin_task_id")
        if oid is None:
            continue
        p = r.metadata.get("parent_task_id")
        if oid not in parent_of or (parent_of[oid] is None and p is not None):
            parent_of[oid] = p
    origins = set(parent_of)
    active = has_dispatch - has_finish - has_result  # 派出且结果已回=完成；仅结果未回才算在途
    current = getattr(state.scope, "task_id", None)   # 当前正在跑的 task 不算可折完成单元
    top = {oid for oid, pid in parent_of.items()
           if oid != current and oid not in active and (pid is None or pid not in origins)}
    return len(top)


async def fold_root_experience(
        state: LoopState, ctx: LoopContext, keep_last: int,
        summary_text: "str | ParsedDigest | Callable[[], Awaitable[str | ParsedDigest]]",
) -> int:
    """跨层折叠 fold：task 的详细度只 3 级（spec 2026-06-29 重订，删 L1 黑盒中间态）。

    - **执行中**：RUNNING/SUSPENDED → task 层 raw（每条工具调用展开），不在本函数管辖。
    - **胶囊**：close 时自动形成（`finalize._supersede_final_raw_segment` 删末 raw 段、留
      USER_PROMPT + TASK_COMPACT_SUMMARY）= task 层胶囊 + agent 层 finish 对（+ dispatch 对）。
      胶囊**只软删、不做 L1 压缩**——一直完整保留,直到超 keep_last 才整体折成摘要。
    - **总结**：超 keep_last 的最老顶层单元 → 连同其 task 层胶囊 + agent 层 finish/dispatch 对 +
      旧 AGENT_COMPACT_SUMMARY 一并 supersede,折成一条新 AGENT_COMPACT_SUMMARY（锚保留集最早
      ts − 1µs，不变量 A）。

    顶层折叠单元 = 一组同 origin_task_id 的 conversation turn（finish 对 + dispatch 对）；顶层 =
    parent_task_id is None 或 ∉ 本 scope origin 集；折某顶层时沿 parent 链纳后代子树。active
    delegating task（有 dispatch 回合无 finish 对 = 未结束）是在途 working set,不当可折顶层单元。

    跨层 supersede 一次原子提交（ids 跨 task 层胶囊 + agent 层对话,provider 按 id 生效、不按
    scope 过滤）。返回 supersede 的总条数。

    `summary_text` 可以是**已算好的摘要文本**，也可以是一个**取摘要的零参 async 函数**
    （Task 5b / P4-L10）：§6.1 的前置降级必须发生在摘要生成**之前**，否则折区里的真图
    会被 `content_to_text` 静默拍扁进摘要输入（连占位都没有），随后记录被 supersede——
    正是 §6.1 要防的「最新的图反而痕迹全无」。摘要由本函数在降级 + 重新 `load_view`
    之后才求值，调用方因此不必把降级逻辑（依赖折区 ids）搬到自己那边去。
    """
    memory = ctx.memory
    # AGENT 视图默认 kinds（对话+摘要）；legacy dispatch 配对已在 load_view 内归一，升序即时序。
    recs = await memory.load_view(
        _agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx,
    )
    # task 层胶囊（跨 task 半址聚合，每条 address.task_id 标来源单元）
    body_recs = await memory.load_view(
        _agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx, kinds=_TASK_VIEW_KINDS,
    )

    # 按 origin 分组 conversation turn（finish 对 + dispatch 对同 origin）+ 记 parent + 最早 ts。
    # parent prefer-non-None：单元 parent 由权威回合（gateway delegate / finish 对）给出，dispatch
    # result 回合不带 parent（None），不得覆盖真实 parent。
    parent_of: dict[str, Any] = {}
    first_ts: dict[str, Any] = {}
    has_dispatch, has_finish = _dispatch_finish_sets(recs)
    for r in recs:
        if r.kind is not MemoryKind.CONVERSATION_TURN:
            continue
        oid = r.metadata.get("origin_task_id")
        if oid is None:
            continue
        p = r.metadata.get("parent_task_id")
        if oid not in parent_of or (parent_of[oid] is None and p is not None):
            parent_of[oid] = p
        if oid not in first_ts or r.timestamp < first_ts[oid]:
            first_ts[oid] = r.timestamp
    origins = set(parent_of)
    # 完成的顶层单元（保留/折叠的对象）= 同 agent 亲做（has_finish）或派给别的 agent 且结果已回
    # （has_result：有 result/tool 回合）。仅「派出但结果未回」才是真·在途 active，不作可折胶囊。
    has_result = {r.metadata.get("origin_task_id") for r in recs
                  if r.role == "tool" and r.metadata.get("origin_task_id") is not None}
    active = has_dispatch - has_finish - has_result
    current = getattr(state.scope, "task_id", None)  # 当前正在跑的 task 不折
    top = [oid for oid, pid in parent_of.items()
           if oid != current and oid not in active and (pid is None or pid not in origins)]
    if len(top) <= keep_last:
        return 0
    top.sort(key=lambda oid: first_ts[oid])

    def _expand(roots: list) -> set:
        """从顶层 origin 出发，沿 parent 链纳入所有后代组（同 agent 内嵌子胶囊）。"""
        out = set(roots)
        changed = True
        while changed:
            changed = False
            for oid, pid in parent_of.items():
                if pid in out and oid not in out:
                    out.add(oid)
                    changed = True
        return out

    # 单一阈值：保最近 keep_last 个胶囊（完整：task 层胶囊 + agent 对话），更老的整体折成摘要。
    fold_top = top if keep_last <= 0 else top[:-keep_last]
    fold_set = _expand(fold_top)        # 整体折成摘要的单元（含其子树）

    def _collect(agent_recs: list, task_recs: list) -> list:
        """本次要 supersede 的 id 全集。降级会换掉 record id，故这一步必须能重跑。"""
        out: list = []
        # 折掉 fold_set 单元的 task 层胶囊（user 回合 + 段摘要等；address.task_id 标来源）
        for r in task_recs:
            if (r.address.task_id if r.address else r.metadata.get("task_id")) in fold_set:
                out.append(r.id)
        # + 其 agent 层 conversation turn（finish 对 + dispatch 对，同 origin 同命运）+ 旧摘要
        for r in agent_recs:
            if (r.kind is MemoryKind.CONVERSATION_TURN
                    and r.metadata.get("origin_task_id") in fold_set):
                out.append(r.id)
            elif r.kind is MemoryKind.SUMMARY:
                out.append(r.id)  # 旧摘要并入新摘要
        return out

    ids = _collect(recs, body_recs)
    if not ids:
        return 0

    # §6.1：折叠范围内的残留真图先无条件降级（不受 keep_recent 保护），理由同
    # collapse_task_layer 处。折区横跨两个视图，故两次 demote_all（各自 load_view）。
    if _media_enabled(ctx):
        demoted = await demote_all(
            memory, ids, ctx.provider_ctx, address=_agent_half(state.scope),
            scope=MemoryScope.TASK, kinds=_TASK_VIEW_KINDS)
        demoted += await demote_all(
            memory, ids, ctx.provider_ctx, address=_agent_half(state.scope),
            scope=MemoryScope.AGENT)
        if demoted:
            # 🔴 必须重新 load_view + 重算 ids：降级换掉了 record id，拿旧 id 去下面那次
            # fold，被降过的一条都 supersede 不掉 → 同一段对话在视图里出现两次
            # （旧的没被 supersede、新的补偿记录也活着），而摘要已按旧内容生成。
            recs = await memory.load_view(
                _agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx)
            body_recs = await memory.load_view(
                _agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx,
                kinds=_TASK_VIEW_KINDS)
            ids = _collect(recs, body_recs)
            if not ids:
                return 0

    # 🔴 摘要在这里才求值——**必须在上面那次前置降级之后**（Task 5b，修 P4-L10）。
    # 反过来（旧序：调用方先算好摘要再传进来）的话，折区里那些被 L0.5 的 keep_recent
    # 保下来的**最新的**图会先被 `content_to_text` 拍扁进摘要输入（连占位都没有、
    # ref 彻底没了），紧接着记录就被下面的 `fold()` supersede → 老图留下可取回的占位、
    # 最新的图反而痕迹全无，正是 §6.1 开头写的「优先级完全颠倒」。
    # 兼容裸 str / 零参协程（既有测试桩形态）——统一经 _resolve_digest（spec:
    # compact-fidelity，degraded 随 metadata 落库）。
    digest = await _resolve_digest(summary_text)

    # 折出新摘要：仅当确有单元被折时写（纯遗忘 = fold(ids, [])）
    if not fold_top:
        await memory.fold(ids, [], ctx.provider_ctx)
        return len(ids)

    # anchor = 仍保留单元的最早足迹 ts − 1µs（新摘要须排在所有保留胶囊之前）。足迹 = agent 层
    # conversation turn **加** task 层胶囊：健康数据里 user 回合与 task 层 user_prompt 同 ts，只看
    # 回合即够；但存量数据可能 user 回合被折而 task 层 user_prompt 存活（跨层折叠不同步），此时只看
    # 回合会漏掉更早的胶囊 ts、令摘要排到该胶囊之后。并入 body_recs ts 对健康数据是 no-op、对该
    # 类 split 令锚点回到最早（防御）。
    surviving = origins - fold_set
    kept_ts: list = []
    for r in recs:
        oid = r.metadata.get("origin_task_id")
        if r.kind is MemoryKind.CONVERSATION_TURN and oid in surviving:
            kept_ts.append(r.timestamp)
    for r in body_recs:
        if (r.address.task_id if r.address else r.metadata.get("task_id")) in surviving:
            kept_ts.append(r.timestamp)
    anchor_ts = (min(kept_ts) if kept_ts else now_utc())
    folded_summary = digest.text or "[Experience compacted]"
    # v2 P3d：跨层遗忘 + 新摘要一次原子 fold（关旧「raw 已删而摘要未写」窗口）
    await memory.fold(ids, [
        MemoryEvent(
            kind=MemoryKind.SUMMARY, scope=MemoryScope.AGENT,
            address=state.scope,
            content=folded_summary,
            timestamp=anchor_ts - timedelta(microseconds=1),
            role="user",
            metadata={"keep_last": keep_last, "folded_count": len(fold_top),
                      "digest_degraded": digest.degraded},
            # 同 collapse_task_layer 处：摘要输入里含 L0.5 占位（§6.1 的前置降级刚写下），
            # 模型**可能**把某个 ref 逐字抄进摘要。抄进来了就仍能被 get_image 定位到，
            # 故必须声明；没抄进来则 `placeholder_refs` 返回空表，这里恒为 no-op。
            blob_refs=placeholder_refs(folded_summary),
        ),
    ], ctx.provider_ctx)
    return len(ids)


async def demote_kept_capsules(state: LoopState, ctx: LoopContext, origin_ids: set) -> int:
    """L2：把 origin_ids 里本 agent 亲做的 rich 胶囊降级成 sub-agent lean 表示。
    - 删该 task 的 task 层 body（USER_PROMPT/段摘要/raw，metadata['task_id'] in origin_ids）。
    - agent 层 finish 对：supersede assistant 槽（act_recap + finish 调用），保留 tool 槽（综合总结回填）
      作 lean 表示。已无 assistant 槽（已 lean / 纯 dispatch 对）的单元跳过。
    无 LLM。返回 supersede 条数。"""
    memory = ctx.memory
    body = await memory.load_view(
        _agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx, kinds=_TASK_VIEW_KINDS)
    turns = await memory.load_view(
        _agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN])
    has_dispatch, has_finish = _dispatch_finish_sets(turns)

    ids: list = []
    for r in body:
        if (r.address.task_id if r.address else r.metadata.get("task_id")) in origin_ids:
            ids.append(r.id)   # 删 task 层 body（降级核心：丢交互细节）
    for r in turns:
        oid = r.metadata.get("origin_task_id")
        if oid not in origin_ids:
            continue
        # 仅降级「本 agent 亲做」单元（有 finish 回合）；纯 dispatch 对本就 lean，不动
        if oid in has_finish and r.role == "assistant":
            ids.append(r.id)   # 折 finish 对 assistant 槽，仅留 tool 槽回填
    if not ids:
        return 0
    await memory.fold(ids, [], ctx.provider_ctx)  # 纯遗忘（v2 P3d）
    return len(ids)


async def _kept_origin_ids(state: LoopState, ctx: LoopContext, keep_last: int) -> list:
    """L1 折后仍保留的最近 keep_last 个顶层单元的 origin_task_id（L2 的降级对象）。

    **从老到新排序**（曾返回无序 set）：L2 逐颗降级、达标即停，顺序决定了先牺牲谁——
    必须是最老的先降，最近完成的胶囊最后才动。
    """
    recs = await ctx.memory.load_view(
        _agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN])
    parent_of, first_ts = {}, {}
    has_dispatch, has_finish = _dispatch_finish_sets(recs)
    for r in recs:
        if r.kind is not MemoryKind.CONVERSATION_TURN:
            continue
        oid = r.metadata.get("origin_task_id")
        if oid is None:
            continue
        p = r.metadata.get("parent_task_id")
        if oid not in parent_of or (parent_of[oid] is None and p is not None):
            parent_of[oid] = p
        if oid not in first_ts or r.timestamp < first_ts[oid]:
            first_ts[oid] = r.timestamp
    origins = set(parent_of)
    # 与 fold_root_experience 同口径：完成单元 = has_finish 或 has_result（结果已回）；当前 task 除外。
    has_result = {r.metadata.get("origin_task_id") for r in recs
                  if r.role == "tool" and r.metadata.get("origin_task_id") is not None}
    active = has_dispatch - has_finish - has_result
    current = getattr(state.scope, "task_id", None)
    top = [oid for oid, pid in parent_of.items()
           if oid != current and oid not in active and (pid is None or pid not in origins)]
    top.sort(key=lambda oid: first_ts[oid])
    return top[-keep_last:] if keep_last > 0 else []


async def escalating_compact(
    state: LoopState, ctx: LoopContext, *, token_estimate: int, trigger: str = "compact"
) -> list[Any]:
    """预算驱动升级式 compact（替 _compact_scope）：L0.5 图片降级 → L1 agent 折 → L2 rich→lean → L3 坍当前 task，
    每级后用 _active_memory_tokens 的增量从 token_estimate 累减，降到 target 以下即停。
    级间不完整重装配（Q4=c，调用方进 act 前重装配一次校正）。无 context_limit 或已达标 → []。"""
    agent = state.agent
    lc = agent.loop_config
    context_limit = agent.loop_guard.context_limit
    reserve = getattr(agent.loop_guard, "reserved_output_tokens", 0)
    eff = effective_limit(context_limit, reserve)
    if eff <= 0:
        return []
    target_ratio = lc.compact_target_ratio if getattr(lc, "compact_target_ratio", 0.0) > 0 \
        else lc.compact_token_ratio
    target_tokens = int(eff * target_ratio)
    keep_last = lc.compact_keep_last
    collapse_keep = getattr(lc, "collapse_keep_last", keep_last)
    # §12 未决参数，暂定 2 —— 从 loop_config 读（与 compact_keep_last 等同构），不硬编码。
    keep_recent_images = getattr(lc, "compact_keep_recent_images", 2)

    est = token_estimate
    if est < target_tokens:
        return []
    # MemoryCompactStarted：有 event_bus 时立即 live 发（folds 含 LLM 摘要、耗时数秒；随批次事后发
    # 会让前端状态条错过整个「压缩中」窗口）。无 bus（单测）时退回塞进返回批次，保持既有契约与用例。
    started = make_event(state, EventType.MEMORY_COMPACT_STARTED, payload={
        "task_id": state.task.id, "agent_id": agent.id, "trigger": trigger,
        "token_estimate": est, "target_tokens": target_tokens})
    bus = getattr(ctx, "event_bus", None)
    events: list[Any] = []
    if bus is not None:
        await bus.emit(started)
    else:
        events.append(started)

    def _finish(evts: list[Any]) -> list[Any]:
        """收尾：聚合本轮各级 MemoryCompacted，追加一条 MemoryCompactFinished。"""
        folded = [e for e in evts if e.type == EventType.MEMORY_COMPACTED]
        evts.append(make_event(state, EventType.MEMORY_COMPACT_FINISHED, payload={
            "task_id": state.task.id, "agent_id": agent.id, "trigger": trigger,
            "total_superseded": sum(e.payload.get("superseded_count", 0) for e in folded),
            "freed_tokens": sum(e.payload.get("freed_tokens", 0) for e in folded),
            "levels": [e.payload.get("source", "") for e in folded],
            "est_before": token_estimate, "est_after": est, "target_tokens": target_tokens}))
        return evts

    last_tokens: int | None = None

    async def _apply(level_coro):
        """跑一级折叠，用活跃 token before/after 增量累减 est。返回 (superseded_count, freed)。

        before 惰性复用上一级的 after（相邻级间省一次重复测量）；首次或跳级后重新测量。"""
        nonlocal est, last_tokens
        # 不变量：前一级的 after 直接当这一级的 before 复用，只在 _active_memory_tokens 是纯快照读、
        # 且两次 _apply 之间没有其他改动内存的操作时才成立。未来若在级间插入其他写操作，须重新测量。
        # 故 L1 的「前置降级 + 摘要 + fold」整体作为**一个** level_coro 交给本函数（Task 5b）：
        # 把降级提到 _apply 之外会破坏这个前提，且降级省下的 token 要么被重复计入、要么彻底丢掉。
        before = last_tokens if last_tokens is not None else await _active_memory_tokens(state, ctx)
        n = await level_coro if inspect.isawaitable(level_coro) else level_coro
        after = await _active_memory_tokens(state, ctx)
        freed = max(0, before - after)
        est -= freed
        last_tokens = after
        return n, freed

    # L0.5 · 图片降级（子设计 §6：排在所有折叠之前）
    # 三条理由：无 LLM（不花一次调用）、单位收益最高（一张图按当前口径最低 1600 token，
    # 满额 5MB 图 40960）、**可逆**（占位仍在原位，模型随时 media:get_image 取回）。
    # L1/L2/L3 折的是记录本身，一旦执行位置就没了；所以先花可逆的额度。
    # 未接 MemoryBlobStore 时 _media_enabled 直接短路，本级连一次 memory 读都不发（§10）。
    if _media_enabled(ctx):
        n, freed = await _apply(demote_for_budget(
            ctx.memory, _agent_half(state.scope), ctx.provider_ctx,
            keep_recent=keep_recent_images, scope=MemoryScope.TASK,
            kinds=_TASK_VIEW_KINDS))
        if n:
            # 注：本级的 n 是**图片张数**，不是记录条数（记录并没有被折走，只是被重写）。
            # 仍填进 superseded_count 以与各级同构、让 MemoryCompactFinished 的聚合口径
            # 不必分叉；真正无歧义的计数在 demoted_images。
            events.append(make_event(state, EventType.MEMORY_COMPACTED, payload={
                "superseded_count": n, "demoted_images": n, "layer": "task",
                "source": "demote_images", "trigger": trigger, "freed_tokens": freed}))
    if est < target_tokens:
        return _finish(events)

    # L1 · agent 折（仅当有可折顶层单元）
    if await _count_root_residues(state, ctx) > keep_last:
        # 🔴 传的是「取摘要的函数」而不是摘要文本（Task 5b，修 P4-L10）：摘要必须在
        # `fold_root_experience` 内部的 §6.1 前置降级**之后**才生成，否则折区里的真图
        # 先被拍扁进摘要输入（连 ref 都不剩）再被 supersede。折区 ids 只有函数内部
        # 算得出，故把摘要推迟进去，而不是把降级提到外面来。
        # 这样整级（降级 + 摘要 + fold）仍是交给 `_apply` 的**一步**，`_apply` 的
        # 「前一级 after 复用为本级 before」不变量不受影响，降级省下的 token 也自然
        # 计进本级的 freed（既不重复计入 L0.5，也不丢）。
        n, freed = await _apply(fold_root_experience(
            state, ctx, keep_last,
            lambda: summarize_for_compact(state, ctx, scope="agent")))
        if n:
            events.append(make_event(state, EventType.MEMORY_COMPACTED, payload={
                "superseded_count": n, "layer": "agent", "source": "root_experience",
                "trigger": trigger, "freed_tokens": freed}))
    if est < target_tokens:
        return _finish(events)

    # L2 · 保留的同 agent rich 胶囊降级 lean（无 LLM）——**从最老一颗开始逐颗降，每降一颗
    # 就重新比一次 target，达标即停**。曾是整批一次降完：哪怕降一颗就够，L1 保下的全部胶囊也
    # 一起被掏空（USER_PROMPT 原文、段摘要、act_recap 全删，只剩综合总结）。逐颗降级让最近
    # 完成的胶囊尽量保持 rich——它们被后续回合引用的可能最大。
    # 每颗各走一次 `_apply`：相邻两次之间没有别的写操作，「上一次 after 复用为本次 before」
    # 的不变量成立；事件仍按级聚合成一条，`MemoryCompactFinished.levels` 不因颗数膨胀。
    l2_n = l2_freed = l2_capsules = 0
    for oid in await _kept_origin_ids(state, ctx, keep_last):
        if est < target_tokens:
            break
        n, freed = await _apply(demote_kept_capsules(state, ctx, {oid}))
        if n:
            l2_n += n
            l2_freed += freed
            l2_capsules += 1
    if l2_n:
        events.append(make_event(state, EventType.MEMORY_COMPACTED, payload={
            "superseded_count": l2_n, "layer": "agent", "source": "demote_lean",
            "trigger": trigger, "freed_tokens": l2_freed, "demoted_capsules": l2_capsules}))
    if est < target_tokens:
        return _finish(events)

    # L3 · 坍缩当前 task（段摘要坍成更少，保 collapse_keep 条；仅当有 task 层材料可折）
    # 计数用 TASK 视图里**会被渲染**的记录（对话回合 + SUMMARY 段摘要）——与
    # collapse_task_layer 的保留名额同口径：retry 累积的是段摘要，只数 raw 会漏计、L3 永不
    # 触发；而 TOOL_AUDIT 不进 prompt，算进来会让门槛虚高、提前开门。
    task_n = len(_rendered_only(await ctx.memory.load_view(
        state.scope, MemoryScope.TASK, ctx.provider_ctx, kinds=_TASK_VIEW_KINDS)))
    if task_n > collapse_keep:
        # 传「取摘要的函数」而非摘要文本（同 L1）：按回合切之后 L3 可能发现无可折之物而
        # no-op，先算好就白烧一次 LLM；且折区图片降级发生在 collapse_task_layer 内部，
        # 摘要必须在那之后生成，否则摘要输入里仍是真图。
        n, freed = await _apply(collapse_task_layer(
            state, ctx, collapse_keep,
            lambda: summarize_for_compact(state, ctx, scope="task")))
        if n:
            events.append(make_event(state, EventType.MEMORY_COMPACTED, payload={
                "superseded_count": n, "layer": "task", "source": "collapse",
                "trigger": trigger, "freed_tokens": freed}))
    logger.info("escalating_compact[%s]: agent=%s task=%s est→%d target=%d",
                trigger, agent.id, state.task.id, est, target_tokens)
    return _finish(events)


class CompactStep(Step):
    """Standalone compaction over state.scope. Invoked inline by PrepareStep."""

    name = "compact"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        return StepOutcome(
            next_step=None,
            events=await escalating_compact(
                state, ctx, token_estimate=state.agent.loop_guard.context_tokens,
                trigger="compact"),
        )
