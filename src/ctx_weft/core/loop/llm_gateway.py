"""LLM 请求发送的统一关口：集中所有「发送前合法化」方法 + 流式转发。

四个 step（act / observe / compact / recognize_intent）各自构造 ``LLMRequest`` 后
都经 :func:`stream_llm` 发送，使「发送前合法化」成为**唯一**关口：装配层（composer）
只负责构造逻辑消息结构，不做合法化；连续同角色 / 孤立 tool result 一律在这里、发送前
统一处理。

合法化六条不变式（:func:`legalize_messages` 内顺序：剥离悬挂 tool_call → 把 tool result 挪到
其 tool_call 之后 → 删空 content → 丢前导非 user → 丢孤立 tool result → 合并连续同角色）：
  1. :func:`drop_dangling_tool_calls` —— 剥离无后继 tool_result 配对的 assistant tool_call。
     reconcile/act 正常已补齐 dangling（见模块尾注），此处是发送前最后**防御性**兜底：只删
     不补、命中即打 ERROR 日志（说明上游对账漏了）。Anthropic 对悬挂 tool_use 直接 400。
  2. :func:`reorder_tool_results_after_calls` —— 把每条 tool result 紧挪到其 assistant
     tool_call 之后。跨层按 timestamp 归并会把 user 回合插进 assistant↔result 之间（OpenAI
     兼容端点据此 400），集合级配对查不出这种错位——本步按 owner 重排消除中间夹角色。只搬不删。
  3. :func:`remove_empty_messages` —— 删除内容为空且无 tool_calls 的 user/assistant 消息
     （Anthropic 对空 content 块 400）。同为防御性兜底，命中打 ERROR 日志。① 把全悬挂的
     assistant 剥成空消息后，正好由此清理。tool 消息即便空也不在此删（删了会制造悬挂）。
  4. :func:`ensure_leading_user` —— 丢弃开头 role != "user" 的消息直到首条为 user（Anthropic
     硬规则，否则 400）。丢弃前导 assistant 后其配对 tool 会成孤儿，由下一步清理。
  5. :func:`drop_orphan_tool_results` —— 丢弃 tool_call_id 无前序 assistant tool_call
     配对的 ``role="tool"`` 消息。compact 丢弃 assistant 但保留 TOOL_RESULT、recall 窗口
     在 assistant↔result 之间截断、跨层按 timestamp 归并错位等都会产生此类孤儿，会被
     Anthropic / OpenAI 直接 400。
  6. :func:`merge_consecutive_messages` —— 合并连续同角色消息（多模态安全）。

本模块仅依赖 protocols，且只被本 loop 层的 step import；故置于 ``core/loop`` 下。若将来
装配等下层也需复用这些合法化函数，应改为下沉到 ``core`` 根，避免下层反向依赖 loop。

注：「悬挂 assistant tool_call（无后继 result）」的**正路**是上游对账补 TOOL_RESULT
（resume 前的 ReconcileStep 重跑工具、act 为被打断工具补 result），本关口的
:func:`drop_dangling_tool_calls` 只是发送前防 400 的兜底，命中打 ERROR 提示上游漏补。
adapter 不应重复实现以上不变式——已在 core 层集中保证。

另注：「prompt 必须以 user 收尾」不下沉到此处——那是 act 装配期的语义兜底（注入具体
文案、且 provider 并不强制），若在此每个 turn 强制，会在工具循环里于 tool result 之后误
插一条 user，篡改正常的工具调用流。该兜底保留在 composer._build_actor_messages。
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
from collections.abc import AsyncIterator
from time import monotonic
from typing import TYPE_CHECKING, Any

from ctx_weft.protocols import LLMMessage, LLMOutageError, TextPart
from ctx_weft.core.utils.content import rehydrate_content, redact_content_for_event
from ctx_weft.protocols.events import EventType
from ctx_weft.core.loop.driver import make_event
from ctx_weft.core.utils.estimate import dynamic_max_tokens, estimate_content_tokens, estimate_tool_calls_tokens
from ctx_weft.providers.llm.stream import DelegatingLLMStream
if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart, LLMChunk, LLMClient, LLMRequest

logger = logging.getLogger(__name__)

# request.metadata 瞬态键：request_prompt_estimate 写入的估算基线（真实 context_tokens 或 0），
# 供 act 回喂时算「真实段 = usage.prompt_tokens − 基线」。
PROMPT_EST_BASE_KEY = "prompt_est_base"

# request.metadata 瞬态键：本次请求的「估算段」——增量路径为 delta、整份路径为 full，两者都是
# tokenizer.count 的直接产出（未经 max(..., context_tokens) 的 floor）。act 回喂必须用这个值而
# 非返回值：返回值在整份路径上可能被 floor 成上一轮的真实 context_tokens（当启发式低估、被
# 真实基线 floor 时），若拿 floor 后的返回值回喂，ratio≈1 的假样本会把伺服系统性拖向 1——
# floor 只应影响 max_tokens 的保守性，不应污染校准。
PROMPT_EST_SEG_KEY = "prompt_est_seg"



def resolve_llm_identity(state) -> tuple[str, str]:
    """本次 LLM 调用实际使用的 (model, account)。

    真值是 state.resolved_model —— 派发时由 AgentLifecycleManager 解出的那一个。
    此前读 session.llm_model 并两级兜底到 agent.runtime / "mock"，那两级
    永远命中不了（runtime 从不填 agent.runtime["llm_model"]），于是未配置
    时恒报 "mock"。ResolvedModel 永远是解析过的确定值，报不出假数据。
    """
    rm = state.resolved_model
    if rm is None:
        raise RuntimeError(
            "resolve_llm_identity: state.resolved_model must be set before dispatch"
        )
    return rm.model, rm.account


def drop_dangling_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """剥离无后继 tool_result 配对的 assistant tool_call（防御性，命中打 ERROR 日志）。

    正路是上游对账补齐（reconcile/act）；此处仅发送前兜底，只删不补——保证不 400，但打
    ERROR 提示上游漏补。多 tool_call 的 assistant 只剥悬挂的那几个，保留有 result 的；全悬挂
    则剥成空 tool_calls（随后由 :func:`remove_empty_messages` 清理）。绝不动有 result 的调用，
    故不会反向制造孤立 tool result。
    """
    resolved = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            kept = [tc for tc in m.tool_calls if tc.get("id") in resolved]
            if len(kept) != len(m.tool_calls):
                dangling = [tc.get("id") for tc in m.tool_calls if tc.get("id") not in resolved]
                logger.error(
                    "stream_llm: dropping %d dangling tool_call(s) with no tool_result "
                    "(上游对账漏补？): %s",
                    len(dangling), dangling,
                )
                m = LLMMessage(
                    role=m.role, content=m.content, tool_calls=kept,
                    tool_call_id=m.tool_call_id, reasoning_content=m.reasoning_content,
                    provider_state=m.provider_state,
                )
        out.append(m)
    return out


def _is_empty_content(content: "str | list[ContentPart]") -> bool:
    """内容是否为「空」：空串/纯空白，或空列表/仅含空白 TextPart。非文本块（图片等）视为有内容。

    与 composer._is_blank_content 语义相近但**不同**——本函数对字符串/TextPart 文本都
    做 ``.strip()``（纯空白 "   " 判空），composer 那份不 strip（纯空白判非空）。两者共存
    今日安全仅因为 legalize_messages 链路最终经本函数把纯空白消息滤掉；composer 侧的
    "   " 不是本函数意义上的「空」不会传导成 bug，纯属两处判据本就不追求一致——但这是个
    没写下来的非局部不变式，改动任一处前请先看 tests/unit/test_composer_vs_gateway_blank_content.py
    钉住的那对语义。"""
    if isinstance(content, str):
        return not content.strip()
    if not content:
        return True
    for p in content:
        if isinstance(p, TextPart):
            if p.text and p.text.strip():
                return False
        else:
            return False
    return True


def remove_empty_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """删除内容为空且无 tool_calls 的 user/assistant 消息（防御性，命中打 ERROR 日志）。

    Anthropic 对空 content 块直接 400。只管 user/assistant：纯 tool_use 的 assistant（有
    tool_calls、无文本）合法须保留；tool 消息即便空也不删（删了会制造悬挂 tool_call）。
    """
    out: list[LLMMessage] = []
    for m in messages:
        if m.role in ("user", "assistant") and not m.tool_calls and _is_empty_content(m.content):
            logger.error(
                "stream_llm: dropping empty-content %s message (no text, no tool_calls)", m.role
            )
            continue
        out.append(m)
    return out


def drop_orphan_tool_results(messages: list[LLMMessage]) -> list[LLMMessage]:
    """丢弃 tool_call_id 无前序 assistant tool_call 配对的孤立 tool result 消息。"""
    seen: set[str] = set()
    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "assistant":
            seen.update(tc.get("id", "") for tc in m.tool_calls if tc.get("id"))
        elif m.role == "tool":
            if not m.tool_call_id or m.tool_call_id not in seen:
                continue  # 孤儿 → 丢弃
        out.append(m)
    return out


def _merge_message_content(
    prev: "str | list[ContentPart]", cur: "str | list[ContentPart]"
) -> "str | list[ContentPart]":
    """连接两条消息的内容。两侧皆为 str → 以空行拼接（保留原文本语义）；任一侧为多模态
    时 → 拼成 ContentPart 列表（str 侧转 TextPart），避免把图片拍扁丢弃。"""
    if isinstance(prev, str) and isinstance(cur, str):
        return f"{prev}\n\n{cur}"

    def as_parts(c: "str | list[ContentPart]") -> "list[ContentPart]":
        return [TextPart(text=c)] if isinstance(c, str) else list(c)

    return as_parts(prev) + as_parts(cur)


def ensure_leading_user(messages: list[LLMMessage]) -> list[LLMMessage]:
    """丢弃开头 role != "user" 的消息直到首条为 user（Anthropic 首条必须 user，否则 400）。

    只动头部、不注入文案——与「以 user 收尾」语义兜底（保留在 composer）不同，对正常工具
    循环（中段 tool result 之后无 user）无影响。前导 assistant 被丢后其配对 tool 会成孤儿，
    由随后的 drop_orphan_tool_results 清理（见 stream_llm 串联顺序）。
    """
    i = 0
    while i < len(messages) and messages[i].role != "user":
        i += 1
    return messages[i:] if i else messages


def reorder_tool_results_after_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """把每条 tool result 紧挪到其 assistant tool_call 之后，消除中间夹着的非 tool 消息。

    OpenAI 兼容端点要求 ``role="tool"`` 紧跟在 ``role="assistant"(tool_calls)`` 之后、按
    ``tool_call_id`` 配对，中间不得夹其它角色（夹了即 400「insufficient tool messages
    following tool_calls message」）。跨层历史按 timestamp 归并会把 user 回合插进
    assistant↔tool_result 之间，甚至让 result 落到 call 之前；集合级配对
    （:func:`drop_dangling_tool_calls` / :func:`drop_orphan_tool_results`）只校验「id 是否
    存在」查不出这种**错位**。本步按 owner 把每个 tool result 移到其 assistant tool_call 之后，
    并按该 assistant 的 ``tool_calls`` 顺序排列。无 owner（没有任何 assistant 调用该 id）的
    孤儿 tool 原地保留，交 :func:`drop_orphan_tool_results` 处理；只搬不删、不补。

    存量歧义留痕（spec: conversation-integrity）：新数据的 id 在摄入点铸为全局唯一，
    命中「同一 id 被多个 assistant 携带」只可能来自改造前的裸 wire id 记录——行为维持
    现状（结果在每个携带者后各发一遍），但 MUST 留痕不静默。
    """
    owner_counts: dict[str, int] = {}
    for m in messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                tcid = tc.get("id")
                if tcid:
                    owner_counts[tcid] = owner_counts.get(tcid, 0) + 1
    dup_ids = [tcid for tcid, n in owner_counts.items() if n > 1]
    if dup_ids:
        logger.error(
            "stream_llm: %d tool_call_id(s) each claimed by multiple assistant messages "
            "(legacy bare-wire ids? their results will be duplicated after every owner): %s",
            len(dup_ids), dup_ids,
        )
    owned_ids = {
        tc.get("id")
        for m in messages
        if m.role == "assistant"
        for tc in m.tool_calls
        if tc.get("id")
    }
    results_by_id: dict[str, list[LLMMessage]] = {}
    for m in messages:
        if m.role == "tool" and m.tool_call_id in owned_ids:
            results_by_id.setdefault(m.tool_call_id, []).append(m)

    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "tool" and m.tool_call_id in owned_ids:
            continue  # 由其 owner assistant 之后统一发出（见下）
        out.append(m)
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                out.extend(results_by_id.get(tc.get("id"), ()))
    return out


def merge_consecutive_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """合并连续相同角色的消息（tool 消息因绑定 tool_call_id 不合并）。"""
    merged: list[LLMMessage] = []
    for msg in messages:
        if (
            merged
            and merged[-1].role == msg.role
            and msg.role != "tool"
        ):
            prev = merged[-1]
            merged[-1] = LLMMessage(
                role=prev.role,
                content=_merge_message_content(prev.content, msg.content),
                tool_calls=prev.tool_calls + msg.tool_calls,
                provider_state=_merge_provider_state(prev.provider_state, msg.provider_state),
            )
        else:
            merged.append(msg)
    return merged


def _merge_provider_state(left: dict, right: dict) -> dict:
    """Keep continuation metadata attached when legalizing consecutive assistants."""
    if not left or not right:
        return left or right
    merged = left | right
    if "anthropic" in left and "anthropic" in right:
        a, b = left["anthropic"], right["anthropic"]
        merged["anthropic"] = a | b
        for key in ("thinking_blocks", "tool_call_ids"):
            merged["anthropic"][key] = [*a.get(key, ()), *b.get(key, ())]
    return merged


def legalize_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """发送前把消息序列规整为 provider 合法形态（六条不变式，见模块 docstring）。

    纯函数、不依赖 llm/网络，便于单测全链路。``stream_llm`` 在发送前调用一次。
    """
    return merge_consecutive_messages(
        drop_orphan_tool_results(
            ensure_leading_user(
                remove_empty_messages(
                    reorder_tool_results_after_calls(
                        drop_dangling_tool_calls(messages)
                    )
                )
            )
        )
    )


def _estimate_message_tokens(m: LLMMessage, count) -> int:
    """单条消息的 provider 计费估算（往大了估）：文本 content + 图片 part + framing +
    tool_calls 参数 + reasoning_content。

    计费项定义在 core.utils.estimate（``estimate_content_tokens`` / ``estimate_tool_calls_tokens``）——
    单一真源，prepare/composer 对 memory 记录/装配消息共用同口径。tool_calls 的 arguments、
    reasoning、图片此前都没计入，是"单轮新增里一坨数不到的东西 > margin"致 400 的洞。

    ``count``：文本费率回调，caller 传 ``tokenizer.count``（已校准）。
    """
    total = estimate_content_tokens(m.content, count=count) \
        + estimate_tool_calls_tokens(m.tool_calls, count=count)
    if m.reasoning_content:
        total += count(m.reasoning_content)
    return total


def _estimate_request_tokens(request: "LLMRequest", count) -> int:
    """估算整份待发 prompt 的 token（system + 全部 messages + tools schema）。

    每条消息经 :func:`_estimate_message_tokens` 计全部计费项（文本 + tool_calls 参数 +
    reasoning + 图片 + framing）。含本轮新加的 role="tool" result。``count``：文本费率
    回调，caller 传 ``tokenizer.count``（已校准）。tools 面费率走单一真源
    ``estimate_tools_tokens``（与 prepare 首估同口径）。
    """
    from ctx_weft.core.utils.estimate import estimate_tools_tokens

    total = count(request.system or "")
    for m in request.messages:
        total += _estimate_message_tokens(m, count)
    total += estimate_tools_tokens(request.tools, count)
    return total


def request_prompt_estimate(tokenizer, request: "LLMRequest", loop_guard, baseline_msg_count: "int | None") -> int:
    """算 caller 侧的 used（本请求真实 prompt token 的最佳估算），供各 step 挂到 request。

    - **增量**（``baseline_msg_count`` 非 None 且有真实基线 ``context_tokens>0``）：
      真实基线 + 仅 ``messages[baseline_msg_count:]``（本轮新增尾段，如工具结果）的估算。
      基线前的历史采信 provider 真实测量值、不重估——避免 len//4 对 CJK 历史大头系统性低估
      （正是它把整份重估压到基线之下、漏掉本轮增量而导致 max_tokens 过大 400 的根因）。
      **工具面追踪（spec: tool-schema-budget）**：请求的工具面指纹（规范化定义哈希，
      覆盖 name+description+schema）与 guard 记录的上次指纹不同时，估算计入 schema 差值
      （新面估算 − 上次面估算）——运行期 pin 大 schema 后估算立即增长；工具面不变时
      增量行为与旧口径逐字节一致（纯消息增量）。首次记录（旧签名为空）不加减值——
      真实基线已含当时的工具面。
    - **首次 / 一次性**（无循环内基线）：``max(整份估算, context_tokens)``——整份估算打底，
      并不低于上一步真实测量值（更保守，永不 400；上步后若发生压缩，偏大只是少给输出）。

    估算全经 ``tokenizer.count``（已校准值）；不再有 raw/factor 概念——伺服校准下沉到
    adapter 的 ``LLMClient.tokenizer`` 内部（见 providers.llm.tokenizer.HeuristicTokenizer）。
    metadata 记两个瞬态键：:data:`PROMPT_EST_BASE_KEY`（估算基线）与 :data:`PROMPT_EST_SEG_KEY`
    （本次估算段，未经 floor 的 tokenizer.count 直接产出），供 usage 到达后 act 回喂
    ``tokenizer.observe`` 用——整份路径的返回值可能被 ``max(full, ctx_tokens)`` floor 成上一轮
    的真实值（正是本函数存在的校准动机场景：启发式低估、被真实基线兜住），若回喂用 floor 后
    的返回值会产出 ratio≈1 的假样本、把伺服系统性拖向 1；估算段是 floor 前的值，不受污染。
    """
    from ctx_weft.core.utils.estimate import estimate_tools_tokens, tools_signature

    ctx_tokens = getattr(loop_guard, "context_tokens", 0) if loop_guard is not None else 0

    def _track_tools() -> int:
        """指纹变化时返回 schema 差值（新 − 上次），并推进 guard 记录；不变返回 0。
        首次（旧签名为空）只记录不加值（真实基线已含当时的面）。guard 字段经
        getattr 读写——SimpleNamespace 测试替身与旧持久化快照都兼容。"""
        if loop_guard is None:
            return 0
        sig = tools_signature(request.tools)
        last_sig = getattr(loop_guard, "last_tools_signature", "")
        if sig == last_sig:
            return 0
        new_est = estimate_tools_tokens(request.tools, tokenizer.count)
        delta = 0
        if last_sig:
            delta = new_est - getattr(loop_guard, "last_tools_est", 0)
        try:
            loop_guard.last_tools_signature = sig
            loop_guard.last_tools_est = new_est
        except Exception:  # pragma: no cover —— 只读替身（防御，不影响估算本身）
            pass
        return delta

    if baseline_msg_count is not None and ctx_tokens > 0:
        delta = sum(
            _estimate_message_tokens(m, tokenizer.count)
            for m in request.messages[baseline_msg_count:]
        )
        delta += _track_tools()
        request.metadata[PROMPT_EST_BASE_KEY] = ctx_tokens
        request.metadata[PROMPT_EST_SEG_KEY] = delta
        return ctx_tokens + delta
    full = _estimate_request_tokens(request, tokenizer.count)
    _track_tools()
    request.metadata[PROMPT_EST_BASE_KEY] = 0
    request.metadata[PROMPT_EST_SEG_KEY] = full
    return max(full, ctx_tokens)


def apply_dynamic_max_tokens(ctx, request: "LLMRequest", loop_guard) -> None:
    """发送前按窗口就地写入 request.max_tokens（纯消费者，不自估）。

    仅当未显式设值（``max_tokens is None``）、有 loop_guard、且 caller 已挂
    ``prompt_token_estimate``（used，由 :func:`request_prompt_estimate` 算好）时生效——
    三者缺一即不动 max_tokens。天花板取 llm.output_ceiling（duck-type，缺省/None → 回退
    llm.context_limit）；margin/floor 从 ctx.config 取，缺省回退安全值。
    """
    if request.max_tokens is not None or loop_guard is None:
        return
    used = request.prompt_token_estimate
    if used is None:
        return
    # spec: tool-schema-budget——发送前超限不硬拒（明示决策，design D4）：估算存在误差，
    # 硬拒会误伤真实可发请求；超限信号转为 max_tokens 按含工具面的 used 自然收紧 +
    # WARNING 留痕 + 既有 compact 比例机制在下轮 prepare 触发压缩。provider 400 仍是
    # 最终防线（且自愈退避在）。
    reserve = max(0, getattr(loop_guard, "reserved_output_tokens", 0))
    eff_window = max(0, loop_guard.context_limit - reserve)
    if used > eff_window:
        logger.warning(
            "prompt estimate %d exceeds effective window %d "
            "(tools_signature=%s, tools_est=%d) — request proceeds with tightened "
            "max_tokens; compaction will trigger on next prepare if the ratio holds",
            used, eff_window,
            getattr(loop_guard, "last_tools_signature", ""),
            getattr(loop_guard, "last_tools_est", 0))
    llm = ctx.llm
    # margin 比例制：固定值只兜小 prompt 的估算残差，误差随体量按比例放大 → margin 也按比例。
    margin = max(
        int(_cfg_val(ctx, "dynamic_max_tokens_margin", 8192)),
        int(_cfg_val(ctx, "dynamic_max_tokens_margin_ratio", 0.05) * used),
    )
    floor = int(_cfg_val(ctx, "dynamic_max_tokens_floor", 1024))
    # 软顶：按 context_limit 比例封顶（不低于 output_min），再与硬上限 output_ceiling 取小。
    # 常态封住"整窗放输出"（省 token + 防 max_tokens 超模型输出上限的 400）；used 越大到
    # L-used-margin 跌破软顶时自然回落到紧缩段。
    ratio = float(_cfg_val(ctx, "dynamic_max_tokens_output_ratio", 0.2))
    min_out = int(_cfg_val(ctx, "dynamic_max_tokens_output_min", 4096))
    soft_cap = max(int(ratio * llm.context_limit), min_out)
    hard_cap = getattr(llm, "output_ceiling", None) or llm.context_limit
    ceiling = min(hard_cap, soft_cap)
    request.max_tokens = dynamic_max_tokens(
        loop_guard.context_limit, used, ceiling, margin=margin, floor=floor,
    )


def stream_llm(
    llm: "LLMClient", request: "LLMRequest", *, stream: bool = True,
    blob_store: "Any" = None, provider_ctx: "Any" = None,
) -> DelegatingLLMStream:
    return DelegatingLLMStream(lambda owner: _stream_llm(
        owner, llm, request, stream=stream, blob_store=blob_store, provider_ctx=provider_ctx))


async def _stream_llm(
    owner: DelegatingLLMStream,
    llm: "LLMClient", request: "LLMRequest", *, stream: bool = True,
    blob_store: "Any" = None, provider_ctx: "Any" = None,
) -> AsyncIterator["LLMChunk"]:
    """发送前合法化 ``request.messages``、把 blob ref 还原成 base64，再流式转发 chunk。

    ⚠️ **本函数不做任何模态门控**（spec 2026-08-28-multimodal-adapter-dispatch）：
    图片原样送到 ``LLMClient`` 面前，发多模态还是降级成占位由实现方决定。曾经这里
    有一个 ``_gate_tool_images``，只降 ``role == "tool"`` 的图——它的前提是「用户递的
    图在入口已被视觉门控拒掉」，入口门控删除后该前提不成立，覆盖面必须扩到所有角色，
    而那正是 adapter 的 ``_prepare_messages`` 在做的事。留在这里就是第二处会分叉的判据。

    已知代价：本函数对纯文本 adapter 一样会 rehydrate 每一个 blob ref（blob get +
    base64 编码，单图最大 5 MiB），rehydrate 完之后 adapter 的 ``_prepare_messages``
    才把它们降级成占位丢弃——这一趟读取白费了。这是「core 不判模态」换来的已知浪费；
    要消除它就得让本函数重新知道 adapter 的能力，那正是本设计明确拒绝重新引入的
    第二判据。不要在这里加判断。

    rehydrate 落在这里而非 adapter（架构裁定 T0）：adapter 的序列化链
    （``_build_payload`` / ``_serialize_messages`` / ``_parts_to_blocks``）全是同步
    函数，而 ``MemoryBlobStore.get`` 是 async。本函数是出网前最后一个 async 关口，一处
    覆盖三家 adapter。

    ``blob_store`` / ``provider_ctx`` 均**带默认值 None**：不传时整段 rehydrate 不
    执行，既有调用方与既有测试行为逐字节不变。刻意用显式参数而不是模块级单例——
    blob store 是 per-runtime 依赖，藏进全局状态会让测试互相污染。
    """
    request.messages = legalize_messages(request.messages)
    if blob_store is not None:
        request.messages = [
            dataclasses.replace(m, content=await rehydrate_content(
                m.content, blob_store=blob_store, ctx=provider_ctx))
            for m in request.messages
        ]
    source = owner.bind(llm.complete(request, stream=stream))
    try:
        async for chunk in source:
            yield chunk
    finally:
        await source.aclose()


# ── 自愈退避包装 ───────────────────────────────────────────────────────────────

# 默认值（ctx.config 缺省时回退；与 RuntimeConfig 默认一致）
_DEFAULT_MAX_ATTEMPTS = 8
_DEFAULT_MAX_DURATION_SEC = 300.0
_DEFAULT_BASE_DELAY_SEC = 2.0
_DEFAULT_MAX_INTERVAL_SEC = 60.0


def _cfg_val(ctx, name: str, default: float) -> float:
    cfg = getattr(ctx, "config", None)
    return getattr(cfg, name, default) if cfg is not None else default


def _is_outage(e: BaseException) -> bool:
    return bool(getattr(e, "outage", False))


def _compute_delay(
    attempt: int,
    *,
    base: float,
    max_interval: float,
    retry_after: "float | None",
) -> float:
    """指数退避延迟（attempt 1-based）；优先采用 Retry-After；上限 max_interval。"""
    if retry_after is not None and retry_after > 0:
        return min(retry_after, max_interval)
    return min(base * (2 ** (attempt - 1)), max_interval)


async def _sleep_cancellable(delay: float, tok) -> None:
    """Sleep up to *delay* seconds; if *tok* cancels first, raise CancelledError."""
    if tok is None:
        await asyncio.sleep(delay)
        return
    waiter = asyncio.ensure_future(tok.wait())
    try:
        await asyncio.wait({waiter}, timeout=delay)
    finally:
        if not waiter.done():
            waiter.cancel()
    tok.raise_if_cancelled()


async def _emit_retry(
    ctx,
    state,
    attempt: int,
    max_attempts: int,
    delay: float,
    error: BaseException,
) -> None:
    bus = getattr(ctx, "event_bus", None)
    if bus is None or state is None:
        return
    await bus.emit(make_event(state, EventType.LLM_RETRY_TRIGGERED, payload={
        "attempt": attempt,
        "max_attempts": max_attempts,
        "next_delay_sec": delay,
        "error_code": getattr(error, "status_code", 0),
        "error": str(error),
    }))


def stream_llm_resilient(ctx, state, request) -> DelegatingLLMStream:
    """Expose the current attempt's shared result without another collector."""
    return DelegatingLLMStream(lambda owner: _stream_llm_resilient(owner, ctx, state, request))


async def _stream_llm_resilient(owner, ctx, state, request) -> AsyncIterator["LLMChunk"]:
    """退避自愈包装：把 retriable+outage 的 LLM 调用在进程内指数退避重试。

    - 仅当本次尝试**尚未 yield 任何 chunk** 时重试（避免重复 token / 污染累积）。
    - 已 yield 后断流（中途 outage）→ 直接 LLMOutageError（靠 /resume 重驱动）。
    - 非 outage 的 retriable（如 _finalize 截断）与永久错 → 原样抛出（保留既有语义）。
    - 预算（max_attempts / max_duration）耗尽 → LLMOutageError。
    - 退避期间尊重 cancel_token。

    「流式侧」4 种 LLM_* 事件（REQUEST_STARTED / PROMPT_SENT / TOKEN_STREAMED /
    REASONING_STREAMED）在此发射（task-4 收敛，spec 2026-09-03 §9.5），对所有调用方
    无条件发射——不看 ``state.origin``。

    Task 4 曾在这里挂过一道临时门禁（``_STREAM_EVENT_ORIGINS``，只放行
    ``LOOP_ACT``/``LOOP_COMPACT``），因为当时 observe.py 的 ``run_observe_react``
    还在自己发一套同形事件（前台 LLM_*、后台 BACKGROUND_OBSERVE_*），放行会重复发射/
    串号。Task 5 删掉了 observe.py 那套自发射，改成所有调用方（act / compact /
    observe / background_observe）统一由本函数发 LLM_*，只靠 ``state.origin``
    （``EventOrigin.LOOP_OBSERVE`` / ``LOOP_BACKGROUND_OBSERVE`` 等）区分前台/后台
    （docs/events-v2.md §3.6：V2 之前这是靠 BackgroundObserve* 那族独立类型做的，
    合并后由 origin 承担）——门禁因此整个删除。

    ``LLM_RESPONSE_FINISHED`` 各调用方自己发射（依赖各自的收尾逻辑：act 依赖软打断
    决策，见 task-4 brief；compact 是单次调用直出；observe/background_observe 共用
    ``run_observe_react``，Task 5 起也在那里补发）。

    request_id：与 act.py 侧 ``_run_llm_turn`` 用同一个确定性公式
    ``f"req_{agent.id}_{state.sequence_counter}"`` 独立算出——两边都在「本次 LLM
    调用的任何事件被发射之前」求值（act.py 在调用本函数之前；本函数在 while 重试
    循环、也就是第一次 emit 之前），因此 ``state.sequence_counter`` 两处读到的是
    同一个值，算出来天然相等，不需要新增参数或跨函数传值。

    ``apply_dynamic_max_tokens`` 的调用位置（task-6 复审修复第二轮）：故意排在
    LLM_REQUEST_STARTED/LLM_PROMPT_SENT 发射**之后**，而不是紧挨在函数开头——这样
    「进了这段代码往下走」就蕴含「STARTED 已经发出」，调用方（如
    ``recognize_intent.py``）的收尾逻辑不必自己猜「STARTED 到底发没发」就能决定该不该
    补发 ``LLM_RESPONSE_FINISHED``。两个事件的 payload 都不读 ``request.max_tokens``，
    挪后不改变事件内容；``apply_dynamic_max_tokens`` 仍在真正发起 LLM 调用（下面
    ``stream_llm``）之前完成，行为不变。
    """
    from ctx_weft.protocols import LLMCallError  # local to avoid re-export confusion

    bus = getattr(ctx, "event_bus", None)
    emit_stream_events = bus is not None and state is not None
    request_id: str | None = None
    if emit_stream_events:
        agent = state.agent
        request_id = f"req_{agent.id}_{state.sequence_counter}"
        model, llm_account = resolve_llm_identity(state)
        turn = request.metadata.get("turn", 0)
        await bus.emit(make_event(state, EventType.LLM_REQUEST_STARTED, payload={
            "request_id": request_id, "model": model, "llm_account": llm_account,
            "turn": turn}))
        await bus.emit(make_event(state, EventType.LLM_PROMPT_SENT, payload={
            "request_id": request_id, "turn": turn, "system": request.system,
            "messages": [
                {"role": m.role, "content": redact_content_for_event(m.content)}
                for m in request.messages
            ],
            "tool_names": [t.name for t in request.tools]}))

    loop_guard = getattr(getattr(state, "agent", None), "loop_guard", None)
    apply_dynamic_max_tokens(ctx, request, loop_guard)

    max_attempts = int(_cfg_val(ctx, "llm_self_heal_max_attempts", _DEFAULT_MAX_ATTEMPTS))
    max_duration = _cfg_val(ctx, "llm_self_heal_max_duration_sec", _DEFAULT_MAX_DURATION_SEC)
    base = _cfg_val(ctx, "llm_self_heal_base_delay_sec", _DEFAULT_BASE_DELAY_SEC)
    max_interval = _cfg_val(ctx, "llm_self_heal_max_interval_sec", _DEFAULT_MAX_INTERVAL_SEC)

    deadline = monotonic() + max_duration
    tok = getattr(ctx, "cancel_token", None)
    attempt = 0
    while True:
        yielded_anything = False
        source = owner.bind(stream_llm(
            ctx.llm, request,
            blob_store=getattr(ctx, "blob_store", None),
            provider_ctx=getattr(ctx, "provider_ctx", None),
        ))
        try:
            try:
                async for chunk in source:
                    if emit_stream_events:
                        if chunk.kind == "token":
                            await bus.emit(make_event(
                                state, EventType.LLM_TOKEN_STREAMED,
                                payload={"request_id": request_id, "delta": chunk.text}))
                        elif chunk.kind == "reasoning":
                            await bus.emit(make_event(
                                state, EventType.LLM_REASONING_STREAMED,
                                payload={"request_id": request_id, "delta": chunk.text}))
                    yielded_anything = True
                    yield chunk
            finally:
                await source.aclose()
            return  # success
        except LLMCallError as e:
            if not getattr(e, "retriable", True) or not _is_outage(e):
                raise  # 永久错 / 非 outage retriable → 原样抛出
            if yielded_anything:
                raise LLMOutageError(f"LLM outage mid-stream: {e}") from e
            if tok is not None:
                tok.raise_if_cancelled()
            attempt += 1
            if attempt >= max_attempts or monotonic() >= deadline:
                raise LLMOutageError(
                    f"LLM self-heal exhausted after {attempt} attempt(s): {e}",
                    status_code=getattr(e, "status_code", 0),
                ) from e
            delay = _compute_delay(
                attempt,
                base=base,
                max_interval=max_interval,
                retry_after=getattr(e, "retry_after_sec", None),
            )
            delay += random.uniform(0, delay * 0.1)  # jitter
            await _emit_retry(ctx, state, attempt, max_attempts, delay, e)
            await _sleep_cancellable(delay, tok)
