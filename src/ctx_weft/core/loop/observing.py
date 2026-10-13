"""观察这件事的两个共用件：ReAct 循环，和「这个 agent 有没有 observer」。

前台 `ObserveStep` 与后台 `background.recap` 都要它们。2026-09-29 从 `steps/observe.py` 提
出来——此前 recap 在模块级 import observe、observe 又在函数内 import background，是这一层
最后一处靠局部 import 撑着的环。提出来之后依赖单向：observe / background 都只依赖本模块。
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from ctx_weft.protocols import LLMMessage, LLMRequest
from ctx_weft.providers.llm.stream import ensure_collected_stream
from ctx_weft.protocols.events import EventType
from ctx_weft.core.loop.driver import make_event
from ctx_weft.core.loop.llm_gateway import (
    request_prompt_estimate, resolve_llm_identity, stream_llm_resilient,
)
from ctx_weft.core.utils.ids import generate_id, mint_turn_call_ids

if TYPE_CHECKING:
    from ctx_weft.core.capabilities.control_tools import ControlResult

logger = logging.getLogger(__name__)


# ── 有没有 observer 这个人格 ──────────────────────────────────────────────────


def has_observe_role(state: "Any") -> bool:
    """这个 agent 的模板配了 observe ROLE facet 吗。

    **「有没有 observer」的唯一判据**，三个下游共用，缺一不可（2026-09-28）：

    - 本文件 `_should_use_llm`：没有 → 前台 observe 走机械判决。
    - `background_observe`：没有 → 判定档降成只摘要档（`_judges` 与它相与）。
    - `act._park_await_user` 的 `finish_park` 分支：没有 → **不让位**。这条容易漏而且漏了
      要命——让位之后 task 的默认态是 park 等人（`_submit_verdict`：拿不到判决就维持
      park），没人判就永远醒不过来，root 再也不会自己 FINISHED。

    2026-09-28 之前缺 ROLE 的模板会从 default 模板借一份（那份默认合并已随之移出 core，
    归 host），所以这个判据几乎永远为真；兜底删掉之后它成了一个真实配置，上面三处必须同时认它。

    **正文非空才算数**，与装配层同一口径：composer 拿到空正文的 facet 会当它不存在、改用
    `_OBSERVER_ROLE_*_FALLBACK`。只判「键在不在」会让两处对同一个模板给出不同答案——判定档
    开着跑，而 ROLE 位上其实是框架兜底文案。目录 loader 产生不了空正文 facet（正文空就不写
    key），host 自实现的 provider 与测试替身可以。
    """
    template = state.extra.get("template")
    if template is None:
        return False
    facet = template.identity.get("observe")
    # getattr：与 composer 取 facet 正文的口径一致（手构 request / 鸭子类型替身在本仓很常见）
    return facet is not None and bool((getattr(facet, "text", "") or "").strip())


# ── Shared ReAct helper ───────────────────────────────────────────────────────


async def run_observe_react(
    state: "Any",
    ctx: "Any",
    *,
    system: str,
    messages: "list[LLMMessage]",
    tools: "Any",
    max_rounds: int,
    terminal_tool_name: str,
) -> "tuple[ControlResult | None, str]":
    """共用 observe/background ReAct：跑多轮 LLM，指定 terminal_tool 被调用时返回其完整 ControlResult 终止。

    返回 (terminal_result, last_text)：
      terminal_result — terminal_tool_name 被调用时的完整 ControlResult（未调用则 None）。
      last_text       — 最后一轮的纯文本。
    非 terminal 控制工具（如 ask_user）只执行副作用，不终止循环。
    不解读 verdict、不写 task 状态（状态写是工具副作用，由调用方绑定的工具决定）。

    Task 5：不再有 event_types 间接层——「流式侧」4 种事件（REQUEST_STARTED/PROMPT_SENT/
    TOKEN_STREAMED/REASONING_STREAMED）统一由 llm_gateway.stream_llm_resilient 发射
    LLM_*，observe 前台与 background observe 之间只靠 state.origin
    （EventOrigin.LOOP_OBSERVE / LOOP_BACKGROUND_OBSERVE）区分——由调用方在调用本函数前
    设好 state.origin（observe 前台由 driver 按 step 名设好；background_observe 在
    launch 出的快照 state 上显式改写）。本函数自己只补发收尾的 LLM_RESPONSE_FINISHED
    （payload 结构对齐 act.py._run_llm_turn），否则 gateway 发的 REQUEST_STARTED 会等不到
    收尾（host SSE 侧的挂死请求，Task 4 在 compact 上踩过同样的坑）。
    """
    agent = state.agent
    model, llm_account = resolve_llm_identity(state)
    current_messages = list(messages)
    last_text = ""
    # 动态 max_tokens 的增量基线：上一轮实际发送条数（本轮 usage 对应的真实 prompt 基线）。
    baseline_msg_count: int | None = None

    for round_num in range(max_rounds):
        # req_id：与 llm_gateway.stream_llm_resilient 内部同一确定性公式独立算出（同
        # act.py._run_llm_turn 的手法）——两边都在本次 LLM 调用任何事件发射前求值，故
        # state.sequence_counter 两处读到同一个值，天然一致，不需要新增参数或跨函数传值。
        req_id = f"req_{agent.id}_{state.sequence_counter}"

        sent_msg_count = len(current_messages)  # 本轮发送条数（append 前）→ 下轮增量基线
        llm_request = LLMRequest(
            model=model,
            system=system,
            messages=list(current_messages),
            tools=tools,
        )
        llm_request.prompt_token_estimate = request_prompt_estimate(
            ctx.llm.tokenizer, llm_request, getattr(agent, "loop_guard", None), baseline_msg_count)
        # turn：gateway 从 metadata["turn"] 取（spec §9.4 统一口径：LLM_* 事件一律用 turn，
        # 不用 round）；round_num 仍是本函数内部的轮次计数局部变量，只是不再直接进 payload。
        llm_request.metadata["turn"] = round_num

        stream = ensure_collected_stream(stream_llm_resilient(ctx, state, llm_request))
        async with aclosing(stream):
            async for _ in stream:
                stream.accept_current()
        result = stream.get_final_result()
        accumulated_text, reasoning_text = result.text, result.reasoning
        tool_calls, usage = list(result.tool_calls), result.usage

        if accumulated_text:
            last_text = accumulated_text

        # 摄入前铸造内部调用标识（spec: conversation-integrity，与 act._run_llm_turn 同口径）：
        # observer 回合不入 task 层 memory（工具是 SILENT/控制面），锚仅取唯一性、不持久；
        # live 消息面（current_messages 的 tool_call↔result 配对）与事件 payload 用同一份值。
        if tool_calls:
            tool_calls = [
                m.call for m in mint_turn_call_ids(
                    tool_calls, anchor=generate_id("asst"), turn_seq=round_num)
            ]

        # 更新 loop_guard（对齐 miniAgents _run_observer：取 actor/observer 的最大值）
        if usage.prompt_tokens > 0:
            agent.loop_guard.context_tokens = max(
                agent.loop_guard.context_tokens, usage.prompt_tokens
            )
            baseline_msg_count = sent_msg_count  # 真实刷新才前移基线（否则下轮退回整份估算）
        # 同步累加 session.token_used
        state.session.token_used += usage.prompt_tokens + usage.completion_tokens

        await ctx.event_bus.emit(make_event(
            state, EventType.LLM_RESPONSE_FINISHED,
            payload={
                "request_id": req_id,
                "content": accumulated_text,
                "reasoning": reasoning_text,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                ],
                "usage": dataclasses.asdict(usage),
                # 本次调用实际使用的模型/账号（host 云端上报按此计账，不受切换竞态影响）
                "llm_model": model,
                "llm_account": llm_account,
                "finish_reason": "tool_use" if tool_calls else "stop",
                "turn": round_num,
            },
        ))

        if not tool_calls:
            # 纯文本轮不直接放弃：observer 常把复述写成正文而忘了调 terminal 工具——
            # 还有剩余轮次时催促其改用工具提交后重试；耗尽轮次才返回 (None, last_text)。
            if round_num + 1 >= max_rounds:
                break
            current_messages.append(LLMMessage(
                role="assistant", content=accumulated_text or "(no reply)",
                provider_state=result.provider_state,
            ))
            current_messages.append(LLMMessage(
                role="user",
                content=(f"Submit the summary above by calling the `{terminal_tool_name}` "
                         "tool (put the content in the tool arguments); do not reply in "
                         "plain text."),
            ))
            continue

        current_messages.append(LLMMessage(
            role="assistant",
            content=accumulated_text,
            tool_calls=[{"id": tc.id, "name": tc.name, "input": tc.arguments} for tc in tool_calls],
            provider_state=result.provider_state,
        ))

        terminal_result = None
        for tc in tool_calls:
            if ctx.capability_gateway is not None:
                result = await ctx.capability_gateway.invoke(
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    state=state,
                    ctx=ctx,
                    tool_call_id=tc.id,
                )
                content = result.content
                # terminal 工具的失败（is_error，如参数非法）不终止循环：错误内容照常 append
                # 进 messages（下方）供模型下一轮改参重试；只有成功结果才是终止结果。否则一次
                # 坏调用的错误文案会被当成 Process Report 终结整个观察（spec: capability-gateway）。
                if tc.name == terminal_tool_name and not result.is_error:
                    terminal_result = result
            else:
                logger.warning(
                    "run_observe_react: no CapabilityGateway for tool '%s'", tc.name
                )
                content = f"[Error: CapabilityGateway not configured, tool '{tc.name}' skipped]"
            current_messages.append(LLMMessage(
                role="tool",
                content=content,
                tool_call_id=tc.id,
            ))

        if terminal_result is not None:
            return terminal_result, last_text

    return None, last_text
