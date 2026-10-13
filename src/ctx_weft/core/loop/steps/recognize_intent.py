"""RecognizeIntentStep: single-shot step that fills task title/description and session goal.

Launched concurrently with ActStep (see launch_recognize_intent) on a snapshot of the loop
state, reusing the capability set PrepareStep already bound. Assembles with
purpose="recognize_intent"; CapabilitySource auto-filters tools to update_task_metadata.
Skips entirely if the title is already set or no recognize_intent tool is bound.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from contextlib import aclosing
from typing import Any

from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome, make_event
from ctx_weft.core.loop.llm_gateway import (
    stream_llm_resilient, request_prompt_estimate, resolve_llm_identity,
)
from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.providers.llm.stream import ensure_collected_stream

logger = logging.getLogger(__name__)

# Strong refs to fire-and-forget tasks when no TaskManager is available (avoids GC).
_background_tasks: set[asyncio.Task] = set()


def should_recognize_intent(task: Any) -> bool:
    """True when this is the root task (no parent) and still lacks a title."""
    return getattr(task, "parent_task_id", None) is None and not getattr(task, "title", "")


def launch_recognize_intent(state: LoopState, ctx: LoopContext) -> asyncio.Task:
    """Fire-and-forget: run RecognizeIntentStep concurrently with act on a snapshot state.

    The snapshot shares agent/task/scope/session (read-mostly) but has its own run_id,
    sequence_counter and a copied extra dict, so it never corrupts the live act state.
    Returns the created asyncio.Task (callers may ignore it).
    """
    snapshot = LoopState(
        run_id=generate_id("run"),
        session=state.session,
        task=state.task,
        agent=state.agent,
        scope=state.scope,
        extra=dict(state.extra),
        resolved_model=state.resolved_model,
        # 孤儿 run：不走 StepDriver.run 的每步 origin 赋值机制（Task 2 的
        # `_STEP_ORIGIN`），构造时就得显式钉住，否则这里发的全部事件
        # （含下面 RUN_STARTED/RUN_FINISHED）origin 都会是空串。
        origin=EventOrigin.LOOP_RECOGNIZE_INTENT,
    )

    async def _run() -> None:
        # 总账 C5：这段快照造了自己的 run_id 却从没配起止事件——host 会看到凭空
        # 出现又凭空消失的 run。补齐（payload 结构照抄 `_run_loop` 的实际发射点，
        # 见 task-5-report）；这条路径没有 StepDriver 也没有 RunOutcome，
        # `initial_step` 用它实际跑的那个 step 名 "recognize_intent"。
        await ctx.event_bus.emit(make_event(snapshot, EventType.RUN_STARTED, payload={
            "run_id": snapshot.run_id,
            "initial_step": "recognize_intent",
        }))
        run_error: Exception | None = None
        was_cancelled = False
        try:
            await RecognizeIntentStep().execute(snapshot, ctx)
        except asyncio.CancelledError:
            # F2：CancelledError 不是 Exception 子类，下面的 except Exception 接不住——
            # 接不住则 run_error 仍是 None，RunFinished 会谎报 completed。记下来，供
            # finally 里的 RunFinished.outcome 用（在异常继续传播之前发出，见下）。
            #
            # 这里**要重新抛出**——与 runtime.py::_run_loop 的 except asyncio.CancelledError
            # 刻意不同，不是疏漏：_run_loop 吞是因为它把取消结果转成了 RunOutcome{kind=
            # CANCELED} 这个**返回值契约**塞回调用方；这里没有这种契约，虽是
            # fire-and-forget、没人 await 结果，但没理由让协作取消在这里被悄悄吸收掉。
            was_cancelled = True
            raise
        except Exception as exc:
            run_error = exc
            logger.exception("recognize_intent concurrent run failed (ignored)")
        finally:
            await ctx.event_bus.emit(make_event(snapshot, EventType.RUN_FINISHED, payload={
                "outcome": (
                    RunOutcomeKind.CANCELED.value if was_cancelled
                    else RunOutcomeKind.COMPLETED.value if run_error is None
                    else RunOutcomeKind.INTERRUPTED.value
                ),
                "final_status": snapshot.task.status,
                "will_retry": False,
                "total_events": snapshot.sequence_counter,
                "total_turns": len(snapshot.transcript),
                "error": str(run_error) if run_error else None,
                "error_type": type(run_error).__name__ if run_error else None,
            }))

    task = asyncio.create_task(_run())
    tm = getattr(ctx, "task_manager", None)
    if tm is not None and hasattr(tm, "track_background"):
        tm.track_background(task)
    else:
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    return task


class RecognizeIntentStep(Step):
    """Single-shot LLM step that enriches task metadata and session goal."""

    name = "recognize_intent"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        target_task = state.task
        target_task_id = target_task.id

        if target_task and target_task.title:
            await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_SKIPPED, payload={
                "task_id": target_task_id,
                "target_task_id": target_task_id,
            }))
            return StepOutcome(next_step=None)

        bound = list(state.extra.get("bound_capabilities", []))
        has_mf_tool = any(
            isinstance(c, ToolCapability) and "recognize_intent" in c.purposes for c in bound
        )
        if not has_mf_tool:
            logger.warning("RecognizeIntentStep: no recognize_intent tool bound for task %s", target_task_id)
            await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_SKIPPED, payload={
                "task_id": target_task_id,
                "target_task_id": target_task_id,
                "reason": "no_tools",
            }))
            return StepOutcome(next_step=None)

        # ── Assemble prompt via shared assembler (CapabilitySource filters by purpose) ──
        from ctx_weft.core.assembler.assembler import ContextRequest

        request = ContextRequest(
            purpose="recognize_intent",
            scope=state.scope,
            task=target_task,
            agent=state.agent,
            session=state.session,
            template=state.extra.get("template"),
            bound_capabilities=bound,
            token_counter=ctx.llm.tokenizer.count,
        )
        prompt = await ctx.assembler.assemble(request)

        from ctx_weft.protocols import LLMRequest

        llm_request = LLMRequest(
            model=resolve_llm_identity(state)[0],
            system=prompt.system,
            messages=prompt.messages,
            tools=prompt.tools,
        )

        await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_STARTED, payload={
            "task_id": target_task_id,
            "target_task_id": target_task_id,
        }))

        # req_id：与 gateway 内部用同一个确定性公式独立算出——两边都在「本次 LLM 调用的
        # 任何事件被发射之前」求值（此处在调用 stream_llm_resilient 之前；gateway 侧在其
        # while 重试循环、也就是第一次 emit 之前），故 state.sequence_counter 两处读到同一个
        # 值，天然一致（同 act.py::_run_llm_turn / compact.py::summarize_for_compact 的手法）。
        req_id = f"req_{state.agent.id}_{state.sequence_counter}"
        model, llm_account = resolve_llm_identity(state)

        from ctx_weft.protocols import ToolCall

        # 复审修复第二轮（task-6）：request_prompt_estimate 在 stream_llm_resilient 被
        # 调用**之前**跑——此刻 LLM_REQUEST_STARTED 不可能已经发出（它是 gateway 函数体
        # 内的第一段代码，函数还没被调用）。单独一个 try：失败仍按 pre-task-6 的既有语义
        # 降级返回（这段原本就在同一个吞异常的 try 里），但**不**补发
        # LLM_RESPONSE_FINISHED——否则会产出一条无 STARTED 匹配的孤儿事件（反向孤儿）。
        # 注：原来这里还有一次 apply_dynamic_max_tokens(ctx, llm_request, _guard) 调用，
        # 是切网关前的遗留——gateway 内部已经调它，这里纯属多余空操作，直接删掉（与
        # act.py / compact.py「完全委托 gateway」的既有约定一致）。
        _guard = getattr(state.agent, "loop_guard", None)
        try:
            llm_request.prompt_token_estimate = request_prompt_estimate(
                ctx.llm.tokenizer, llm_request, _guard, None)
        except Exception:
            logger.exception(
                "RecognizeIntentStep: prompt token estimate failed for task %s", target_task_id)
            return StepOutcome(next_step=None)

        # 这个 try 才是「STARTED 确定已发出」的区间：stream_llm_resilient 把
        # apply_dynamic_max_tokens 挪到了 LLM_REQUEST_STARTED/LLM_PROMPT_SENT 发射之后
        # （llm_gateway.py，同一轮复审修复）——进了这段代码往下走，就意味着 STARTED
        # 已经发出，下面的 except 分支据此才需要、也才能安全地补发 LLM_RESPONSE_FINISHED。
        stream = ensure_collected_stream(stream_llm_resilient(ctx, state, llm_request))
        try:
            async with aclosing(stream):
                async for _ in stream:
                    stream.accept_current()
        except Exception as exc:
            partial = stream.partial_result()
            if getattr(exc, "retriable", False):
                logger.warning("RecognizeIntentStep: LLM call failed for task %s: %s", target_task_id, exc)
            else:
                logger.exception("RecognizeIntentStep: LLM call failed for task %s", target_task_id)
            # 复审修复（task-6）：gateway 在进入重试循环**之前**就无条件发了
            # LLM_REQUEST_STARTED/LLM_PROMPT_SENT——任何退出路径都必须配对
            # LLM_RESPONSE_FINISHED，否则 host SSE 看到一条永远等不到收尾的挂死请求
            # （同成功路径下方那条收尾的理由）。finish_reason="error" 是本 step 自己的
            # 错误收尾语义，不对应任何 provider 原生 finish_reason 取值；content/reasoning/
            # usage 用异常发生前已累积到的部分值（可能全空）。
            #
            # 就地吞掉异常、不让它向上冒泡：这不是本次改动引入的行为，是 commit 920bb05
            # （总账 C5）已经明确裁定的既有设计——「正常路径（含内部已捕获、降级处理的
            # 失败）记 completed，只有真正逃出这段代码的未捕获异常才记 interrupted」。
            # recognize_intent 是尽力而为的元数据补全（失败不影响 task 主流程），outer
            # `launch_recognize_intent._run()` 因此仍会把这次失败报成
            # RUN_FINISHED(outcome=COMPLETED)——按该文档化的既定口径延续，未在本次改动中
            # 变更（复审记录见 task-6-report.md）。
            await ctx.event_bus.emit(make_event(
                state, EventType.LLM_RESPONSE_FINISHED,
                payload={
                    "request_id": req_id, "content": partial.text, "reasoning": partial.reasoning,
                    "tool_calls": [],
                    "usage": dataclasses.asdict(partial.usage),
                    "llm_model": model, "llm_account": llm_account,
                    "finish_reason": "error"}))
            return StepOutcome(next_step=None)

        result = stream.get_final_result()
        text, reasoning, usage = result.text, result.reasoning, result.usage
        tool_call_obj: ToolCall | None = result.tool_calls[-1] if result.tool_calls else None
        tool_name = tool_call_obj.name if tool_call_obj is not None else ""
        tool_args = tool_call_obj.arguments if tool_call_obj is not None else {}

        # LLM_RESPONSE_FINISHED：gateway 只发流式侧 4 个事件，收尾事件由各调用方自己发
        # （同 act.py / compact.py 的既定分工）——否则 LLM_REQUEST_STARTED 有始无终，host
        # SSE 会看到一条永远等不到收尾的挂死请求。payload 结构照抄 act.py::_run_llm_turn。
        await ctx.event_bus.emit(make_event(
            state, EventType.LLM_RESPONSE_FINISHED,
            payload={
                "request_id": req_id, "content": text, "reasoning": reasoning,
                "tool_calls": (
                    [{"id": tool_call_obj.id, "name": tool_call_obj.name,
                      "arguments": tool_call_obj.arguments}]
                    if tool_call_obj is not None else []
                ),
                "usage": dataclasses.asdict(usage),
                "llm_model": model, "llm_account": llm_account,
                "finish_reason": "tool_use" if tool_name else "stop"}))

        if tool_name and ctx.capability_gateway is not None:
            await ctx.capability_gateway.invoke(
                tool_name=tool_name,
                arguments=tool_args,
                state=state,
                ctx=ctx,
            )

        await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_TOOL_CALL, payload={
            "title": tool_args.get("title", ""),
            "description": tool_args.get("description", ""),
            "session_goal": tool_args.get("session_goal", ""),
        }))
        await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_COMPLETED, payload={
            "title": tool_args.get("title", ""),
            "description": tool_args.get("description", ""),
            "session_goal": tool_args.get("session_goal", ""),
            # 只透出不记账：意图识别 LLM 开销此前无账可查（spec 2026-07-16 §5.7）
            "usage": dataclasses.asdict(usage),
        }))

        return StepOutcome(next_step=None)
