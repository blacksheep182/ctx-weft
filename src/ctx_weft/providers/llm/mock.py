"""MockLLMAdapter：测试用 LLM 模拟器。

按预定义的 response 序列依次返回；用于 Phase 1 集成测试与单测。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ctx_weft.core.utils.content import content_to_text
from ctx_weft.protocols import LLMChunk, LLMClient, LLMRequest, LLMUsage, ToolCall
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.llm.stream import CollectedLLMStream


@dataclass
class MockResponse:
    """一次 LLM 调用的预期返回。"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    chunk_size: int = 16  # streaming 时每个 chunk 的字符数
    # usage 拆分模拟（事件层/联调测试用；cache 之和应 ≤ 估算的 prompt_tokens，测试自行保证）
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


#: 判定工具的对外名（`qualify("control:report_task_outcome")`）。
OBSERVER_TOOL = "control__report_task_outcome"


def is_observer_request(request) -> bool:
    """这次调用是一轮 observer 吗——按能力面里有没有那个判定工具认。"""
    names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
    return OBSERVER_TOOL in names


def observer_success_with(
    verdict: str, *, tool_call_id: str = "tc_observer",
    recap: str = "the actor did the work", summary: str = "done",
) -> MockResponse:
    """一轮判 ``verdict`` 的 observer 回合。"""
    return MockResponse(text="", tool_calls=[ToolCall(
        id=tool_call_id, name=OBSERVER_TOOL,
        arguments={"task_status": verdict, "act_recap": recap,
                   "task_summary": summary, "next_step_hint": ""},
    )])


def observer_success(
    *, tool_call_id: str = "tc_observer", recap: str = "the actor did the work",
    summary: str = "done", hint: str = "",
) -> MockResponse:
    """一轮判 success 的 observer 回合。测试里给后台 observe 用。

    2026-09-27 起「让位」的两个边界（`plain_text` / `finish_park`）都要 verdict，而
    **verdict 缺失 ≡ retry**——拿不到判决 task 就维持 park。于是任何想让 root task 真正
    终结的测试，都得让它的 LLM 替身能应这一轮；`finish_task` 不再自己就是终态转移。
    """
    return MockResponse(text="", tool_calls=[ToolCall(
        id=tool_call_id, name=OBSERVER_TOOL,
        arguments={"task_status": "success", "act_recap": recap,
                   "task_summary": summary, "next_step_hint": hint},
    )])


class MockLLMAdapter(LLMClient):
    """按 responses 队列依次返回；用完后 raise。

    **例外：observer 回合自带兜底**（2026-09-27）。root 的纯文本与 `finish_task` 收尾如今
    都 park + 后台判定，于是每个这样的回合都多一次 observer 调用。那是绝大多数用例根本不
    关心的一次调用，但它有两种坑人的方式：脚本用完时撞 `RuntimeError: exhausted`（被后台
    那层 `except Exception` 静默吞掉，task 永远停在 park），或者更阴——从通用响应池里吃掉
    一条**不对形**的（比如一池子 `finish_task`），于是拿不到 verdict，同样永远 park，而且
    把后面每一条的序号都错开一位。

    规则：**请求里带着判定工具，而下一条脚本不是判定回合 → 回一个空回合，不动脚本指针。**

    **刻意不替它判 success。** 没有 verdict ≡ retry（task 维持 park），那正是改造之前的
    结果——那时后台 observe 吃掉一条不对形的响应、拿不到判决。一大批用例（暂停 / TTFT /
    撤销窗口那些竞态）钉的就是这个形态，替它们判 success 会让 task 终结、气泡的 task 变终态，
    整条应答路径走去另一支。想让这一轮真正终结的用例请**显式排一条判定路由**
    （`observer_success()` 备好了），别依赖这里。

    所以本兜底只修一件事：别让后台观察**偷吃**一条 act 响应、把后面每条的序号都错开一位。

    想让 root 的回合真正终结的用例，传 ``observer_verdict="success"`` 一行开掉：兜底改成回
    一个判 success 的判定回合。显式 opt-in 而不是默认——默认必须与改造前同结果，否则一大批
    钉「保持 park」的用例会被静默改语义（这个坑踩过一轮）。
    """

    def __init__(
        self,
        responses: list[MockResponse],
        context_limit: int = 100_000,
        output_reserve: int = 4096,
        *,
        observer_verdict: str | None = None,
    ) -> None:
        #: 兜底判定回合回什么。None = 回空回合（拿不到 verdict ≡ retry，维持 park，与改造前
        #: 同结果）；给了值就回那个判决。见类 docstring。
        self._observer_verdict = observer_verdict
        self._responses = list(responses)
        self._idx = 0
        self._context_limit = context_limit
        self._output_reserve = output_reserve
        # 记录最近一次调用的 request（供测试断言）
        self.last_request: LLMRequest | None = None
        #: 最近一次**actor** 回合的 request（`is_observer_request` 为假的那些）。
        #:
        #: 2026-09-28 加：`observe._should_use_llm` 删掉 root 降级后，root task 的收尾也要过
        #: 前台 observe，于是「一次 run 的最后一个 request」不再是 actor 的那个——断言 act
        #: prompt 结构的用例拿 `last_request` 会读到观察者回合（而且脚本通常没为观察者准备
        #: 回复 → 走空回合兜底 → 被 ReAct 的催促轮又推了几条消息，面目全非）。
        #: 想断言「actor 收到了什么」就用这个。
        self.last_act_request: LLMRequest | None = None
        self._tokenizers: dict[str, HeuristicTokenizer] = {}

    def tokenizer_for(self, model: str) -> HeuristicTokenizer:
        """按 model 惰性分桶的校准 tokenizer（_FixedModelClient 经此取绑定模型那只）。"""
        if model not in self._tokenizers:
            self._tokenizers[model] = HeuristicTokenizer()
        return self._tokenizers[model]

    @property
    def tokenizer(self) -> HeuristicTokenizer:
        return self.tokenizer_for("mock")

    @property
    def context_limit(self) -> int:
        return self._context_limit

    @property
    def output_reserve(self) -> int:
        return self._output_reserve

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def complete(self, request: LLMRequest, stream: bool = True) -> CollectedLLMStream:
        return self.open_stream(request, stream)

    def open_stream(self, request: LLMRequest, stream: bool = True) -> CollectedLLMStream:
        return CollectedLLMStream(self._complete(request, stream))

    def _complete(
        self,
        request: LLMRequest,
        stream: bool = True,
    ) -> AsyncIterator[LLMChunk]:
        self.last_request = request
        if not is_observer_request(request):
            self.last_act_request = request

        # observer 回合的兜底（见类 docstring）：只在**脚本没为它准备**时接手——判据是
        # 「下一条脚本不是判定回合」。回空回合（= 拿不到 verdict ≡ retry，与改造前同结果），
        # 且不动 `_idx`：那一池子 act 回合的序号不该被后台观察挤掉一位。
        if is_observer_request(request) and not self._next_is_observer():
            if self._observer_verdict is not None:
                return self._stream(
                    observer_success_with(self._observer_verdict), request)
            return self._stream(MockResponse(text=""), request)

        if self._idx >= len(self._responses):
            raise RuntimeError(
                f"MockLLMAdapter exhausted: called {self._idx + 1} times but only "
                f"{len(self._responses)} responses configured"
            )
        response = self._responses[self._idx]
        self._idx += 1

        return self._stream(response, request)

    def _next_is_observer(self) -> bool:
        """下一条脚本是不是一轮判定回合（调了那个判定工具）。"""
        if self._idx >= len(self._responses):
            return False
        return any(tc.name == OBSERVER_TOOL
                   for tc in self._responses[self._idx].tool_calls)

    async def _stream(
        self,
        response: MockResponse,
        request: LLMRequest,
    ) -> AsyncIterator[LLMChunk]:
        # 文本分 chunk 流出
        text = response.text
        size = max(1, response.chunk_size)
        for i in range(0, len(text), size):
            yield LLMChunk(kind="token", text=text[i : i + size])

        # tool_calls 一次性返回
        for tc in response.tool_calls:
            yield LLMChunk(kind="tool_call", tool_call=tc)

        # usage
        prompt_text = request.system + "\n".join(
            content_to_text(m.content) for m in request.messages
        )
        tok = self.tokenizer_for(request.model or "mock")
        prompt_tokens = tok.count(prompt_text)
        completion_tokens = tok.count(text)
        yield LLMChunk(
            kind="usage",
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cache_read_tokens=response.cache_read_tokens,
                cache_write_tokens=response.cache_write_tokens,
                reasoning_tokens=response.reasoning_tokens,
                # input_tokens 自动派生 = prompt − read − write
            ),
        )

        yield LLMChunk(
            kind="done",
            finish_reason="tool_use" if response.tool_calls else "stop",
        )
