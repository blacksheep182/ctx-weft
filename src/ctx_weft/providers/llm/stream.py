"""Per-request collection shared by adapters and every model-stream consumer.

Native adapters keep raw text here and only emit its visible deltas. Legacy
clients supply already-visible chunks. Neither gateways nor business callers
accumulate a second copy. Accepted positions are lengths, never text snapshots.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from ctx_weft.protocols.llm import LLMChunk, LLMStream, LLMStreamResult, LLMUsage, ToolCall
from ctx_weft.providers.llm.text_calls import ContentGate, clean_visible, merge_content


class ReasoningCandidate:
    """Existing non-delta compatibility rule, owned by the request state."""

    def __init__(self) -> None:
        self.held = ""
        self.delta_seen = False

    def observe(self, text: str) -> None:
        if text and not self.delta_seen:
            self.held = text if self.held and text.startswith(self.held) else self.held + text

    def take(self) -> str:
        text, self.held = self.held, ""
        return text


@dataclass(frozen=True)
class _Position:
    text: int = 0
    reasoning: int = 0
    tools: int = 0
    usage: LLMUsage | None = None
    finish_reason: str | None = None


class StreamState:
    """The only mutable full-content owner for one underlying request attempt."""

    def __init__(self, *, raw_text: bool = False) -> None:
        self.raw_text = raw_text
        self.content_text = ""
        self.gate = ContentGate()
        self.tool_buffers: dict[int, dict[str, Any]] = {}
        self.reasoning_candidate = ReasoningCandidate()
        self._reasoning: list[tuple[int | None, str]] = []
        self._thinking: dict[int, dict[str, Any]] = {}
        self._tools: list[ToolCall] = []
        self._delivered = _Position()
        self._accepted = _Position()
        self._final: LLMStreamResult | None = None
        self._closed_result: LLMStreamResult | None = None

    def feed_text(self, text: str) -> str:
        self.content_text = merge_content(self.content_text, text)
        return self.gate.feed(self.content_text)

    def feed_tool(
        self,
        index: int,
        *,
        call_id: str = "",
        name: str = "",
        arguments: str = "",
        append_name: bool = False,
        reset: bool = False,
    ) -> None:
        if reset:
            self.tool_buffers[index] = {"id": "", "name": "", "arguments": ""}
        buf = self.tool_buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if call_id:
            buf["id"] = call_id
        if name:
            buf["name"] = buf["name"] + name if append_name else name
        if arguments:
            buf["arguments"] = merge_content(buf["arguments"], arguments)

    def start_thinking(self, index: int) -> None:
        self._thinking[index] = {"signature": "", "closed": False}

    def signature(self, index: int, signature: str) -> None:
        if index in self._thinking:
            self._thinking[index]["signature"] = signature

    def stop_block(self, index: int) -> None:
        if index in self._thinking:
            self._thinking[index]["closed"] = True

    def record(self, chunk: LLMChunk) -> None:
        p = self._delivered
        text, reasoning, tools, usage, finish = (
            p.text,
            p.reasoning,
            p.tools,
            p.usage,
            p.finish_reason,
        )
        if chunk.kind == "token":
            if not self.raw_text:
                self.content_text += chunk.text
            text += len(chunk.text)
        elif chunk.kind == "reasoning":
            index = getattr(chunk, "reasoning_index", None)
            if self._reasoning and self._reasoning[-1][0] == index:
                self._reasoning[-1] = (index, self._reasoning[-1][1] + chunk.text)
            else:
                self._reasoning.append((index, chunk.text))
            reasoning += len(chunk.text)
        elif chunk.kind == "tool_call" and chunk.tool_call is not None:
            self._tools.append(chunk.tool_call)
            tools += 1
        elif chunk.kind == "usage" and chunk.usage is not None:
            usage = chunk.usage
        if getattr(chunk, "finish_reason", None) is not None:
            finish = chunk.finish_reason
        self._delivered = _Position(text, reasoning, tools, usage, finish)

    def accept_current(self) -> None:
        self._accepted = self._delivered

    def _result(self, position: _Position) -> LLMStreamResult:
        visible = clean_visible(self.content_text) if self.raw_text else self.content_text
        remaining = position.reasoning
        parts: list[str] = []
        block_parts: dict[int, list[str]] = {}
        incomplete: set[int] = set()
        for index, text in self._reasoning:
            take = text[:remaining]
            remaining -= len(take)
            parts.append(take)
            if index is not None:
                block_parts.setdefault(index, []).append(take)
                if len(take) != len(text):
                    incomplete.add(index)
        reasoning_text = "".join(parts)
        single_block = len(block_parts) == 1 and all(
            index is not None for index, _ in self._reasoning
        )
        blocks = [
            {
                "type": "thinking",
                "thinking": reasoning_text
                if single_block and index in block_parts
                else "".join(block_parts.get(index, ())),
                "signature": info["signature"],
            }
            for index, info in self._thinking.items()
            if info["closed"] and info["signature"] and index not in incomplete
        ]
        provider_state = (
            {
                "anthropic": {
                    "thinking_blocks": blocks,
                    "tool_call_ids": tuple(tc.id for tc in self._tools[: position.tools]),
                }
            }
            if blocks
            else {}
        )
        return LLMStreamResult(
            text=visible[: position.text],
            reasoning=reasoning_text,
            tool_calls=tuple(self._tools[: position.tools]),
            usage=position.usage or LLMUsage(),
            finish_reason=position.finish_reason,
            provider_state=provider_state,
        )

    def partial_result(self) -> LLMStreamResult:
        if self._closed_result is not None:
            return self._closed_result
        if self._final is not None:
            if self._accepted == self._delivered:
                return self._final
            # A stream can finish before a business consumer accepts its last chunk.
            p, full = self._accepted, self._final
            return LLMStreamResult(
                text=full.text[: p.text],
                reasoning=full.reasoning[: p.reasoning],
                tool_calls=full.tool_calls[: p.tools],
                usage=p.usage or LLMUsage(),
                finish_reason=p.finish_reason,
            )
        return self._result(self._accepted)

    def finish(self) -> None:
        if self._final is None:
            self._final = self._result(self._delivered)
            self._release_buffers()

    def close(self) -> None:
        if self._final is None and self._closed_result is None:
            self._closed_result = self._result(self._accepted)
            self._release_buffers()

    def _release_buffers(self) -> None:
        self.content_text = ""
        self.tool_buffers.clear()
        self._reasoning.clear()
        self._thinking.clear()
        self._tools.clear()
        self.reasoning_candidate.held = ""

    def get_final_result(self) -> LLMStreamResult:
        if self._final is None:
            raise RuntimeError("Model stream has not completed; use partial_result().")
        return self._final


_CURRENT_STATE: ContextVar[StreamState | None] = ContextVar("llm_stream_state", default=None)


def current_stream_state() -> StreamState | None:
    """Compatibility transforms can use the current request's state, never adapter state."""
    return _CURRENT_STATE.get()


class CollectedLLMStream:
    def __init__(
        self,
        source: AsyncIterator[LLMChunk] | Callable[[CollectedLLMStream], AsyncIterator[LLMChunk]],
        *,
        raw_text: bool = False,
    ) -> None:
        self.state = StreamState(raw_text=raw_text)
        self._source = source(self) if callable(source) else source
        self._finished = False
        self._closed = False
        self._reading = False
        self._read_task: asyncio.Task[Any] | None = None

    def new_attempt(self) -> StreamState:
        self.state = StreamState(raw_text=self.state.raw_text)
        _CURRENT_STATE.set(self.state)
        return self.state

    def __aiter__(self) -> CollectedLLMStream:
        return self

    async def __anext__(self) -> LLMChunk:
        if self._closed or self._finished:
            raise StopAsyncIteration
        if self._reading:
            raise RuntimeError("A model stream supports only one reader.")
        self._reading = True
        self._read_task = asyncio.current_task()
        token = _CURRENT_STATE.set(self.state)
        try:
            chunk = await self._source.__anext__()
            self.state.record(chunk)
            return chunk
        except StopAsyncIteration:
            self._finished = True
            self.state.finish()
            raise
        finally:
            _CURRENT_STATE.reset(token)
            self._reading = False
            self._read_task = None

    def accept_current(self) -> None:
        self.state.accept_current()

    def partial_result(self) -> LLMStreamResult:
        return self.state.partial_result()

    def get_final_result(self) -> LLMStreamResult:
        return self.state.get_final_result()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                reader = self._read_task
                if reader is not None and reader is not asyncio.current_task():
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                close = getattr(self._source, "aclose", None)
                if close is not None:
                    await close()
            finally:
                self.state.close()

    async def __aenter__(self) -> CollectedLLMStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


def ensure_collected_stream(source: AsyncIterator[LLMChunk]) -> LLMStream:
    """Reuse native stream ownership; collect only complete-only legacy clients."""
    return source if isinstance(source, LLMStream) else CollectedLLMStream(source)


class DelegatingLLMStream:
    """Gateway lifecycle/retry wrapper. It owns no body or reasoning buffers."""

    def __init__(self, source: Callable[[DelegatingLLMStream], AsyncIterator[LLMChunk]]) -> None:
        self._source = source(self)
        self._inner: LLMStream | None = None
        self._finished = False
        self._closed = False
        self._read_task: asyncio.Task[Any] | None = None

    def bind(self, source: AsyncIterator[LLMChunk]) -> LLMStream:
        self._inner = ensure_collected_stream(source)
        return self._inner

    def __aiter__(self) -> DelegatingLLMStream:
        return self

    async def __anext__(self) -> LLMChunk:
        if self._closed or self._finished:
            raise StopAsyncIteration
        if self._read_task is not None:
            raise RuntimeError("A model stream supports only one reader.")
        self._read_task = asyncio.current_task()
        try:
            return await self._source.__anext__()
        except StopAsyncIteration:
            self._finished = True
            raise
        finally:
            self._read_task = None

    def accept_current(self) -> None:
        if self._inner is not None:
            self._inner.accept_current()

    def partial_result(self) -> LLMStreamResult:
        return self._inner.partial_result() if self._inner is not None else LLMStreamResult()

    def get_final_result(self) -> LLMStreamResult:
        if not self._finished or self._inner is None:
            raise RuntimeError("Model stream has not completed; use partial_result().")
        return self._inner.get_final_result()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                reader = self._read_task
                if reader is not None and reader is not asyncio.current_task():
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                await self._source.aclose()
            finally:
                if self._inner is not None:
                    await self._inner.aclose()

    async def __aenter__(self) -> DelegatingLLMStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
