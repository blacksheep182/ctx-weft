"""Topic 10: single ownership, accepted results, and unchanged incremental events."""

# ruff: noqa: RUF001 -- Chinese punctuation is intentional test data.
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from ctx_weft.core.loop.observing import run_observe_react
from ctx_weft.core.loop.steps.act import _run_llm_turn
from ctx_weft.core.loop.steps.compact import summarize_for_compact
from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep
from ctx_weft.protocols import EventType, LLMChunk, LLMMessage, LLMRequest
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.protocols.events import EventOrigin
from ctx_weft.providers.llm.anthropic import AnthropicAdapter, _serialize_messages
from ctx_weft.providers.llm.openai import OpenAIAdapter
from ctx_weft.providers.llm.stream import CollectedLLMStream
from tests.integration.test_interactive_task import _act_state_ctx


def sse(event):
    return ("data: " + json.dumps(event, ensure_ascii=False) + "\n\n").encode()


def body_for(provider):
    if provider == "openai":
        return b"".join(
            [
                sse({"choices": [{"delta": {"reasoning_content": part}}]})
                for part in ["先核对，", "再回答。"]
            ]
            + [
                sse({"choices": [{"delta": {"content": part}}]})
                for part in ["您好，", "答案是42。"]
            ]
            + [sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}), b"data: [DONE]\n\n"]
        )
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
        *[
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": part},
            }
            for part in ["先核对，", "再回答。"]
        ],
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "signed-thinking"},
        },
        {"type": "content_block_stop", "index": 0},
        *[
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": part},
            }
            for part in ["您好，", "答案是42。"]
        ],
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]
    return b"".join(map(sse, events))


def offline_adapter(provider, body=None):
    base = OpenAIAdapter if provider == "openai" else AnthropicAdapter

    class Offline(base):
        requests = 0

        def _make_client(self):
            def respond(request):
                self.requests += 1
                return httpx.Response(200, content=body if body is not None else body_for(provider))

            return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    return Offline(api_key="offline", max_http_retries=1)


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("entry", ["act", "observe", "compact", "recognize_intent"])
async def test_all_four_entries_reuse_one_collector_and_preserve_deltas(
    provider, entry, monkeypatch
):
    owners = []
    original = CollectedLLMStream.__init__

    def record_owner(self, *args, **kwargs):
        original(self, *args, **kwargs)
        owners.append(self)

    monkeypatch.setattr(CollectedLLMStream, "__init__", record_owner)
    adapter = offline_adapter(provider)
    state, ctx, task, *_ = _act_state_ctx(True, adapter)
    events = []

    async def emit(event):
        events.append(event)

    async def assemble(request):
        return state.assembled_prompt

    ctx.event_bus = SimpleNamespace(emit=emit)
    ctx.assembler = SimpleNamespace(assemble=assemble)
    state.origin = {
        "act": EventOrigin.LOOP_ACT,
        "observe": EventOrigin.LOOP_OBSERVE,
        "compact": EventOrigin.LOOP_COMPACT,
        "recognize_intent": EventOrigin.LOOP_RECOGNIZE_INTENT,
    }[entry]
    try:
        if entry == "act":
            await _run_llm_turn(
                state, ctx, state.assembled_prompt, state.assembled_prompt.messages, 1
            )
        elif entry == "observe":
            await run_observe_react(
                state,
                ctx,
                system="",
                messages=state.assembled_prompt.messages,
                tools=[],
                max_rounds=1,
                terminal_tool_name="unused",
            )
        elif entry == "compact":
            await summarize_for_compact(state, ctx)
        else:
            task.title = ""
            state.extra["bound_capabilities"] = [
                ToolCapability(id="metadata", name="metadata", purposes=["recognize_intent"])
            ]
            await RecognizeIntentStep().execute(state, ctx)
        assert adapter.requests == 1
        assert len(owners) == 1
        result = owners[0].get_final_result()
        assert result.text == "您好，答案是42。"
        assert result.reasoning == "先核对，再回答。"
        assert owners[0].get_final_result() is result
        assert owners[0].state.content_text == ""
        assert owners[0].state._reasoning == []
        assert owners[0].state.tool_buffers == {}
        finished = [e for e in events if e.type == EventType.LLM_RESPONSE_FINISHED]
        assert len(finished) == 1
        assert finished[0].payload["content"] is result.text
        assert finished[0].payload["reasoning"] is result.reasoning
        assert [e.payload["delta"] for e in events if e.type == EventType.LLM_TOKEN_STREAMED] == [
            "您好，",
            "答案是42。",
        ]
        assert [
            e.payload["delta"] for e in events if e.type == EventType.LLM_REASONING_STREAMED
        ] == ["先核对，", "再回答。"]
        assert all("provider_state" not in e.payload for e in events)
    finally:
        await adapter._client.aclose()


@pytest.mark.parametrize("kind", ["token", "reasoning"])
async def test_read_but_unaccepted_chunk_is_excluded_after_close(kind):
    async def source():
        yield LLMChunk(kind=kind, text="已接收")
        yield LLMChunk(kind=kind, text="未接收")

    stream = CollectedLLMStream(source())
    await anext(stream)
    stream.accept_current()
    await anext(stream)
    with pytest.raises(RuntimeError, match="not completed"):
        stream.get_final_result()
    await stream.aclose()
    result = stream.partial_result()
    assert (result.text if kind == "token" else result.reasoning) == "已接收"
    assert stream.partial_result() is result
    assert stream.state.content_text == ""
    assert stream.state._reasoning == []


async def test_native_raw_text_does_not_leak_past_accepted_position():
    body = b"".join(sse({"choices": [{"delta": {"content": t}}]}) for t in ["第一段", "第二段"])
    adapter = offline_adapter("openai", body)
    stream = adapter.open_stream(LLMRequest(model="mock", system="", messages=[]))
    try:
        await anext(stream)
        stream.accept_current()
        await anext(stream)
        await stream.aclose()
        assert stream.partial_result().text == "第一段"
    finally:
        await adapter._client.aclose()


async def test_signed_thinking_is_replayed_from_the_message_after_tool_id_minting():
    adapter = offline_adapter("anthropic")
    stream = adapter.open_stream(LLMRequest(model="mock", system="", messages=[]))
    try:
        async for _ in stream:
            stream.accept_current()
        result = stream.get_final_result()
        signed = result.provider_state["anthropic"]["thinking_blocks"][0]
        assert signed["thinking"] is result.reasoning
        messages = [
            LLMMessage(
                role="assistant",
                content=result.text,
                tool_calls=[{"id": "internal-minted-id", "name": "lookup", "arguments": {}}],
                provider_state=result.provider_state,
            ),
            LLMMessage(role="tool", content="ok", tool_call_id="internal-minted-id"),
        ]
        blocks = _serialize_messages(messages)[0]["content"]
        assert blocks[0] == {
            "type": "thinking",
            "thinking": "先核对，再回答。",
            "signature": "signed-thinking",
        }
        assert blocks[-1]["id"] == "internal-minted-id"
    finally:
        await adapter._client.aclose()


@pytest.mark.parametrize("texts", [["", "first"], ["", "first", "second"]])
async def test_signed_blocks_keep_their_text_and_order_including_an_empty_block(texts):
    async def source(owner):
        state = owner.state
        for index, text in enumerate(texts):
            state.start_thinking(index)
            if text:
                yield LLMChunk(kind="reasoning", text=text, reasoning_index=index)
            state.signature(index, f"signature-{index}")
            state.stop_block(index)

    stream = CollectedLLMStream(source)
    async for _ in stream:
        stream.accept_current()
    result = stream.get_final_result()
    blocks = result.provider_state["anthropic"]["thinking_blocks"]
    assert result.reasoning == "".join(texts)
    assert [(b["thinking"], b["signature"]) for b in blocks] == [
        (text, f"signature-{index}") for index, text in enumerate(texts)
    ]


async def test_a_stream_rejects_concurrent_readers():
    started, release = asyncio.Event(), asyncio.Event()

    async def source():
        started.set()
        await release.wait()
        yield LLMChunk(kind="token", text="one")

    stream = CollectedLLMStream(source())
    task = asyncio.create_task(anext(stream))
    await started.wait()
    try:
        with pytest.raises(RuntimeError, match="one reader"):
            await anext(stream)
    finally:
        release.set()
        await task
        await stream.aclose()


async def test_close_stops_a_pending_read_and_keeps_the_accepted_result():
    started, closed = asyncio.Event(), asyncio.Event()

    async def source():
        try:
            yield LLMChunk(kind="token", text="accepted")
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    stream = CollectedLLMStream(source())
    await anext(stream)
    stream.accept_current()
    read = asyncio.create_task(anext(stream))
    await started.wait()
    await stream.aclose()
    assert read.cancelled()
    assert closed.is_set()
    assert stream.partial_result().text == "accepted"


def test_legalization_keeps_signed_thinking_on_the_corresponding_assistant():
    from ctx_weft.core.loop.llm_gateway import legalize_messages

    provider_state = {
        "anthropic": {
            "thinking_blocks": [
                {"type": "thinking", "thinking": "reasoning", "signature": "signature"}
            ]
        }
    }
    messages = [
        LLMMessage(role="user", content="go"),
        LLMMessage(role="assistant", content="preamble"),
        LLMMessage(
            role="assistant",
            content="answer",
            tool_calls=[{"id": "internal-id", "name": "lookup", "arguments": {}}],
            provider_state=provider_state,
        ),
        LLMMessage(role="tool", content="result", tool_call_id="internal-id"),
    ]
    assistant = legalize_messages(messages)[1]
    assert assistant.provider_state is provider_state
    assert _serialize_messages([assistant])[0]["content"][0]["signature"] == "signature"


async def test_inline_http_retry_discards_hidden_content_from_the_failed_attempt(monkeypatch):
    import ctx_weft.providers.llm.openai as openai_module

    calls, attempts = [], []

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse({"choices": [{"delta": {"content": "<think>failed attempt"}}]})
            raise httpx.ReadError("offline disconnect")

    class Offline(OpenAIAdapter):
        def _make_client(self):
            def respond(request):
                calls.append(request)
                if len(calls) == 1:
                    return httpx.Response(200, stream=BrokenBody())
                return httpx.Response(200, content=body_for("openai"))

            return httpx.AsyncClient(transport=httpx.MockTransport(respond))

    async def no_sleep(delay):
        pass

    monkeypatch.setattr(openai_module.asyncio, "sleep", no_sleep)
    from ctx_weft.providers.llm.stream import StreamState

    original = StreamState.feed_text

    def observe_attempt(self, text):
        attempts.append(self)
        return original(self, text)

    monkeypatch.setattr(StreamState, "feed_text", observe_attempt)
    adapter = Offline(api_key="offline", max_http_retries=2)
    stream = adapter.open_stream(LLMRequest(model="offline", system="", messages=[]))
    try:
        async for _ in stream:
            stream.accept_current()
        result = stream.get_final_result()
        assert len(calls) == 2
        assert attempts[0] is not attempts[-1]
        assert result.text == "您好，答案是42。"
        assert result.reasoning == "先核对，再回答。"
    finally:
        await adapter._client.aclose()
