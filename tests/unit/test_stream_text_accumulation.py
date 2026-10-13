"""Topic 10 only: adapter buffers -> visible deltas -> act -> saved body.

All inputs use ordinary incremental content and a normal protocol ending.
SSE compatibility, reasoning field aliases, finish reasons, cancellation,
queue overflow, and the known prefix-merge defect are outside this suite.
Expected bodies are literals, independent of production cleanup functions.
"""

import json

import httpx
import pytest

import ctx_weft.providers.llm.stream as stream_module
from ctx_weft.core.loop.steps.act import ActStep, _ingest_assistant_turn, _run_llm_turn
from ctx_weft.protocols import EventType, MemoryEventType
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.llm.openai import OpenAIAdapter
from ctx_weft.providers.llm.text_calls import ContentGate
from tests.integration.test_interactive_task import _act_state_ctx


class _RecordingBus(InProcessEventBus):
    def __init__(self):
        super().__init__()
        self.events = []

    async def emit(self, event):
        await super().emit(event)
        self.events.append(event)


def _event(delta, finish_reason=None):
    payload = {
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8")


class _ReplayAdapter(OpenAIAdapter):
    def __init__(self, pieces):
        self.pieces = pieces
        self.token_deltas = []
        self.request_count = 0
        super().__init__(
            api_key="offline-topic-10", model="offline-topic-10",
            base_url="https://topic-10.invalid", max_http_retries=1,
        )

    def _make_client(self):
        body = b"".join(_event({"content": piece}) for piece in self.pieces)
        body += _event({}, "stop") + b"data: [DONE]\n\n"

        def handle(request):
            self.request_count += 1
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=body,
            )

        return httpx.AsyncClient(transport=httpx.MockTransport(handle))

    async def complete(self, request, stream=True):
        async for chunk in super().complete(request, stream):
            if chunk.kind == "token":
                self.token_deltas.append(chunk.text)
            yield chunk


@pytest.mark.parametrize("piece_size", [1, 4, 13, 4096])
@pytest.mark.parametrize("case", ["plain", "inline-think", "text-tool"])
async def test_shared_collector_preserves_visible_body_through_memory(case, piece_size, monkeypatch):
    cases = {
        "plain": (
            "开头：你好，世界。\n第二行：保留空格  和符号 <x>。末尾。\n",
            "开头：你好，世界。\n第二行：保留空格  和符号 <x>。末尾。\n",
        ),
        "inline-think": (
            "开头：<think>先计算，然后核对。</think>答案是42。\n",
            "开头：答案是42。\n",
        ),
        "text-tool": (
            '开头：正在查询。\n<tool_call>{"name":"lookup",'
            '"arguments":{"id":"doc-1"}}</tool_call>',
            "开头：正在查询。\n",
        ),
    }
    raw, expected_body = cases[case]
    pieces = [raw[i:i + piece_size] for i in range(0, len(raw), piece_size)]
    # Keep this diagnosis independent of topic 2's known prefix-guessing defect.
    prefix = ""
    for piece in pieces:
        assert not prefix or not piece.startswith(prefix)
        prefix += piece

    gate_inputs = []

    class _TracingGate(ContentGate):
        def feed(self, content_text):
            gate_inputs.append(content_text)
            return super().feed(content_text)

    monkeypatch.setattr(stream_module, "ContentGate", _TracingGate)
    adapter = _ReplayAdapter(pieces)
    state, ctx, _task, _hitl, memory = _act_state_ctx(True, adapter)
    bus = _RecordingBus()
    ctx.event_bus = bus
    try:
        if case == "text-tool":
            # Exercise the actual turn accumulator and persistence function, but
            # do not dispatch a tool: tool execution is outside topic 10.
            prompt = state.assembled_prompt
            turn = await _run_llm_turn(state, ctx, prompt, list(prompt.messages), 1)
            assert turn.text == expected_body
            assert [(tc.name, tc.arguments) for tc in turn.tool_calls] == [
                ("lookup", {"id": "doc-1"}),
            ]
            await _ingest_assistant_turn(
                state, ctx, turn.text, turn.tool_calls, turn.usage, 1,
                anchor=turn.anchor, minted=turn.minted,
            )
        else:
            await ActStep().execute(state, ctx)

        assert adapter.request_count == 1
        assert len(gate_inputs) == len(pieces)
        assert gate_inputs[-1] == raw
        assert "".join(adapter.token_deltas) == expected_body
        emitted_body = "".join(
            event.payload["delta"] for event in bus.events
            if event.type == EventType.LLM_TOKEN_STREAMED
        )
        assert emitted_body == expected_body
        finished = [event for event in bus.events if event.type == EventType.LLM_RESPONSE_FINISHED]
        assert len(finished) == 1
        assert finished[0].payload["content"] == expected_body
        records = await memory.recall_recent(
            state.scope, [MemoryEventType.LLM_RESPONSE], 10, ctx.provider_ctx,
        )
        assert [record.content for record in records] == [expected_body]
    finally:
        await adapter._client.aclose()
