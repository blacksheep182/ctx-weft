"""Async Anthropic LLM adapter.

Implements LLMClient protocol using httpx for async HTTP.
Supports streaming via Anthropic SSE protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ctx_weft.protocols import (
    LLMCallError, LLMChunk, LLMClient, LLMMessage, LLMRequest, LLMTool, LLMUsage, ToolCall,
)
from ctx_weft.providers.llm._finalize import build_finalize_chunks, parse_tool_arguments
from ctx_weft.providers.llm._modality import downgrade_for_text_only
from ctx_weft.providers.llm._schema import sanitize_boolean_schemas
from ctx_weft.providers.llm.stream import CollectedLLMStream
from ctx_weft.providers.llm.text_calls import unwrap_raw_arguments
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer

logger = logging.getLogger(__name__)

_ANTHROPIC_API_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"

_NON_RETRIABLE_CODES = frozenset({400, 401, 403, 404})
_RETRIABLE_CODES     = frozenset({429, 500, 502, 503, 504})

# I2: 空白/空 tool_result 的占位内容——查证 Anthropic 对 tool_result.content 为空字符串
# / 空数组同样 400（与其对纯空/纯空白 text block 的拒绝同源），故不能像 assistant 分支
# 那样兜底成 "" 或 []，必须是非空的合法 block。
_EMPTY_TOOL_RESULT_CONTENT: list[dict[str, str]] = [{"type": "text", "text": "(empty)"}]


class AnthropicAdapter(LLMClient):
    """Async Anthropic Messages API adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-6",
        base_url: str = _ANTHROPIC_API_URL,
        context_limit: int = 200_000,
        output_reserve: int = 8192,
        timeout_sec: int = 120,
        max_http_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._context_limit = context_limit
        self._output_reserve = output_reserve
        self._timeout = timeout_sec
        self._max_http_retries = max_http_retries
        self._client = self._make_client()
        self._tokenizers: dict[str, HeuristicTokenizer] = {}

    def tokenizer_for(self, model: str) -> HeuristicTokenizer:
        """按 model 惰性分桶的校准 tokenizer（_FixedModelClient 经此取绑定模型那只）。"""
        if model not in self._tokenizers:
            self._tokenizers[model] = HeuristicTokenizer()
        return self._tokenizers[model]

    @property
    def tokenizer(self) -> HeuristicTokenizer:
        return self.tokenizer_for(self._model)

    def _make_client(self) -> httpx.AsyncClient:
        """Build the httpx client. Overridable to inject SSL verification."""
        return httpx.AsyncClient(timeout=self._timeout)

    @property
    def model(self) -> str:
        """构造时配置的模型名（duck-typed，非协议必需）：host 裸 adapter 直传时，
        runtime 执行前经 getattr 读取回填 session.llm_model——实际调用一直用它替换
        "mock"（见 _build_payload），事件账面须与之同源。"""
        return self._model

    @property
    def context_limit(self) -> int:
        return self._context_limit

    @property
    def output_reserve(self) -> int:
        return self._output_reserve

    @property
    def supports_tool_calling(self) -> bool:
        return True

    def complete(self, request: LLMRequest, stream: bool = True) -> AsyncIterator[LLMChunk]:
        return self.open_stream(request, stream)

    def open_stream(self, request: LLMRequest, stream: bool = True) -> CollectedLLMStream:
        return CollectedLLMStream(lambda owner: self._stream(request, owner), raw_text=True)

    async def _stream(self, request: LLMRequest, owner: CollectedLLMStream) -> AsyncIterator[LLMChunk]:
        payload = self._build_payload(request)
        headers = self._headers()
        url = f"{self._base_url}/v1/messages"

        produced = False  # 是否已向消费者吐过 chunk（流已开始）
        for attempt in range(self._max_http_retries):
            state = owner.new_attempt()
            input_tokens: int | None = None
            cache_read = 0
            cache_write = 0
            finish_reason: str | None = None
            usage: LLMUsage | None = None

            try:
                # 流式 read 超时 = 闲置(自上一字节起)超时,而非整段时长上限:正常流(token/思考/
                # 工具参数分片,以及 Anthropic 周期性 ping)持续来字节会不断重置计时器,故长工具调用/
                # 长思考不会误触发。仅当连接静默(中途断网/死 socket,无字节无 RST)超过 timeout 才触发
                # ReadTimeout(TransportError)→ produced=True 时抛 outage → INTERRUPTED;否则会永久挂起。
                async with self._client.stream(
                    "POST", url, headers=headers, json=payload,
                    timeout=httpx.Timeout(self._timeout, read=self._timeout),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        err_body = body.decode()

                        if resp.status_code in _NON_RETRIABLE_CODES:
                            raise LLMCallError(
                                f"Anthropic API error {resp.status_code}: {err_body}",
                                status_code=resp.status_code,
                                retriable=False,
                            )

                        if resp.status_code in _RETRIABLE_CODES and attempt < self._max_http_retries - 1:
                            try:
                                delay = float(resp.headers.get("retry-after", 2 ** attempt))
                            except (ValueError, TypeError):
                                delay = float(2 ** attempt)
                            delay = min(delay, 60.0)
                            logger.warning(
                                "Anthropic API %d (attempt %d/%d), retrying in %.1fs: %.200s",
                                resp.status_code, attempt + 1, self._max_http_retries, delay, err_body,
                            )
                            await asyncio.sleep(delay)
                            continue

                        retry_after_hdr = resp.headers.get("retry-after")
                        try:
                            retry_after_val = float(retry_after_hdr) if retry_after_hdr else None
                        except (ValueError, TypeError):
                            retry_after_val = None
                        is_retriable = resp.status_code in _RETRIABLE_CODES
                        raise LLMCallError(
                            f"Anthropic API error {resp.status_code}: {err_body}",
                            status_code=resp.status_code,
                            retriable=is_retriable,
                            outage=is_retriable,
                            retry_after_sec=retry_after_val,
                        )

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        event_type = event.get("type", "")

                        if event_type == "message_start":
                            usage_data = (event.get("message") or {}).get("usage") or {}
                            input_tokens = usage_data.get("input_tokens")
                            cache_read = usage_data.get("cache_read_input_tokens") or 0
                            cache_write = usage_data.get("cache_creation_input_tokens") or 0

                        elif event_type == "content_block_start":
                            block = event.get("content_block") or {}
                            idx = event.get("index", 0)
                            block_type = block.get("type", "")
                            if block_type == "tool_use":
                                state.feed_tool(idx, call_id=block.get("id", ""), name=block.get("name", ""), reset=True)
                            elif block_type == "thinking":
                                state.start_thinking(idx)

                        elif event_type == "content_block_delta":
                            delta = event.get("delta") or {}
                            delta_type = delta.get("type", "")
                            idx = event.get("index", 0)

                            if delta_type == "text_delta":
                                text = delta.get("text") or ""
                                if text:
                                    tok = state.feed_text(text)
                                    if tok:
                                        produced = True
                                        yield LLMChunk(kind="token", text=tok)

                            elif delta_type == "thinking_delta":
                                thinking = delta.get("thinking") or ""
                                if thinking:
                                    produced = True
                                    yield LLMChunk(kind="reasoning", text=thinking, reasoning_index=idx)

                            elif delta_type == "signature_delta":
                                state.signature(idx, delta.get("signature") or "")

                            elif delta_type == "input_json_delta":
                                partial = delta.get("partial_json") or ""
                                if idx in state.tool_buffers:
                                    # merge_content：兼容增量与「字符级前缀重发」两种流式语义，
                                    # 避免重发被盲拼成畸形（同 content 的 N5 防御）。
                                    state.feed_tool(idx, arguments=partial)
                                    # 见 openai.py：工具调用参数流式期间发无负载心跳，让 act 流式
                                    # 循环顶部的暂停/取消检查点有机会运行；并标记 produced 使断流走
                                    # retriable（任务层干净整跑），而非静默 inline 重发整个长工具调用。
                                    produced = True
                                    yield LLMChunk(kind="tool_call_partial")

                        elif event_type == "content_block_stop":
                            state.stop_block(event.get("index", 0))

                        elif event_type == "message_delta":
                            stop_reason = event.get("delta", {}).get("stop_reason")
                            usage_data = event.get("usage") or {}
                            output_tokens = usage_data.get("output_tokens", 0)
                            # 部分代理在尾包重发输入侧字段——带了就覆盖（以尾包为准）
                            if usage_data.get("input_tokens") is not None:
                                input_tokens = usage_data.get("input_tokens")
                            if usage_data.get("cache_read_input_tokens") is not None:
                                cache_read = usage_data.get("cache_read_input_tokens") or 0
                            if usage_data.get("cache_creation_input_tokens") is not None:
                                cache_write = usage_data.get("cache_creation_input_tokens") or 0
                            # Anthropic 的 input_tokens 不含缓存部分 → 归一为「全部输入」口径，
                            # 保证 core 的 context 阈值/预算拿到真实上下文规模（开缓存后不失真）
                            uncached = input_tokens or 0
                            prompt_total = uncached + cache_read + cache_write
                            usage = LLMUsage(
                                prompt_tokens=prompt_total,
                                completion_tokens=output_tokens,
                                total_tokens=prompt_total + output_tokens,
                                cache_read_tokens=cache_read,
                                cache_write_tokens=cache_write,
                                input_tokens=uncached,
                                # reasoning_tokens 恒 0：thinking 计入 output_tokens 无单列，
                                # 不用流式 thinking 文本估算——估算值混进计费口径就是错账
                            )
                            finish_reason = stop_reason or "stop"
                            break  # 终止事件 → 跳出循环做统一收尾

                    # 流以任何方式结束（message_delta / 自然结束）→ 统一收尾：
                    # 截断判定、native 规整、文本/思考还原、usage、done（见 _finalize）。
                    for ch in build_finalize_chunks(
                        content_text=state.content_text,
                        native_tool_calls=_parse_tool_blocks(state.tool_buffers),
                        had_native_buffer=bool(state.tool_buffers),
                        saw_terminal=finish_reason is not None,
                        usage=usage,
                        finish_reason=finish_reason,
                        emitted_visible_len=state.gate.emitted_len,
                    ):
                        yield ch
                    return

            except httpx.TransportError as exc:
                # 已吐过 chunk（流中断）→ 不在 adapter 内重试（会重复 token），
                # 抛 retriable 让 TaskManager 整任务干净重跑。
                if produced:
                    logger.warning(
                        "Anthropic stream interrupted mid-flight (tool_call_in_progress=%s): %s",
                        bool(state.tool_buffers), exc,
                    )
                    raise LLMCallError(
                        f"LLM stream interrupted mid-flight: {exc}", retriable=True, outage=True,
                    ) from exc
                if attempt < self._max_http_retries - 1:
                    delay = float(2 ** attempt)
                    logger.warning(
                        "Anthropic transport error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._max_http_retries, delay, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise LLMCallError(str(exc), retriable=False) from exc

    async def list_models(self) -> list[str]:
        """List available model ids via GET {base}/v1/models. Raises on HTTP error."""
        resp = await self._client.get(f"{self._base_url}/v1/models", headers=self._headers())
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [m["id"] for m in data if isinstance(m, dict) and "id" in m]

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _prepare_messages(self, request: LLMRequest) -> list[LLMMessage]:
        """出网前的模态处置——**本类是纯文本 adapter**，图片降级成文本占位并告警。

        能力由「注册了哪个 adapter 类」表达（spec 2026-08-28 §2「类型即声明」）：
        要发图请用 ``AnthropicMultimodalAdapter``，它覆盖本方法为原样透传。

        接缝开在这里而不是改 ``_serialize_messages``：后者已经能正确处理图片 part
        （``_parts_to_blocks`` 都在），两种 adapter 的唯一区别是**图片能不能活着走到
        它面前**。故 ``_serialize_messages`` 一字不改。

        纯文本请求返回同一对象、不记日志（无图会话逐字节不受影响）。
        """
        return downgrade_for_text_only(
            request, adapter_hint="AnthropicMultimodalAdapter")

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model if request.model and request.model != "mock" else self._model
        messages = _serialize_messages(self._prepare_messages(request))
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            # 必填字段：网关一般已算好 request.max_tokens；缺时兜底用 output_reserve（预期输出量）。
            "max_tokens": request.max_tokens or self._output_reserve,
        }
        if request.system:
            payload["system"] = request.system
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = _map_tools(request.tools)
        return payload


class AnthropicMultimodalAdapter(AnthropicAdapter):
    """支持图片输入的 Anthropic adapter。

    与基类的唯一区别：``_prepare_messages`` 原样透传，图片得以走到
    ``_serialize_messages`` 面前被拼成 Anthropic image block。

    能力由类型表达（spec 2026-08-28 §2）——host 注册哪个类，就是在声明该账号下的
    模型收不收图。core 对此零判断。
    """

    def _prepare_messages(self, request: LLMRequest) -> list[LLMMessage]:
        return request.messages


def _parse_tool_blocks(blocks: dict[int, dict[str, Any]]) -> list[ToolCall]:
    """把累积的 tool_use 块解析成规整 ToolCall：丢弃 id/name 为空者；
    arguments 解析失败先试畸形救援，仍不行才保留 ``{"_raw": ...}`` 交 gateway 报错（不静默丢）。"""
    calls: list[ToolCall] = []
    for tb in blocks.values():
        if not tb["id"] or not tb["name"]:
            continue
        args = parse_tool_arguments(tb["arguments"])
        calls.append(ToolCall(id=tb["id"], name=tb["name"], arguments=args))
    return calls


def _serialize_messages(messages: list[LLMMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "system":
            i += 1
            continue
        if m.role == "assistant":
            # Signed thinking belongs to this assistant message, not the shared adapter.
            thinking = m.provider_state.get("anthropic", {}).get("thinking_blocks", [])
            content_blocks: list[dict] = [dict(block) for block in thinking] if m.tool_calls else []
            if isinstance(m.content, str):
                if m.content:
                    content_blocks.append({"type": "text", "text": m.content})
            else:
                content_blocks.extend(_parts_to_blocks(m.content))
            for tc in (m.tool_calls or []):
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    # 兜底 {"_raw": ...} 解包回真实参数（Anthropic 的 input 必须是对象，无法像
                    # OpenAI 那样回吐原始文本）；真畸形无法解包时保持原样，交 gateway 报错。
                    "input": unwrap_raw_arguments(tc.get("input", tc.get("arguments", {}))),
                })
            # content_blocks 为空 ⟹ 兜底的 "" 与拍扁结果一致；两条分支达成"为空"的条件不同，
            # 且不对称——str 分支：m.content 为空串才不进 if（纯空白 str 仍会产出 text block，
            # 不受此收紧影响）；parts 分支：_parts_to_blocks 会跳过空/纯空白文本 part（不是
            # 「每个 part 恰好产出一个 block」，该不变式已不成立），故纯空白 TextPart 若不伴随
            # 图片/tool_calls 时 content_blocks 才会为空。
            result.append({"role": "assistant", "content": content_blocks or ""})
            i += 1
        elif m.role == "tool":
            tool_results: list[dict] = []
            while i < len(messages) and messages[i].role == "tool":
                tm = messages[i]
                # I2: parts 形态经 _parts_to_blocks 会把纯空白/空 TextPart 滤掉（同 I1），
                # 若唯一的 part 就是空白文本，结果是 `content: []`——Anthropic 大概率因此
                # 400（已查证：tool_result.content 为空字符串同样被拒，"text content blocks
                # must be non-empty" 系列报告一致；见报告 I2 小节）。str 形态则原样透传
                # "   "，同一份"空白工具输出"两种形态因此产出不一致的 wire。这里让两条路径
                # 收敛到同一个占位文本块，而不是空字符串/空列表——占位块保证非空、可被
                # provider 接受，且不改变非空白文本的既有行为（正常文本 str/parts 均不受影响）。
                if isinstance(tm.content, str):
                    content: Any = tm.content if tm.content.strip() else _EMPTY_TOOL_RESULT_CONTENT
                else:
                    blocks = _parts_to_blocks(tm.content)
                    content = blocks if blocks else _EMPTY_TOOL_RESULT_CONTENT
                if tm.tool_call_id:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tm.tool_call_id,
                        "content": content,
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": "unknown",
                        "content": content,
                    })
                i += 1
            result.append({"role": "user", "content": tool_results})
        else:
            content = (m.content if isinstance(m.content, str)
                       else _parts_to_blocks(m.content))
            result.append({"role": m.role, "content": content})
            i += 1
    return result


def _parts_to_text(parts: Any) -> str:
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        texts: list[str] = []
        for p in parts:
            # dict 形态 part 的分支已删（Phase 3c Task E/E2）：dict 是**协议违规**，已在
            # ``LLMMessage`` / ``MemoryRecord`` / ``MemoryEvent`` 三处 ``__post_init__``
            # 边界归一成 dataclass；留着分支会让后来者以为 dict 是受支持的形态。
            if hasattr(p, "text"):
                texts.append(p.text)
            # else: 非文本 dataclass part（ImagePart 等）——跳过，语义对齐 core/utils.content_to_text
        return " ".join(texts)
    return str(parts)


def _parts_to_blocks(parts: Any) -> list[dict[str, Any]]:
    """把 ContentPart 列表转成 Anthropic wire blocks（文本 → text block，图片 → image block）。

    到达本函数时 source_type 已恒为 "base64"——**这不是数据模型的性质，而是上游的
    保证**：Phase 3b 起图片在入口被外部化成 ``blob:<sha>`` ref，由
    ``core.loop.llm_gateway.stream_llm`` 在出网前（本函数之前的最后一个 async 关口）
    调 ``rehydrate_content`` 还原回 base64（架构裁定 T0：本函数是同步的，
    ``MemoryBlobStore.get`` 是 async，没法在这里 await）。取不到图时上游已把该 part 降级成
    文本占位，故本函数无需处理 ref。**绕过 gateway 直调 adapter 的路径上此保证不成立**，
    那条路上的 ref 会被当 base64 写进 payload。"""
    blocks: list[dict[str, Any]] = []
    for p in parts:
        # dict 形态 part 的分支已删（Phase 3c Task E/E2）：dict 是**协议违规**，已在
        # ``LLMMessage.__post_init__``（本函数入参恒为 ``LLMMessage.content``）与
        # ``MemoryRecord`` / ``MemoryEvent`` 三处边界归一成 dataclass，到这里不再有 dict。
        # 万一有（绕过构造器直调本函数），落到最后一支 → ``getattr`` 取不到 → 当作空文本
        # 跳过，不会 raise：adapter 在同步出网主路径上，任何 raise 都会掀掉整个 LLM 请求。
        if getattr(p, "type", None) == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": p.media_type,
                    "data": p.data,
                },
            })
        else:
            # I1: 同上对 None 兜底；注意不能改成 `or str(p)`——那会把无 .text 属性的
            # 对象整个 repr 当文本泄漏进发给模型的文本（Phase 2 修过的同类泄漏）。
            text = getattr(p, "text", None) or ""
            if text.strip():  # 同上：纯空白块与空块同属 provider 400 的一类（spec §13）
                blocks.append({"type": "text", "text": text})
    return blocks


def _map_tools(tools: list[LLMTool]) -> list[dict[str, Any]]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": sanitize_boolean_schemas(
                t.input_schema or {"type": "object", "properties": {}}
            ),
        }
        for t in tools
    ]
