"""Async OpenAI (and compatible) LLM adapter.

Implements LLMClient protocol using httpx for async HTTP.
Compatible with OpenAI chat completions API + any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ctx_weft.protocols import (
    LLMCallError, LLMChunk, LLMMessage, LLMRequest, LLMTool, LLMUsage, RAW_ARGS_KEY, ToolCall,
    LLMClient,
)
from ctx_weft.providers.llm._finalize import build_finalize_chunks, parse_tool_arguments
from ctx_weft.providers.llm._modality import downgrade_for_text_only
from ctx_weft.providers.llm._schema import sanitize_boolean_schemas
from ctx_weft.providers.llm.stream import CollectedLLMStream
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer

logger = logging.getLogger(__name__)

_OPENAI_API_URL = "https://api.openai.com"

_NON_RETRIABLE_CODES = frozenset({400, 401, 403, 404})
_RETRIABLE_CODES     = frozenset({429, 500, 502, 503, 504})

# OpenAI 的 role="tool" 只接受纯文本，图片必须重定位到随后的 user 消息（Phase 3c Task B）。
# 这条标记把「图去哪了」告诉模型，替代改前的**静默丢弃**。
#
# **必须逐字节确定**（同用户裁定 D2 的缓存理由）：不得含 blob sha / 随机 id / 时间戳 /
# 跨调用计数器——tool 结果处在缓存前缀的中段，标记每次不同会把其后的整段前缀缓存砸掉。
# 同一段内的多张图共用同一条标记，不加序号（图按 tool 消息顺序合并进同一条 user 消息，
# 模型不需要靠序号对齐）。
# 本仓所有图片占位的清单见 `core/media/refs.py` 模块 docstring（L6 收口，裁定 R1）。
# 这一条是**单向渲染、永不回读**，故留在 wire 层不搬进 `core/media/`；
# 只有 L0.5 的占位需要被解析回来。文案为英文，与清单里其余四条一致。
_TOOL_IMAGE_NOTICE = "\n\n[image see the following message]"


class OpenAIAdapter(LLMClient):
    """Async OpenAI chat completions adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = _OPENAI_API_URL,
        context_limit: int = 128_000,
        output_reserve: int = 4096,
        timeout_sec: int = 120,
        max_http_retries: int = 3,
        tool_choice: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._context_limit = context_limit
        self._output_reserve = output_reserve
        self._timeout = timeout_sec
        self._max_http_retries = max_http_retries
        # None = 省略 tool_choice（vLLM 不开 --enable-auto-tool-choice 会拒绝带 "auto" 的请求）。
        # 显式给 "auto"/"none"/"required" 才发送。
        self._tool_choice = tool_choice
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
        url = self._chat_url()

        produced = False  # 是否已向消费者吐过 chunk（流已开始）
        for attempt in range(self._max_http_retries):
            state = owner.new_attempt()
            last_tc_idx: int | None = None  # 供缺 index 的续传 delta 归到最后活跃桶
            finish_reason: str | None = None
            usage: LLMUsage | None = None
            try:
                # 流式 read 超时 = 闲置(自上一字节起)超时,而非整段时长上限:正常流(token/思考/
                # 工具参数分片)持续来字节会不断重置计时器,故长工具调用/长思考不会误触发。仅当连接
                # 静默(中途断网/死 socket,无字节无 RST)超过 timeout 才触发 ReadTimeout(TransportError)
                # → produced=True 时抛 outage → INTERRUPTED;否则会永久挂起。
                async with self._client.stream(
                    "POST", url, headers=headers, json=payload,
                    timeout=httpx.Timeout(self._timeout, read=self._timeout),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        err_body = body.decode()

                        if resp.status_code in _NON_RETRIABLE_CODES:
                            raise LLMCallError(
                                f"OpenAI API error {resp.status_code}: {err_body}",
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
                                "OpenAI API %d (attempt %d/%d), retrying in %.1fs: %.200s",
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
                            f"OpenAI API error {resp.status_code}: {err_body}",
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

                        # usage 可能随 finish_reason 同包，也可能在其后的 choices=[] 尾包里——
                        # 任何带 usage 的 event 都更新，并读到流尾再产出（不再 finish 即 return）。
                        usage_data = event.get("usage") or {}
                        if usage_data:
                            prompt_details = usage_data.get("prompt_tokens_details") or {}
                            completion_details = usage_data.get("completion_tokens_details") or {}
                            # 缓存命中：标准 prompt_tokens_details.cached_tokens →
                            # DeepSeek 方言 prompt_cache_hit_tokens → 0（方言封死在 adapter 内）
                            cache_read = (prompt_details.get("cached_tokens")
                                          or usage_data.get("prompt_cache_hit_tokens") or 0)
                            usage = LLMUsage(
                                prompt_tokens=usage_data.get("prompt_tokens", 0),
                                completion_tokens=usage_data.get("completion_tokens", 0),
                                total_tokens=usage_data.get("total_tokens", 0),
                                cache_read_tokens=cache_read,
                                cache_write_tokens=0,  # OpenAI 系不区分/不计费缓存写入
                                # input_tokens 不传 → __post_init__ 派生 prompt − cached
                                # （OpenAI 只报 cached 子集，无未缓存原始值可记）
                                reasoning_tokens=completion_details.get("reasoning_tokens", 0) or 0,
                            )

                        choices = event.get("choices") or []
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                        reasoning = delta.get("reasoning_content") or ""
                        if reasoning:
                            produced = True
                            yield LLMChunk(kind="reasoning", text=reasoning)

                        text = delta.get("content") or ""
                        if text:
                            tok = state.feed_text(text)
                            if tok:
                                produced = True
                                yield LLMChunk(kind="token", text=tok)

                        tc_deltas = delta.get("tool_calls") or []
                        for tc_delta in tc_deltas:
                            idx = _resolve_tc_index(tc_delta, state.tool_buffers, last_tc_idx)
                            last_tc_idx = idx
                            fn = tc_delta.get("function") or {}
                            name = fn.get("name")
                            args = fn.get("arguments")
                            if args:
                                args = args if isinstance(args, str) else json.dumps(args)
                            state.feed_tool(
                                idx, call_id=tc_delta.get("id") or "",
                                name=(name if isinstance(name, str) else str(name)) if name else "",
                                arguments=args or "", append_name=True,
                            )
                        if tc_deltas:
                            # 工具调用参数流式累积期间不产出 token：消费者的 async-for 会一直挂起，
                            # act 流式循环顶部的暂停/取消检查点无从触发 → 发个无负载心跳让其有机会运行。
                            # 同时标记 produced：本轮已在生成工具调用，若此后 transport 断流，走 retriable
                            # 让任务层干净整跑，而非静默 inline 重发整个长工具调用（见 except 分支）。
                            produced = True
                            yield LLMChunk(kind="tool_call_partial")

                    # 流以任何方式结束（[DONE] / 自然结束）→ 统一收尾：截断判定、
                    # native 规整、文本/思考还原、usage、done（见 _finalize）。
                    for ch in build_finalize_chunks(
                        content_text=state.content_text,
                        native_tool_calls=_parse_buffers(state.tool_buffers),
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
                        "OpenAI stream interrupted mid-flight (tool_call_in_progress=%s): %s",
                        bool(state.tool_buffers), exc,
                    )
                    raise LLMCallError(
                        f"LLM stream interrupted mid-flight: {exc}", retriable=True, outage=True,
                    ) from exc
                if attempt < self._max_http_retries - 1:
                    delay = float(2 ** attempt)
                    logger.warning(
                        "OpenAI transport error (attempt %d/%d), retrying in %.1fs: %s",
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

    def _make_client(self) -> httpx.AsyncClient:
        """Build the httpx client. Overridable to inject SSL verification."""
        return httpx.AsyncClient(timeout=self._timeout)

    def _chat_url(self) -> str:
        """Chat-completions endpoint. Overridable for endpoint inference."""
        return f"{self._base_url}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _prepare_messages(self, request: LLMRequest) -> list[LLMMessage]:
        """出网前的模态处置——**本类是纯文本 adapter**，图片降级成文本占位并告警。

        能力由「注册了哪个 adapter 类」表达（spec 2026-08-28 §2「类型即声明」）：
        要发图请用 ``OpenAIMultimodalAdapter``，它覆盖本方法为原样透传。

        降级之后 ``_serialize_messages`` 里的 tool 图重定位（``_TOOL_IMAGE_NOTICE``）
        天然空转：已无 image part，``relocated`` 恒为空，一条 user 消息都不追加。
        故那段逻辑不需要任何改动。

        纯文本请求返回同一对象、不记日志（无图会话逐字节不受影响）。
        """
        return downgrade_for_text_only(
            request, adapter_hint="OpenAIMultimodalAdapter")

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        model = request.model if request.model and request.model != "mock" else self._model
        messages = _serialize_messages(request.system, self._prepare_messages(request))
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.temperature != 1.0:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = _map_tools(request.tools)
            if self._tool_choice is not None:
                payload["tool_choice"] = self._tool_choice
        return payload


class OpenAIMultimodalAdapter(OpenAIAdapter):
    """支持图片输入的 OpenAI adapter。

    与基类的唯一区别：``_prepare_messages`` 原样透传，图片得以走到
    ``_serialize_messages`` 面前——被拼成 ``image_url`` data URL，且 ``role="tool"``
    里的图按既有逻辑重定位到随后的 user 消息（OpenAI 的 tool 只收纯文本）。

    能力由类型表达（spec 2026-08-28 §2）——host 注册哪个类，就是在声明该账号下的
    模型收不收图。core 对此零判断。
    """

    def _prepare_messages(self, request: LLMRequest) -> list[LLMMessage]:
        return request.messages


def _resolve_tc_index(
    tc_delta: dict[str, Any], buffers: dict[int, dict[str, Any]], last_idx: int | None,
) -> int:
    """决定本条 tool_call delta 该并入哪个 index 桶（容忍上游漏发 index）。

    OpenAI 流式协议里，一个 tool call 的首 delta 带 ``index``+``id``+``name``，其后的参数
    续传 delta 只带 ``index``。部分兼容端（vLLM/国产模型）在续传 delta 甚至第二个 call 上漏发
    ``index``；旧逻辑一律 ``get("index", 0)`` 会把它们错并进桶 0，参数首尾相接成畸形。规则：

      - 显式带 ``index`` → 用它（协议正道）。
      - 缺 ``index`` 但带 ``id``：id 命中现有桶 → 复用该桶（有的端每片都回带 id）；否则是
        「新 call」→ 另开一个新桶（现有最大 index + 1）。
      - 缺 ``index`` 且不带 ``id`` → 纯参数续传 → 归最后活跃桶（``last_idx``，无则 0）。
    """
    raw_idx = tc_delta.get("index")
    if raw_idx is not None:
        return raw_idx
    tc_id = tc_delta.get("id")
    if tc_id:
        for k, buf in buffers.items():
            if buf.get("id") == tc_id:
                return k
        return max(buffers) + 1 if buffers else 0
    return last_idx if last_idx is not None else 0


def _parse_buffers(buffers: dict[int, dict[str, Any]]) -> list[ToolCall]:
    """把累积的 tool_call 缓冲解析成规整 ToolCall：丢弃 id/name 为空者；
    arguments 解析失败先试畸形救援，仍不行才保留 ``{"_raw": ...}`` 交 gateway 报错（不静默丢）。"""
    calls: list[ToolCall] = []
    for buf in buffers.values():
        if not buf["id"] or not buf["name"]:
            continue
        args = parse_tool_arguments(buf["arguments"])
        calls.append(ToolCall(id=buf["id"], name=buf["name"], arguments=args))
    return calls


def _dump_tool_arguments(args: Any) -> str:
    """把 assistant tool_call 的参数序列化回给模型（OpenAI 要求 arguments 为 JSON 字符串）。

    不变式：**发出去的 arguments 必须永远是合法 JSON**——严格 OpenAI 兼容端（vLLM 等）会对
    历史 tool_call 的 arguments 再做一次 ``json.loads``，非法 JSON 直接 400
    （``Expecting ',' delimiter``）。

    兜底 ``{"_raw": <原文>}``（真畸形、finalize 没能解包）绝不把 ``_raw`` 哨兵当参数名喂回去
    （模型会照抄 ``_raw`` 死循环），也绝不逐字回吐畸形串（严格服务端会二次解析 → 400）：
      - 原文本身是合法 JSON（罕见的流式拼接抖动救回、finalize 未解包的非 dict）→ 照发（服务端能解析）。
      - 原文畸形 → 降级成合法空对象 ``"{}"``。畸形调用仍以 tool_call 形态流经 gateway，gateway 回一条
        带原文的 error tool_result 驱动模型下轮自纠（见 capability_gateway ``_raw`` 出口）；本处只需
        保证线上 arguments 合法，畸形原文不靠这格传回。
    """
    if isinstance(args, dict) and list(args) == [RAW_ARGS_KEY] and isinstance(args[RAW_ARGS_KEY], str):
        raw = args[RAW_ARGS_KEY]
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            return "{}"
        return raw
    return json.dumps(args)


def _serialize_messages(
    system: str, messages: list[LLMMessage]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})

    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            content: Any = (m.content if isinstance(m.content, str)
                            else _parts_to_blocks(m.content))
            tc_list = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": _dump_tool_arguments(tc.get("arguments", tc.get("input", {}))),
                    },
                }
                for tc in (m.tool_calls or [])
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": content or None, "tool_calls": tc_list}
            if m.reasoning_content:
                msg["reasoning_content"] = m.reasoning_content
            result.append(msg)
            i += 1
        elif m.role == "tool":
            # 批处理**整段连续 tool 消息**（与 anthropic.py 的 tool 分支同构）：每条仍只发
            # 文本，图片攒起来，整段结束后合并成一条 user 消息追加（见 _TOOL_IMAGE_NOTICE）。
            relocated: list[Any] = []
            while i < len(messages) and messages[i].role == "tool":
                tm = messages[i]
                if isinstance(tm.content, str):
                    # 纯文本 str 形态：原样透传，wire 逐字节不变。
                    content = tm.content
                else:
                    images = [p for p in (tm.content or [])
                              if getattr(p, "type", None) == "image"]
                    text = _parts_to_text(tm.content)
                    if images:
                        relocated.extend(images)
                        # 纯图片结果（text 为空）时不拼前导空行——OpenAI 的 tool content
                        # 为空会被拒（同 anthropic.py 的 _EMPTY_TOOL_RESULT_CONTENT 理由），
                        # 标记本身就是那条非空占位。
                        content = (f"{text}{_TOOL_IMAGE_NOTICE}" if text
                                   else _TOOL_IMAGE_NOTICE.lstrip("\n"))
                    else:
                        content = text
                result.append({
                    "role": "tool",
                    "tool_call_id": tm.tool_call_id or "",
                    "content": content,
                })
                i += 1
            if relocated:
                result.append({"role": "user", "content": _parts_to_blocks(relocated)})
        else:
            content = (m.content if isinstance(m.content, str)
                       else _parts_to_blocks(m.content))
            entry: dict[str, Any] = {"role": m.role, "content": content}
            if m.role == "assistant" and m.reasoning_content:
                entry["reasoning_content"] = m.reasoning_content
            result.append(entry)
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
    """把 ContentPart 列表转成 OpenAI wire blocks（文本 → text block，图片 → image_url block）。

    到达本函数时 source_type 已恒为 "base64"——**这不是数据模型的性质，而是上游的
    保证**：Phase 3b 起图片在入口被外部化成 ``blob:<sha>`` ref，由
    ``core.loop.llm_gateway.stream_llm`` 在出网前（本函数之前的最后一个 async 关口）
    调 ``rehydrate_content`` 还原回 base64（架构裁定 T0：本函数是同步的，
    ``MemoryBlobStore.get`` 是 async，没法在这里 await）。取不到图时上游已把该 part 降级成
    文本占位，故本函数无需处理 ref。**绕过 gateway 直调 adapter 的路径上此保证不成立**，
    那条路上的 ref 会被拼进 data URL。"""
    blocks: list[dict[str, Any]] = []
    for p in parts:
        # dict 形态 part 的分支已删（Phase 3c Task E/E2）：dict 是**协议违规**，已在
        # ``LLMMessage.__post_init__``（本函数入参恒为 ``LLMMessage.content``）与
        # ``MemoryRecord`` / ``MemoryEvent`` 三处边界归一成 dataclass，到这里不再有 dict。
        # 万一有（绕过构造器直调本函数），落到最后一支 → ``getattr`` 取不到 → 当作空文本
        # 跳过，不会 raise：adapter 在同步出网主路径上，任何 raise 都会掀掉整个 LLM 请求。
        if getattr(p, "type", None) == "image":
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{p.media_type};base64,{p.data}"},
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
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": sanitize_boolean_schemas(
                    t.input_schema or {"type": "object", "properties": {}}
                ),
            },
        }
        for t in tools
    ]
