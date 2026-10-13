"""LLM 接入协议层。

定义 core 与 host 之间关于 LLM 的硬契约（第 5 个接入协议，与 Memory / Capability /
Knowledge / Template 并列）。包含两个**相互独立**的扩展点：

[Adapter 契约] 实现一个新 LLM adapter 所需的全部类型：
  - LLMClient     — 要实现的协议本身（host 实现，core 只依赖此抽象）
  - LLMRequest    — complete() 入参（读 model / system / messages / tools）
  - LLMMessage    — request.messages 元素
  - LLMTool       — request.tools 元素
  - LLMChunk      — complete() 必须 yield 的流式产物
  - ToolCall      — chunk.kind == "tool_call" 的 payload
  - LLMUsage      — chunk.kind == "usage" 的 payload
  - LLMCallError  — 错误/重试契约（TaskManager 读 .retriable 决定是否重试）
  - Tokenizer     — LLMClient.tokenizer 绑定的同步 token 计数 + 真实用量回喂协议
  多模态 adapter 还会用到 protocols.context.ContentPart（LLMMessage.content 的元素）。

[Resolver 契约] 仅"多账号 LLM provider"需要，与写单个 adapter 无关：
  - LLMClientResolver — 注册进 ProviderRegistry.register_llm_provider

零运行时依赖（仅依赖 protocols.context）。详见设计文档 §11。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from ctx_weft.protocols.context import ContentPart, normalize_content_parts


# ══════════════════════════════════════════════════════════════════════════════
# Adapter 契约：数据类型（adapter 读 LLMRequest/Message/Tool，产 LLMChunk/ToolCall/Usage）
# ══════════════════════════════════════════════════════════════════════════════


# ── Errors ────────────────────────────────────────────────────────────────────


class LLMCallError(RuntimeError):
    """LLM API 调用错误，携带 HTTP 状态码与分类标志。

    三类故障各走不同路径（retriable × outage 组合）：

    retriable=False              永久错（401 认证 / 400 参数 / 404 模型 / transport 重试耗尽）
                                 → 立即 FAILED，重试无意义。
    retriable=True, outage=True  基础设施类瞬时故障（429 限流 / 5xx / 网络 / transport 断流）
                                 → 由自愈层（stream_llm_resilient）进程内退避等待 LLM 恢复；
                                 预算耗尽则 _run_loop 走 INTERRUPTED（可 /resume）。**不**经
                                 TaskManager 重试、**不**增 failure_counter——LLM 宕机不是任务失败。
    retriable=True, outage=False 内容类可重试错（_finalize 截断 / 半截 tool call / 空响应）：
                                 LLM 在线但本轮输出退化 → 由 TaskManager 有界重试该任务（重跑一轮
                                 通常即可），不当作基础设施中断。outage 标志由 adapter 在抛错点按
                                 来源设置（只有 adapter 能区分"断流"与"内容截断"），核心层只读不猜。
    retry_after_sec              provider 给了 Retry-After 时，自愈退避优先采用。
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        retriable: bool = True,
        outage: bool = False,
        retry_after_sec: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retriable = retriable
        self.outage = outage
        self.retry_after_sec = retry_after_sec


class LLMOutageError(LLMCallError):
    """瞬时 LLM 基础设施故障，应按可恢复中断（INTERRUPTED）处理，**不是** task 失败。

    自愈层在"预算耗尽"或"已吐 chunk 后中途断流"时抛此异常；_run_loop 用
    ``except LLMOutageError`` 干净分流到 INTERRUPTED 路径。
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        retry_after_sec: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status_code=status_code,
            retriable=True,
            outage=True,
            retry_after_sec=retry_after_sec,
        )


# ── Message ───────────────────────────────────────────────────────────────────


@dataclass
class LLMMessage:
    """统一 LLM 对话消息。"""

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[ContentPart]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_content: str | None = None  # 部分模型支持的"内部推理"内容
    # Request-local continuation data (e.g. signed Anthropic thinking blocks).
    # Not part of front-end events or persistent conversation storage.
    provider_state: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """把 dict 形态的 content 归一回 ContentPart dataclass（多模态 Phase 3c Task E2）。

        Task E 只在 ``MemoryRecord`` 归一，断言「dict 在 core 内部结构性消失」——该断言
        **只对经 memory 的路径成立**：本类是普通 dataclass，宿主/自定义流程可直接构造；
        ``rehydrate_content`` 又刻意是「dict 进 dict 出」。于是 Task E 删掉两家 adapter 的
        dict 分支后，dict 图片在出网路径上被**静默丢弃**（实测
        ``_parts_to_blocks([{...image...}]) → []``，无 raise 无 log）——以前是错误地兜底，
        现在是无声地丢，诊断上更糟。

        **不改成「adapter 里 raise」**：adapter 在同步出网主路径上，抛异常会掀掉整个 LLM
        请求（同 Phase 3b 对 ``MemoryBlobStore.get`` 恒不抛的取向）。归一是正解，且落在类型
        自己的边界——``dataclasses.replace``（gateway rehydrate 后即用）会重跑本方法，
        故经 gateway 的路径也一并覆盖。

        归一实现是三处边界共用的 ``core.content.normalize_content_parts``——spec §3① 要求
        形态转换收在归一层。经 ``protocols.context.normalize_content_parts`` 这个**惰性绑定**
        转调：protocols 是比 core 低的层，模块级导入 core 会把依赖反向；而把
        ``from ... import`` 写进 ``__post_init__`` 则要为每次构造付 361 ns 的 import 开销
        （见该绑定的 docstring 实测）。

        性能：本方法在出网热路径上无条件被调。``str``（绝大多数消息）与已合规的 dataclass
        列表都返回**同一对象**、不重建——纯文本路径逐字节不变。
        """
        self.content = normalize_content_parts(self.content)


# ── Tool ──────────────────────────────────────────────────────────────────────


@dataclass
class LLMTool:
    """传给 LLM 的工具定义。"""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema

    def to_prompt_text(self) -> str:
        """渲染为 system prompt 可读文本（miniAgents 风格）。"""
        return f"{self.name}: {self.description}"


# ── Tool call（LLM 返回）───────────────────────────────────────────────────────


# adapter 在「native 工具参数没解析成 JSON」时的兜底哨兵：把原始未解析文本原样塞进此键
# （见 providers/llm 的 _parse_buffers/_parse_tool_blocks），不静默丢、交 gateway 报错。
# finalize 会尝试解包（unwrap_raw_arguments）；gateway 见到残留的它则给直白报错。
# 注意别把这个 key 回灌进模型上下文——否则模型会误当参数名照抄，陷入 doom loop。
RAW_ARGS_KEY = "_raw"


@dataclass
class ToolCall:
    """LLM 返回的一次工具调用。"""

    id: str  # LLM 分配的 call id（OpenAI/Anthropic 都有）
    name: str
    arguments: dict[str, Any]


# ── Usage ─────────────────────────────────────────────────────────────────────


@dataclass
class LLMUsage:
    """token 使用统计。

    口径（跨 provider 归一，由各 adapter 负责翻译）：
      输入侧：prompt_tokens      — 本次请求的全部输入（含缓存读/写部分）
             cache_read_tokens  — 输入中命中缓存的部分
             cache_write_tokens — 输入中本次写入缓存的部分
                                  （Anthropic cache_creation；OpenAI 系恒 0）
             input_tokens       — 实际未缓存输入（全价计费部分）。
                                  ⚠ 与 prompt_tokens 的区分：prompt 是「总输入」，
                                  input 是「实际输入」；命名对齐 Anthropic API 的
                                  input_tokens（其原生口径即未缓存部分）。
      输出侧：completion_tokens  — 全部输出
             reasoning_tokens   — 输出中属于推理/thinking 的子集
                                  （OpenAI/DeepSeek 单列；Anthropic 无单列恒 0）
    不变式：
      prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
      reasoning_tokens ≤ completion_tokens
      total_tokens = prompt_tokens + completion_tokens
    input_tokens 未显式给出（哨兵 -1）时在 __post_init__ 按不变式自动派生
    （异常账钳 0），保证任何构造写法下账目自洽；显式传入的值原样保留、不钳制。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_tokens: int = -1  # 实际输入；未显式给出时自动派生
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            self.input_tokens = max(
                0,
                self.prompt_tokens - self.cache_read_tokens - self.cache_write_tokens,
            )


# ── Request / 流式 chunk ──────────────────────────────────────────────────────


@dataclass
class LLMRequest:
    """统一 LLM 请求。"""

    model: str
    system: str
    messages: list[LLMMessage]
    tools: list[LLMTool] = field(default_factory=list)
    max_tokens: int | None = None
    # caller 估算的本请求真实 prompt token（used）：真实基线 + 本轮增量，由
    # gateway.request_prompt_estimate 算好、网关据此实时算 max_tokens。瞬态字段，
    # 各 adapter 显式挑字段拼 payload，不入线上请求体。
    prompt_token_estimate: int | None = None
    temperature: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)  # trace_id / run_id / 计费标签


@dataclass
class LLMChunk:
    """流式 chunk。adapter 的 complete() 始终 yield 此类型（stream=False 时也聚合成序列）。"""

    kind: Literal["token", "reasoning", "tool_call", "tool_call_partial", "usage", "done"]
    text: str = ""
    tool_call: ToolCall | None = None
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    reasoning_index: int | None = None  # optional provider content-block identity


@dataclass(frozen=True)
class LLMStreamResult:
    """A completed/accepted snapshot. Callers treat its values as read-only."""

    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage = field(default_factory=LLMUsage)
    finish_reason: str | None = None
    provider_state: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMStream(Protocol):
    """One reader, one per-request result owner; result access never reads the network."""

    def __aiter__(self) -> LLMStream: ...
    async def __anext__(self) -> LLMChunk: ...
    def accept_current(self) -> None: ...
    def partial_result(self) -> LLMStreamResult: ...
    def get_final_result(self) -> LLMStreamResult: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class StreamingLLMClient(Protocol):
    """Optional extension; legacy complete-only clients remain supported."""

    def open_stream(self, request: LLMRequest, stream: bool = True) -> LLMStream: ...


# ══════════════════════════════════════════════════════════════════════════════
# Adapter 契约：协议本身
# ══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class Tokenizer(Protocol):
    """同步、纯本地的 token 计数 + 真实用量回喂。禁止网络调用。

    count 返回**已校准**估算（内部校准结构对 core 不可见）；observe 由循环在真实
    usage 到达后回喂 (估算段, 真实段)，实现据此自校准（如伺服 EMA）。
    """

    def count(self, text: str) -> int: ...

    def observe(self, estimated: int, actual: int) -> None: ...


@runtime_checkable
class LLMClient(Protocol):
    """LLM 调用统一门面。adapter 实现这个协议；core 只依赖此抽象。"""

    @property
    @abstractmethod
    def context_limit(self) -> int:
        """LLM 的硬上下文上限。"""
        ...

    @property
    @abstractmethod
    def output_reserve(self) -> int:
        """输入侧为输出预留的 token 量 → 喂 session.reserved_output_tokens
        （→ effective_limit → 装配预算 / compact 触发 / 限额停机）。**不**参与 per-request
        的输出上限（那是 output_ceiling）。经 ModelConfig 配置，未配时按窗口尺寸取默认
        （core.estimate.default_output_reserve）。"""
        ...

    # 可选（duck-typed，非协议必需）：output_ceiling -> int | None
    #   单次输出的收紧上限（per-request max_tokens 的硬天花板）。网关经 getattr 读取，
    #   缺省/None → 回退 context_limit。实现方（_FixedModelClient）可提供；未提供者网关自动回退。

    # ⚠️ **core 对模态能力零判断**（spec 2026-08-28-multimodal-adapter-dispatch）。
    #   core 会把 `list[ContentPart]`（含 ImagePart）原样透传到 complete()，实现方
    #   自行决定发多模态、降级成纯文本、还是报错。曾经有一个 duck-typed 的
    #   `supports_vision` 被 core 在入口读取并据以拒绝内容——它不在本协议上，host
    #   自写的 adapter 几乎必然读不到，于是一律被误判为无视觉能力。已删除。
    #   内置实现的做法见 providers/llm/{anthropic,openai}.py 的 _prepare_messages。

    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """是否支持 tool calling。"""
        ...

    @abstractmethod
    def complete(
        self,
        request: LLMRequest,
        stream: bool = True,
    ) -> AsyncIterator[LLMChunk]:
        """流式 LLM 调用——返回 chunk iterator。

        当 stream=False 时，adapter 内部也应聚合为单个 chunk 序列。
        """
        ...

    @property
    @abstractmethod
    def tokenizer(self) -> "Tokenizer":
        """该 client 绑定模型的 tokenizer。count 已含校准；observe 回喂真实用量。"""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Resolver 契约：仅多账号 LLM provider 实现（与写单个 adapter 无关）
# ══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class LLMClientResolver(Protocol):
    """Protocol: resolve an LLMClient by account name + model name.

    Registered into ProviderRegistry as the multi-account LLM provider.
    """

    def get_client(
        self,
        account: str | None = None,
        model: str | None = None,
    ) -> LLMClient:
        ...
