"""LLMProvider — multi-account LLM provider.

Manages multiple LLMAccount configs and resolves LLMClient instances.
Implements the LLMClientResolver protocol so it can be registered
into ProviderRegistry via providers.register_llm_provider().
"""

from __future__ import annotations

import dataclasses
import logging
import time
from dataclasses import dataclass, field

from ctx_weft.core.utils.estimate import default_output_reserve
from ctx_weft.protocols import LLMClient, LLMMessage, LLMRequest
from ctx_weft.providers.llm.store import LLMAccountStoreProtocol

logger = logging.getLogger(__name__)

# 纯文本与多模态是**两个 style**，不是一个 style 加一个开关——能力由「注册了哪个
# adapter 类」表达（spec 2026-08-28-multimodal-adapter-dispatch §2「类型即声明」）。
# 存量 account 的 "anthropic" / "openai" 行为不变，仍拿到纯文本 adapter，零迁移。
SUPPORTED_STYLES = {
    "anthropic", "anthropic-multimodal",
    "openai", "openai-multimodal",
}

# 模型可用性探测的补全预算。不用 1：个别推理模型对极小 max_tokens 会直接拒绝，
# 取一个小而安全的值——只需请求被接受、流能跑完，不关心实际产出多少 token。
_PROBE_MAX_TOKENS = 16


@dataclass
class ModelConfig:
    name: str
    context_limit: int
    # 输入侧输出预留：None → get_client 按窗口尺寸取默认(default_output_reserve)。喂
    # reserved_output_tokens，不参与 per-request 输出上限（那是 output_ceiling）。
    output_reserve: int | None = None
    output_ceiling: int | None = None  # 单次输出收紧上限；None → 网关回退 context_limit


@dataclass
class LLMAccount:
    """One API endpoint + credentials + list of available models."""

    name: str
    style: str          # 见 24 行上方 SUPPORTED_STYLES（"anthropic" | "anthropic-multimodal" | "openai" | "openai-multimodal"）
    api_key: str
    base_url: str = ""
    models: list[ModelConfig] = field(default_factory=list)
    default_model: str = ""
    timeout_sec: int = 120


# ── Internal wrapper ──────────────────────────────────────────────────────────


class _FixedModelClient:
    """Thin LLMClient wrapper that injects a fixed model into every request."""

    def __init__(
        self,
        adapter: LLMClient,
        model: str,
        context_limit: int,
        output_reserve: int,
        output_ceiling: int | None = None,
        account: str = "",
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._context_limit = context_limit
        self._output_reserve = output_reserve
        self._output_ceiling = output_ceiling
        self._account = account

    @property
    def model(self) -> str:
        """实际解析出的模型名（account/model 缺省经 default 解析后的真值）。
        duck-typed 非协议必需：runtime 执行前经 getattr 读取回填 session.llm_model，
        事件层（resolve_llm_identity）才有真值可报，不落 "mock" 兜底。"""
        return self._model

    @property
    def account(self) -> str:
        """实际解析出的账号名（同上，回填 session.llm_provider）。"""
        return self._account

    @property
    def context_limit(self) -> int:
        return self._context_limit

    @property
    def output_reserve(self) -> int:
        return self._output_reserve

    @property
    def output_ceiling(self) -> int | None:
        return self._output_ceiling

    @property
    def supports_tool_calling(self) -> bool:
        return self._adapter.supports_tool_calling

    def complete(self, request: LLMRequest, stream: bool = True):
        return self._adapter.complete(
            dataclasses.replace(request, model=self._model), stream
        )

    def open_stream(self, request: LLMRequest, stream: bool = True):
        from ctx_weft.providers.llm.stream import ensure_collected_stream
        return ensure_collected_stream(self.complete(request, stream))

    @property
    def tokenizer(self):
        return self._adapter.tokenizer_for(self._model)


# ── LLMProvider ───────────────────────────────────────────────────────────────


class LLMProvider:
    """Multi-account LLM provider. Registered into ProviderRegistry.

    Usage:
        store = LLMAccountStore()
        provider = LLMProvider(store)
        provider.load_from_store()
        registry.register_llm_provider(provider)

        # Later, in session execution:
        llm = provider.get_client(account="claude", model="claude-opus-4-7")
    """

    def __init__(self, store: LLMAccountStoreProtocol, *, max_http_retries: int = 3) -> None:
        self._accounts: dict[str, LLMAccount] = {}
        self._adapters: dict[str, LLMClient] = {}
        self._store = store
        self._max_http_retries = max_http_retries

    # ── Account CRUD ──────────────────────────────────────────────────────────

    def register_account(self, account: LLMAccount, *, persist: bool = True) -> None:
        if account.name in self._accounts:
            raise KeyError(f"LLM account already registered: '{account.name}'")
        if account.style not in SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style '{account.style}'. Must be one of: {SUPPORTED_STYLES}")
        if not account.default_model and account.models:
            account.default_model = account.models[0].name

        self._adapters[account.name] = self._build_adapter(account)
        self._accounts[account.name] = account
        if persist:
            self._store.save(account)
        logger.info("LLMProvider: registered account '%s' (%s)", account.name, account.style)

    def delete_account(self, name: str) -> None:
        if name not in self._accounts:
            raise KeyError(f"LLM account not found: '{name}'")
        del self._accounts[name]
        del self._adapters[name]
        self._store.delete(name)

    def get_account(self, name: str) -> LLMAccount:
        if name not in self._accounts:
            raise KeyError(f"LLM account not found: '{name}'")
        return self._accounts[name]

    def list_accounts(self) -> list[LLMAccount]:
        return list(self._accounts.values())

    def is_registered(self, name: str) -> bool:
        return name in self._accounts

    # ── Model management ─────────────────────────────────────────────────────

    def add_model(
        self,
        name: str,
        model: str,
        context_limit: int | None = None,
        output_reserve: int | None = None,
        output_ceiling: int | None = None,
    ) -> LLMAccount:
        account = self.get_account(name)
        if not any(m.name == model for m in account.models):
            account.models.append(ModelConfig(
                name=model,
                context_limit=context_limit or 128_000,
                output_reserve=output_reserve,     # None → get_client 按窗口尺寸取默认
                output_ceiling=output_ceiling,     # None → 网关回退 context_limit
            ))
        if not account.default_model:
            account.default_model = model
        self._store.save(account)
        return account

    def remove_model(self, name: str, model: str) -> LLMAccount:
        account = self.get_account(name)
        account.models = [m for m in account.models if m.name != model]
        if account.default_model == model:
            account.default_model = account.models[0].name if account.models else ""
        self._store.save(account)
        return account

    def set_default_model(self, name: str, model: str) -> LLMAccount:
        account = self.get_account(name)
        if not any(m.name == model for m in account.models):
            raise ValueError(f"Model '{model}' not in account '{name}'")
        account.default_model = model
        self._store.save(account)
        return account

    # ── Client resolution ─────────────────────────────────────────────────────

    def get_client(
        self,
        account: str | None = None,
        model: str | None = None,
    ) -> LLMClient:
        """Resolve a LLMClient for the given account + model.

        account=None → first registered account.
        model=None   → account's default_model.
        """
        if not self._accounts:
            raise RuntimeError("No LLM accounts registered. Register one via LLMProvider.register_account() (or have the host register it) before resolving a client.")

        name = account or next(iter(self._accounts))
        if name not in self._accounts:
            # 账号可能已被删除（会话/任务仍钉着旧账号名）→ 回退到默认账号 + 默认模型，
            # 而不是直接 KeyError 让任务失败。沿用旧模型名会落到默认账号上不匹配，故置空。
            logger.warning("LLM account '%s' not found; falling back to default account.", name)
            name = next(iter(self._accounts))
            model = None
        acc = self.get_account(name)
        adapter = self._adapters[name]

        resolved_model = model or acc.default_model
        if not resolved_model:
            raise ValueError(
                f"No model specified and account '{name}' has no default model. "
                "Add one via LLMProvider.add_model() or set a default via LLMProvider.set_default_model()."
            )

        model_cfg = next((m for m in acc.models if m.name == resolved_model), None)
        ctx_limit = model_cfg.context_limit if model_cfg else 128_000
        # output_reserve 未配（None）→ 按窗口尺寸取默认；显式值（含 0）原样采用。
        cfg_reserve = model_cfg.output_reserve if model_cfg else None
        reserve = cfg_reserve if cfg_reserve is not None else default_output_reserve(ctx_limit)
        ceiling = model_cfg.output_ceiling if model_cfg else None

        return _FixedModelClient(adapter, resolved_model, ctx_limit, reserve, ceiling, account=name)

    # ── Model discovery / connectivity (host-facing; not on the protocol) ──────

    async def fetch_models(self, style: str, api_key: str, base_url: str = "") -> list[str]:
        """List models for ad-hoc (unregistered) credentials. Builds a transient adapter."""
        probe = LLMAccount(name="_probe", style=style, api_key=api_key, base_url=base_url)
        adapter = self._build_adapter(probe)  # raises ValueError on unsupported style
        return await adapter.list_models()

    async def fetch_models_for_account(self, name: str) -> list[str]:
        """List models for a registered account (reuses its live adapter)."""
        self.get_account(name)  # raises KeyError if missing
        return await self._adapters[name].list_models()

    async def ping(self, style: str, api_key: str, base_url: str = "") -> float:
        """Probe ad-hoc credentials; returns latency in ms. Raises on failure."""
        t0 = time.perf_counter()
        await self.fetch_models(style, api_key, base_url)
        return (time.perf_counter() - t0) * 1000.0

    async def ping_account(self, name: str) -> float:
        """Probe a registered account; returns latency in ms. Raises on failure."""
        t0 = time.perf_counter()
        await self.fetch_models_for_account(name)
        return (time.perf_counter() - t0) * 1000.0

    async def _probe_model(self, adapter: LLMClient, model: str) -> None:
        """Send one minimal completion with `model` and drain the stream.

        Validates that the model name exists AND the account has permission to
        call it: a bad/unauthorized model makes the adapter raise LLMCallError
        (HTTP 4xx) instead of merely listing models. Unlike fetch_models/ping,
        this actually exercises the configured model.
        """
        request = LLMRequest(
            model=model,
            system="",
            messages=[LLMMessage(role="user", content="ping")],
            max_tokens=_PROBE_MAX_TOKENS,
        )
        async for _ in adapter.complete(request, stream=True):
            pass

    async def verify_model(self, style: str, api_key: str, base_url: str = "", *, model: str) -> float:
        """Verify a model is callable with ad-hoc credentials; returns latency in ms.

        Builds a transient adapter and runs a minimal completion. Raises on failure
        (unsupported style, bad credentials, unknown model, no permission)."""
        probe = LLMAccount(name="_probe", style=style, api_key=api_key, base_url=base_url)
        adapter = self._build_adapter(probe)  # raises ValueError on unsupported style
        t0 = time.perf_counter()
        await self._probe_model(adapter, model)
        return (time.perf_counter() - t0) * 1000.0

    async def verify_model_for_account(self, name: str, model: str) -> float:
        """Verify a model is callable for a registered account (reuses its live adapter);
        returns latency in ms. Raises KeyError if missing, or on call failure."""
        self.get_account(name)  # raises KeyError if missing
        t0 = time.perf_counter()
        await self._probe_model(self._adapters[name], model)
        return (time.perf_counter() - t0) * 1000.0

    # ── Persistence ───────────────────────────────────────────────────────────

    def load_from_store(self) -> int:
        """Load all accounts from store. Returns count loaded."""
        count = 0
        for account in self._store.list_all():
            if account.name in self._accounts:
                continue
            try:
                self.register_account(account, persist=False)
                count += 1
            except Exception:
                logger.exception("LLMProvider: failed to load account '%s'", account.name)
        if count:
            logger.info("LLMProvider: restored %d account(s) from store", count)
        return count

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_adapter(self, account: LLMAccount) -> LLMClient:
        # 分派必须精确匹配：fetch_models / verify_model 两个 ungated call site 靠 _build_adapter
        # 自己拒绝未知 style。前缀匹配会让 "anthropicX" 这样的近似值也被接受（降级成纯文本），
        # 破坏了入口防御（只有 register_account 经过 SUPPORTED_STYLES 门控）。故必须精确匹配。
        style = account.style
        if style in ("anthropic", "anthropic-multimodal"):
            from ctx_weft.providers.llm.anthropic import (
                AnthropicAdapter,
                AnthropicMultimodalAdapter,
            )
            cls = AnthropicMultimodalAdapter if style == "anthropic-multimodal" else AnthropicAdapter
            return cls(
                api_key=account.api_key,
                base_url=account.base_url or "https://api.anthropic.com",
                timeout_sec=account.timeout_sec,
                max_http_retries=self._max_http_retries,
            )
        if style in ("openai", "openai-multimodal"):
            from ctx_weft.providers.llm.openai import (
                OpenAIAdapter,
                OpenAIMultimodalAdapter,
            )
            cls = OpenAIMultimodalAdapter if style == "openai-multimodal" else OpenAIAdapter
            return cls(
                api_key=account.api_key,
                base_url=account.base_url or "https://api.openai.com",
                timeout_sec=account.timeout_sec,
                max_http_retries=self._max_http_retries,
            )
        raise ValueError(f"Unsupported style: {account.style}")
