"""ctx-weft 协议层。

定义 V1 的硬契约：
- Authorizer / AuthorizationDecision（capability 授权契约：「这次该不该跑」）
- RerunAuthorizer / RerunContext（崩溃恢复时「已经跑过一次、结果不明，还该不该再跑」。
  与 Authorizer 分开注册、分开解析：默认放行型 authorizer 对后一个问题会答「允许」，
  等于自动重跑副作用）
- RecoveryPolicy（provider 在自己的 ToolCapability 上声明：重跑安不安全）
- KnowledgeProvider（§4.2）
- MemoryProvider（§4.3，统一协议：ingest + recall_* + subscribe + apply_compact）
- CapabilityProvider（§4.4）
- HITL：值域（HitlForm / HitlOutcome / PREFACE_*）+ 交换类型（HitlAsk、HitlDecision、
  HitlReply、HitlRequestView）+ 封闭的 Delivery 联合及其三个成员。
  host 与自定义 authorizer/provider 需要构造或读取的全部 HITL 类型都在这里
- AgentTemplate / IdentityFacet（§4.6）
- LLMClient / LLMClientResolver（§11，LLM 接入契约）
- ProviderContext
- Event / EventFilter / EventType / EventBus / EventStore（事件体系，host-facing 契约）

零运行时依赖；所有方法 `async def`；所有返回值是 dataclass。
"""

from ctx_weft.protocols.capability import (
    AgentCapability,
    AgentCapabilityProvider,
    AuthorizationDecision,
    Authorizer,
    Capability,
    CapabilityEvent,
    CapabilityProvider,
    CapabilityProviderInfo,
    Purpose,
    RecoveryPolicy,
    RerunAuthorizer,
    RerunContext,
    SessionScopedCapabilityProvider,
    SkillCapability,
    SkillCapabilityProvider,
    SkillIndexConsumer,
    SkillDefinition,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.context import (
    BLOB_REF_PREFIX,
    ContentPart,
    Citation,
    ImagePart,
    ProviderContext,
    TextPart,
)
from ctx_weft.protocols.events import (
    EVENT_TYPES,
    TRANSIENT_EVENT_TYPES,
    Event,
    EventBlobStore,
    EventBus,
    EventFilter,
    EventStore,
    EventType,
    NullEventBlobStore,
    SubscriptionHandle,
)
from ctx_weft.protocols.filesystem import FS_PROVIDER_NAME, FsTool, SpillSink
from ctx_weft.protocols.hitl import (
    HITL_FORM_APPROVAL,
    HITL_FORM_QUESTION,
    HITL_FORM_WAIT,
    HITL_OUTCOME_ACCEPTED,
    HITL_OUTCOME_CANCELLED,
    HITL_OUTCOME_REJECTED,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    Delivery,
    HitlAsk,
    HitlDecision,
    HitlForm,
    HitlOutcome,
    HitlReply,
    HitlRequestView,
    NoResumeDelivery,
    ToolResultDelivery,
    UserTurnDelivery,
)
from ctx_weft.protocols.llm import (
    # adapter 契约：实现一个 LLM adapter 所需的全部类型
    LLMCallError,
    LLMChunk,
    LLMClient,
    LLMMessage,
    LLMOutageError,
    LLMRequest,
    LLMStream,
    LLMStreamResult,
    StreamingLLMClient,
    LLMTool,
    LLMUsage,
    RAW_ARGS_KEY,
    ToolCall,
    Tokenizer,
    # resolver 契约：仅多账号 LLM provider 需要
    LLMClientResolver,
)
from ctx_weft.protocols.knowledge import (
    KnowledgeDoc,
    KnowledgeProvider,
    KnowledgeProviderInfo,
    KnowledgeQuery,
)
from ctx_weft.protocols.memory import (
    MemoryBlobStore,
    NullMemoryBlobStore,
    EVENT_LAYER,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryKind,
    MemoryLayer,
    MemoryScope,
    MemoryProvider,
    MemoryProviderInfo,
    MemoryRecord,
    Subscription,
)
from ctx_weft.protocols.template import (
    AgentTemplate,
    CapabilityRef,
    IdentityFacet,
    LoopConfig,
    MemoryConfig,
)

__all__ = [
    # Capability
    "AgentCapability",
    "AgentCapabilityProvider",
    "AuthorizationDecision",
    "Authorizer",
    "RerunAuthorizer",
    "RerunContext",
    "RecoveryPolicy",
    "Capability",
    "CapabilityEvent",
    "CapabilityProvider",
    "CapabilityProviderInfo",
    "Purpose",
    "SessionScopedCapabilityProvider",
    "SkillCapability",
    "SkillCapabilityProvider",
    "SkillIndexConsumer",
    "SkillDefinition",
    "ToolCapability",
    "ToolCapabilityProvider",
    # Events
    "EVENT_TYPES",
    "TRANSIENT_EVENT_TYPES",
    "Event",
    "EventBlobStore",
    "EventBus",
    "EventFilter",
    "EventStore",
    "EventType",
    "NullEventBlobStore",
    "SubscriptionHandle",
    # Filesystem capability
    "FS_PROVIDER_NAME",
    "FsTool",
    "SpillSink",
    # HITL
    "HITL_FORM_APPROVAL",
    "HITL_FORM_QUESTION",
    "HITL_FORM_WAIT",
    "HITL_OUTCOME_ACCEPTED",
    "HITL_OUTCOME_CANCELLED",
    "HITL_OUTCOME_REJECTED",
    "PREFACE_AFTER_INTERRUPT",
    "PREFACE_AFTER_INTERRUPT_EDIT",
    "PREFACE_NORMAL",
    "Delivery",
    "HitlAsk",
    "HitlDecision",
    "HitlForm",
    "HitlOutcome",
    "HitlReply",
    "HitlRequestView",
    "NoResumeDelivery",
    "ToolResultDelivery",
    "UserTurnDelivery",
    # Context
    "BLOB_REF_PREFIX",
    "ContentPart",
    "Citation",
    "ImagePart",
    "ProviderContext",
    "TextPart",
    # LLM — adapter 契约
    "LLMCallError",
    "LLMChunk",
    "LLMClient",
    "LLMMessage",
    "LLMOutageError",
    "LLMRequest",
    "LLMStream",
    "LLMStreamResult",
    "StreamingLLMClient",
    "LLMTool",
    "LLMUsage",
    "RAW_ARGS_KEY",
    "ToolCall",
    "Tokenizer",
    # LLM — resolver 契约（多账号 provider）
    "LLMClientResolver",
    # Knowledge
    "KnowledgeDoc",
    "KnowledgeProvider",
    "KnowledgeProviderInfo",
    "KnowledgeQuery",
    # Memory
    "MemoryKind",
    "EVENT_LAYER",
    "MemoryEvent",
    "MemoryEventType",
    "MemoryLayer",  # host 兼容别名（P4c）——host 迁移后删
    "MemoryScope",
    "MemoryProvider",
    "MemoryProviderInfo",
    "MemoryRecord",
    "MemoryAddress",
    "Subscription",
    # Blob store（多模态字节侧，与 MemoryProvider 同模块）
    "MemoryBlobStore",
    "NullMemoryBlobStore",
    # Template
    "AgentTemplate",
    "CapabilityRef",
    "IdentityFacet",
    "LoopConfig",
    "MemoryConfig",
]
