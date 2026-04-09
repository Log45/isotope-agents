from typing import Any
import json
import re
import time
import random

from openai import APIConnectionError, APITimeoutError, RateLimitError

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, convert_to_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

# Model Providers
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
# Add more providers as needed

from model_config import ModelConfig
from models import IsotopeProcessFormat

# Per-section cap keeps requests well under provider limits; PDF text can be huge.
_MAX_SUMMARY_SECTION_CHARS = 350_000


def _sanitize_llm_text(text: Any, *, max_chars: int | None = _MAX_SUMMARY_SECTION_CHARS) -> str:
    """Make text safe for JSON request bodies (OpenAI rejects malformed UTF-8 / surrogates)."""
    s = text if isinstance(text, str) else str(text or "")
    s = s.replace("\x00", "")
    try:
        s = s.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    except (UnicodeEncodeError, UnicodeDecodeError):
        s = s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    if max_chars is not None and len(s) > max_chars:
        s = s[: max_chars - 40].rstrip() + "\n\n[... content truncated for API limits ...]"
    return s


def _retry_after_seconds_from_openai_message(exc: BaseException) -> float | None:
    """Parse suggested wait from OpenAI error text, e.g. 'Please try again in 886ms'."""
    msg = str(exc)
    m = re.search(r"try again in ([\d.]+)\s*ms", msg, re.IGNORECASE)
    if m:
        return max(float(m.group(1)) / 1000.0, 0.0)
    m = re.search(r"try again in ([\d.]+)\s*s(?:ec(?:ond)?s?)?\b", msg, re.IGNORECASE)
    if m:
        return max(float(m.group(1)), 0.0)
    return None


def _invoke_chat_respecting_rate_limits(model: Any, messages: list[Any], *, max_attempts: int = 12) -> Any:
    """Retry chat completion on OpenAI TPM/RPM limits and transient network failures."""
    for attempt in range(max_attempts):
        try:
            return model.invoke(messages)
        except RateLimitError as e:
            if attempt >= max_attempts - 1:
                raise
            wait = _retry_after_seconds_from_openai_message(e)
            if wait is None:
                wait = min(90.0, 1.5 * (2**attempt))
            wait += random.uniform(0.0, 0.35)
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError):
            if attempt >= max_attempts - 1:
                raise
            wait = min(60.0, 2.0 * (2**attempt)) + random.uniform(0.0, 0.5)
            time.sleep(wait)
    raise RuntimeError("invoke retry loop exited without return or raise")

# TODO: Add quantization support to experiment with larger models
"""
From langchain docs:

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm)


"""


def _role_for_fold(m):
    """Return 'system', 'user', or 'assistant' for folding; None if unknown."""
    if m is None:
        return None
    if isinstance(m, SystemMessage):
        return "system"
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if hasattr(m, "type") and getattr(m, "type", None) == "system":
        return "system"
    if hasattr(m, "role"):
        r = getattr(m, "role", None)
        if r in ("system", "user", "assistant"):
            return r
    # Fallback: check class name in case of different import paths
    name = type(m).__name__
    if name == "SystemMessage":
        return "system"
    if name == "HumanMessage":
        return "user"
    if name == "AIMessage":
        return "assistant"
    return None


def _content_str(m):
    return (m.content or "") if hasattr(m, "content") else ""


def _fold_system_into_user_for_hf(messages):
    """Normalize to strict user/assistant alternation for HuggingFace chat templates.
    - Folds leading system into the first user message.
    - Merges consecutive user messages into one.
    """
    if not messages:
        return messages
    if len(messages) == 1:
        return messages
    out = []
    i = 0
    while i < len(messages):
        r = _role_for_fold(messages[i])
        if r == "system":
            # Collect all consecutive system, then one user if present; fold into single user
            parts = []
            while i < len(messages) and _role_for_fold(messages[i]) == "system":
                parts.append(_content_str(messages[i]))
                i += 1
            if i < len(messages) and _role_for_fold(messages[i]) == "user":
                parts.append(_content_str(messages[i]))
                i += 1
            combined = "\n\n".join(p for p in parts if p).strip()
            out.append(HumanMessage(content=combined or " "))
        elif r == "user":
            # Merge consecutive user messages
            parts = []
            while i < len(messages) and _role_for_fold(messages[i]) == "user":
                parts.append(_content_str(messages[i]))
                i += 1
            combined = "\n\n".join(p for p in parts if p).strip()
            out.append(HumanMessage(content=combined or " "))
        else:
            # assistant or other: keep as-is
            out.append(messages[i])
            i += 1
    return out


def _ensure_hf_chat_compat(messages):
    """
    Final HuggingFace safety pass: always send only a single user message.
    HF templates require strict user/assistant alternation and last message = user;
    the agent often sends [user, assistant] on the second call, so we force a single user.
    """
    msgs = list(messages)
    user_msgs = [m for m in msgs if _role_for_fold(m) == "user"]
    if user_msgs:
        return [user_msgs[-1]]
    return [HumanMessage(content=" ")]


def _is_phi_model(model_name: str | None) -> bool:
    if not model_name:
        return False
    name = model_name.lower()
    return "microsoft/phi-" in name or "/phi-" in name


def _normalize_phi_messages_for_hf(messages):
    """Preserve role structure for Phi chat templates.

    Phi model cards recommend strict system/user/assistant formatting. This keeps
    turns intact (instead of collapsing to one user message) while still ensuring
    a valid ending user turn for HF chat templates.
    """
    msgs = [m for m in messages if _role_for_fold(m) in ("system", "user", "assistant")]
    if not msgs:
        return [HumanMessage(content=" ")]

    # Merge consecutive messages with the same role to maintain alternation.
    merged = []
    for m in msgs:
        role = _role_for_fold(m)
        if merged and _role_for_fold(merged[-1]) == role:
            merged[-1].content = (merged[-1].content or "") + "\n\n" + (_content_str(m) or "")
        else:
            merged.append(m)

    # Ensure conversation doesn't end on assistant (HF templates generally require last=user).
    while merged and _role_for_fold(merged[-1]) == "assistant":
        merged.pop()
    if not merged:
        return [HumanMessage(content=" ")]

    # Ensure at least one user turn exists.
    if not any(_role_for_fold(m) == "user" for m in merged):
        merged.append(HumanMessage(content=" "))

    return merged


class HuggingFaceMessageNormalizer(BaseChatModel):
    """Wraps a HuggingFace chat model so messages are normalized to BaseMessage instances before invoke.
    Fixes 'Last message must be a HumanMessage!' when the agent passes message dicts or other representations.
    """

    # Inner may be BaseChatModel or RunnableBinding (returned by bind_tools); use Any so Pydantic accepts both
    inner: Any
    model_name: str | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = list(convert_to_messages(messages))
        if _is_phi_model(self.model_name):
            messages = _normalize_phi_messages_for_hf(messages)
        else:
            messages = _fold_system_into_user_for_hf(messages)
            messages = _ensure_hf_chat_compat(messages)
        if isinstance(self.inner, BaseChatModel):
            return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        # inner is RunnableBinding (from bind_tools); invoke returns AIMessage
        response = self.inner.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = list(convert_to_messages(messages))
        if _is_phi_model(self.model_name):
            messages = _normalize_phi_messages_for_hf(messages)
        else:
            messages = _fold_system_into_user_for_hf(messages)
            messages = _ensure_hf_chat_compat(messages)
        if isinstance(self.inner, BaseChatModel):
            return await self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        response = await self.inner.ainvoke(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        bound = self.inner.bind_tools(tools, **kwargs)
        return HuggingFaceMessageNormalizer(inner=bound, model_name=self.model_name)

    @property
    def _llm_type(self) -> str:
        return getattr(self.inner, "_llm_type", "huggingface")

with open("./prompts/system.md", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

with open("./prompts/target.md", "r", encoding="utf-8") as f:
    TARGET_MATERIAL_PROMPT = f.read()
    
with open("./prompts/acid.md", "r", encoding="utf-8") as f:
    ACID_PROMPT = f.read()

with open("./prompts/resin.md", "r", encoding="utf-8") as f:
    RESIN_PROMPT = f.read()
    
with open("./prompts/elution.md", "r", encoding="utf-8") as f:
    ELUTION_PROMPT = f.read()
    
with open("./prompts/products.md", "r", encoding="utf-8") as f:
    FINAL_PRODUCT_PROMPT = f.read()
    
with open("./prompts/summarize_paper.md", "r", encoding="utf-8") as f:
    SUMMARIZE_PAPER_PROMPT = f.read()

with open("./prompts/summarize_section.md", "r", encoding="utf-8") as f:
    SUMMARIZE_SECTION_PROMPT = f.read()
    
with open("./prompts/extract_all.md", "r", encoding="utf-8") as f:
    EXTRACT_ALL_PROMPT = f.read()

PROVIDERS = {
    "huggingface": ChatHuggingFace,
    "openai": ChatOpenAI,
}

class State(AgentState):
    context: list[Document]
    
class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
        state_schema = State

        def __init__(self, vector_store: VectorStore):
            super().__init__()
            self.vector_store = vector_store

        def before_model(self, state: AgentState) -> dict[str, object] | None:
            """Retrieve documents and augment the last message with context."""
            query = state["messages"][-1]
            # Support both .content (LangChain) and .text (some representations)
            query_text = getattr(query, "content", None) or getattr(query, "text", "") or ""
            retrieved_docs = self.vector_store.similarity_search(query_text)

            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

            augmented_message_content = (
                f"{query_text}\n\n"
                "Use the following context to answer the query:\n"
                f"{docs_content}"
            )
            # Use the same id as the query so add_messages *replaces* it instead of appending.
            # Otherwise we get [user, user], breaking HuggingFace's "roles must alternate user/assistant" rule.
            query_id = query.get("id") if isinstance(query, dict) else getattr(query, "id", None)
            return {
                "messages": state["messages"][:-1] + [HumanMessage(content=augmented_message_content, id=query_id)],
                "context": retrieved_docs,
            }

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.retrieve_middleware = RetrieveDocumentsMiddleware(vector_store=self.vector_store)
        self._model_cache: dict[str, Any] = {}

    def _cache_key(self, provider: str | None, model: str | None, model_params: dict[str, Any]) -> str:
        try:
            params_key = json.dumps(model_params, sort_keys=True, default=str)
        except Exception:
            params_key = str(model_params)
        return f"{provider}::{model}::{params_key}"

    def _get_or_create_model(self, provider: str, model: str, model_params: dict[str, Any]):
        key = self._cache_key(provider, model, model_params)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached

        model_obj = init_chat_model(model, model_provider=provider, **model_params)
        if provider == "huggingface":
            model_obj = HuggingFaceMessageNormalizer(inner=model_obj, model_name=model)
        self._model_cache[key] = model_obj
        return model_obj

    def create_agent(
        self,
        model: str | None = None,
        provider: str | None = None,
        response_format: BaseModel = None,
        config: ModelConfig | None = None,
    ):
        """Create an agent with the RAG middleware.

        Preferred usage is to pass `config` (loaded from YAML via `load_model_config`).
        """
        if config is not None:
            provider = config.provider
            model = config.model
            model_params = dict(config.params or {})
        else:
            model_params = {}

        if provider:
            # Backwards compatible defaults if caller didn't pass any params
            if not model_params:
                model_params = {"max_new_tokens": 2048} if provider == "huggingface" else {"max_tokens": 2048}

            model_obj = self._get_or_create_model(provider, model, model_params)
        else:
            model_obj = model
        return create_agent(
            model_obj,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=response_format,
            middleware=[self.retrieve_middleware],
        )

class IsotopeProcessExtractionAgent():
    def __init__(self, 
        model: str | None = None,
        provider: str | None = None,
        response_format: BaseModel = IsotopeProcessFormat,
        config: ModelConfig | None = None,):
        self._model_cache: dict[str, Any] = {}
        self.agent = self.create_agent(model=model, provider=provider, response_format=response_format, config=config)
        
    def _cache_key(self, provider: str | None, model: str | None, model_params: dict[str, Any]) -> str:
        try:
            params_key = json.dumps(model_params, sort_keys=True, default=str)
        except Exception:
            params_key = str(model_params)
        return f"{provider}::{model}::{params_key}"

    def _get_or_create_model(self, provider: str, model: str, model_params: dict[str, Any]):
        key = self._cache_key(provider, model, model_params)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached

        model_obj = init_chat_model(model, model_provider=provider, **model_params)
        if provider == "huggingface":
            model_obj = HuggingFaceMessageNormalizer(inner=model_obj, model_name=model)
        self._model_cache[key] = model_obj
        return model_obj
    
    def extract_from_summaries(self, summaries: str) -> IsotopeProcessFormat:
        # LangGraph-backed agents expect an input state dict, not a raw messages list.
        response = self.agent.invoke({"messages": [HumanMessage(content=summaries)]})
        return response["structured_response"]
    
    def create_agent(
        self,
        model: str | None = None,
        provider: str | None = None,
        response_format: BaseModel = None,
        config: ModelConfig | None = None,
    ):
        """Create an agent with the RAG middleware.

        Preferred usage is to pass `config` (loaded from YAML via `load_model_config`).
        """
        if config is not None:
            provider = config.provider
            model = config.model
            model_params = dict(config.params or {})
        else:
            model_params = {}

        if provider:
            # Backwards compatible defaults if caller didn't pass any params
            if not model_params:
                model_params = {"max_new_tokens": 2048} if provider == "huggingface" else {"max_tokens": 2048}

            model_obj = self._get_or_create_model(provider, model, model_params)
        else:
            model_obj = model
        return create_agent(
            model_obj,
            tools=[],
            system_prompt=EXTRACT_ALL_PROMPT,
            response_format=response_format,
        )
    
class SummarizerAgent:
    def __init__(self, model: str | None = None,
        provider: str | None = None,
        response_format: BaseModel = None,
        config: ModelConfig | None = None,):
        self._model_cache: dict[str, Any] = {}
        
        if config is not None:
            provider = config.provider
            model = config.model
            model_params = dict(config.params or {})
        else:
            model_params = {}

        if provider:
            # Backwards compatible defaults if caller didn't pass any params
            if not model_params:
                model_params = {"max_new_tokens": 1024} if provider == "huggingface" else {"max_tokens": 2048}

            self._model = self._get_or_create_model(provider, model, model_params)
        else:
            self._model = model
        
    def _cache_key(self, provider: str | None, model: str | None, model_params: dict[str, Any]) -> str:
        try:
            params_key = json.dumps(model_params, sort_keys=True, default=str)
        except Exception:
            params_key = str(model_params)
        return f"{provider}::{model}::{params_key}"

    def _get_or_create_model(self, provider: str, model: str, model_params: dict[str, Any]):
        key = self._cache_key(provider, model, model_params)
        cached = self._model_cache.get(key)
        if cached is not None:
            return cached

        model_obj = init_chat_model(model, model_provider=provider, **model_params)
        if provider == "huggingface":
            model_obj = HuggingFaceMessageNormalizer(inner=model_obj, model_name=model)
        self._model_cache[key] = model_obj
        return model_obj

    def summarize_full(self, sections: list[Document]) -> str:
        # Sanitize extracted PDF text; no tools / no structured output — use the chat model directly
        # to avoid LangGraph shaping the API request for a simple completion.
        parts = [_sanitize_llm_text(doc.page_content) for doc in sections]
        context = "\n\n".join(parts)
        prompt = f"Paper:\n\n{context}"
        response = _invoke_chat_respecting_rate_limits(
            self._model,
            [
                SystemMessage(content=_sanitize_llm_text(SUMMARIZE_PAPER_PROMPT, max_chars=None)),
                HumanMessage(content=prompt),
            ],
        )
        return _agent_invoke_result_to_text(response)
    
    def summarize_sections(self, sections: list[Document]) -> str:
        summaries = []
        system_content = _sanitize_llm_text(SUMMARIZE_SECTION_PROMPT, max_chars=None)
        for doc in sections:
            section_label = str(doc.metadata.get("section", "Unknown"))
            body = _sanitize_llm_text(doc.page_content)
            prompt = f"Section ({section_label}):\n\n{body}"
            response = _invoke_chat_respecting_rate_limits(
                self._model,
                [SystemMessage(content=system_content), HumanMessage(content=prompt)],
            )
            summaries.append(
                f"Summary of {section_label}:\n{_agent_invoke_result_to_text(response)}"
            )
        return "\n\n".join(summaries)


def _agent_invoke_result_to_text(result: Any) -> str:
    """Normalize various agent.invoke() return shapes into plain text."""
    # Some agents return an AIMessage directly; others return a dict-like state.
    if hasattr(result, "content"):
        return (result.content or "").strip()
    if isinstance(result, dict):
        msgs = result.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            if hasattr(last, "content"):
                return (last.content or "").strip()
        # Fallback: try common keys
        for k in ("output", "text", "content"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return str(result).strip()