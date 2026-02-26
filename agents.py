from typing import Any

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


class HuggingFaceMessageNormalizer(BaseChatModel):
    """Wraps a HuggingFace chat model so messages are normalized to BaseMessage instances before invoke.
    Fixes 'Last message must be a HumanMessage!' when the agent passes message dicts or other representations.
    """

    # Inner may be BaseChatModel or RunnableBinding (returned by bind_tools); use Any so Pydantic accepts both
    inner: Any

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = list(convert_to_messages(messages))
        messages = _fold_system_into_user_for_hf(messages)
        messages = _ensure_hf_chat_compat(messages)
        if isinstance(self.inner, BaseChatModel):
            return self.inner._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        # inner is RunnableBinding (from bind_tools); invoke returns AIMessage
        response = self.inner.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        messages = list(convert_to_messages(messages))
        messages = _fold_system_into_user_for_hf(messages)
        messages = _ensure_hf_chat_compat(messages)
        if isinstance(self.inner, BaseChatModel):
            return await self.inner._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        response = await self.inner.ainvoke(messages)
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        bound = self.inner.bind_tools(tools, **kwargs)
        return HuggingFaceMessageNormalizer(inner=bound)

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
            print(state["messages"])
            return {
                "messages": state["messages"][:-1] + [HumanMessage(content=augmented_message_content, id=query_id)],
                "context": retrieved_docs,
            }

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.retrieve_middleware = RetrieveDocumentsMiddleware(vector_store=self.vector_store)

    def create_agent(self, model, provider=None, response_format: BaseModel = None):
        """Create an agent with the RAG middleware."""
        if provider:
            # Use max_new_tokens for HuggingFace to avoid conflict with pipeline's max_length
            model_kwargs = {"max_new_tokens": 1024} if provider == "huggingface" else {"max_tokens": 1024}
            model = init_chat_model(model, model_provider=provider, **model_kwargs)
            if provider == "huggingface":
                model = HuggingFaceMessageNormalizer(inner=model)
        return create_agent(
            model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=response_format,
            middleware=[self.retrieve_middleware],
        )