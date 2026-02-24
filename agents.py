from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain.agents import create_agent
from pydantic import BaseModel

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
            retrieved_docs = self.vector_store.similarity_search(query.text)

            docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

            augmented_message_content = (
                f"{query.text}\n\n"
                "Use the following context to answer the query:\n"
                f"{docs_content}"
            )
            return {
                "messages": [query.model_copy(update={"content": augmented_message_content})],
                "context": retrieved_docs,
            }

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.retrieve_middleware = RetrieveDocumentsMiddleware(vector_store=self.vector_store)

    def create_agent(self, model, response_format: BaseModel = None):
        """Create an agent with the RAG middleware."""
        return create_agent(
            model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=response_format,
            middleware=[self.retrieve_middleware],
        )