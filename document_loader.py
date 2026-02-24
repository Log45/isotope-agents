import fitz  # PyMuPDF
import re
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Embeddings
from langchain.embeddings import Embeddings # Base class for embeddings, not used directly but required for type hints
from langchain_huggingface.embeddings import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
# Vector Stores
from langchain_core.vectorstores import VectorStore # Base class for vector stores, not used directly but required for type hints
from langchain_community.vectorstores import Chroma, Weaviate, Qdrant, Pinecone, Redis, FAISS


SECTION_PATTERNS = [
    r"^\s*\d+\s+(Abstract|Introduction|Related Works?|Methodology|Results?|Discussion|Conclusion|References)\s*$",
    r"^\s*(Abstract|Introduction|Related Works?|Methodology|Results?|Discussion|Conclusion|References)\s*$",
    r"^\s*\d+\.\d+\s+.+$",      # 2.1 Subsections
    r"^\s*\d+\.\d+\.\d+\s+.+$", # 2.1.1 Sub-subsections
]

SECTION_REGEXES = [
    # Top-level numbered sections: "1. Introduction"
    re.compile(r"^\s*\d+\.\s+[A-Z][A-Za-z].+$"),

    # Subsections: "2.1. Cobalt-55"
    re.compile(r"^\s*\d+\.\d+\.\s+[A-Z][A-Za-z].+$"),
    
    # All caps sections: "METHODS"
    re.compile(r"^[A-Z][A-Z\s\-]{3,}$"),

    # Named sections without numbers (rare but real)
    re.compile(r"^(Abstract|Introduction|Results|Discussion|Conclusion|References)\b"),
]


def load_pdf_sections_structured(file_path):
    import fitz
    doc = fitz.open(file_path)

    documents = []
    current_section = "Front Matter"
    buffer = []


    def flush(page_num):
        nonlocal buffer
        if buffer:
            documents.append(
                Document(
                    page_content="\n".join(buffer).strip(),
                    metadata={
                        "section": current_section,
                        "page": page_num,
                        "source": "PDF"
                    }
                )
            )
            buffer = []

    for page_idx, page in enumerate(doc):
        for line in page.get_text().splitlines():
            clean = line.strip()

            if not clean:
                continue

            if any(r.match(clean) for r in SECTION_REGEXES):
                flush(page_idx)
                current_section = clean
            else:
                buffer.append(clean)

    flush(page_idx)
    return documents

def load_pdf_sections(file_path: str) -> list[Document]:
    doc = fitz.open(file_path)
    docs = []
    current_section = "Unknown"
    buffer = ""

    for page_index, page in enumerate(doc):
        text = page.get_text()
        lines = text.splitlines()

        for line in lines:
            # simple heading detection by pattern
            if line.strip().isupper() and len(line.strip()) < 100:
                # commit previous content
                if buffer.strip():
                    docs.append(Document(page_content=buffer,
                                         metadata={"section": current_section,
                                                   "page": page_index}))
                current_section = line.strip()
                buffer = ""
            else:
                buffer += line + "\n"

    # append last chunk
    if buffer.strip():
        docs.append(Document(page_content=buffer,
                             metadata={"section": current_section,
                                       "page": page_index}))
    return docs

class SectionAwareSplitter(RecursiveCharacterTextSplitter):
    def split_text(self, text: str) -> list[str]:
        # split into paragraphs
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        for p in paras:
            # break long procedural paragraphs
            sub_blocks = re.split(
                r"(?<=\.)\s+(?=(After|Then|Subsequently|Next|Finally))", p
            )
            chunks.extend([b.strip() for b in sub_blocks if b.strip()])

        final = []
        for c in chunks:
            final.extend(super().split_text(c))
        return final

def make_langchain_docs(sections: list[Document], splitter = SectionAwareSplitter(chunk_size=600, chunk_overlap=120)) -> list[Document]:
    result_docs = []
    chunk_id = 0

    for sec in sections:
        text = sec.page_content
        chunks = splitter.split_text(text)

        for c in chunks:
            result_docs.append(Document(
                page_content=c,
                metadata={
                    "section": sec.metadata["section"], 
                    "chunk_id": chunk_id,
                    "source": "PDF",
                }
            ))
            chunk_id += 1

    return result_docs

def extract_tables(file_path: str) -> list[Document]:
    elements = partition_pdf(
        filename=file_path,
        strategy="fast",
        infer_table_structure=True,
    )

    table_docs = []

    for el in elements:
        if isinstance(el, Table):
            table_docs.append(
                Document(
                    page_content=el.text,
                    metadata={
                        "section": "Table",
                        "page": getattr(el.metadata, "page_number", None),
                        "source": "unstructured",
                    },
                )
            )

    return table_docs

def load_and_process_pdf(file_path: str) -> list[Document]:
    sections = load_pdf_sections_structured(file_path)
    table_docs = extract_tables(file_path)
    langchain_docs = make_langchain_docs(sections)

    return langchain_docs + table_docs

def create_vector_store(docs: list[Document], embedding_model: Embeddings, vectorstore_cls: VectorStore = Chroma, **vectorstore_kwargs) -> VectorStore:
    return vectorstore_cls.from_documents(docs, embedding_model, **vectorstore_kwargs)