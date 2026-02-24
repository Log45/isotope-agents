from document_loader import load_and_process_pdf, create_vector_store, extract_paper_metadata
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.messages import HumanMessage
from agents import RAGPipeline, TARGET_MATERIAL_PROMPT, ACID_PROMPT, RESIN_PROMPT, ELUTION_PROMPT, FINAL_PRODUCT_PROMPT
from dotenv import load_dotenv
from models import PaperMetadata, TargetMaterial, AcidOrSolvent, ResinOrColumn, ElutionCondition, FinalProduct, IsotopeProcessFormat
import os
import json

load_dotenv()

def test_document_loader():
    for file in os.listdir("./papers"):
        docs = load_and_process_pdf(f"./papers/{file}")
        docs = [doc for doc in docs if len(doc.page_content.split()) > 1]
        out_string = ""
        for doc in docs:
            out_string += f"Section: {doc.metadata['section']}, Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}\n"
            out_string += doc.page_content + "\n" 
            out_string += "-" * 80 + "\n"
        with open(f"./out/{file}.txt", "w", encoding="utf-8") as f:        
            f.write(out_string)

def extract_isotope_process_info(file_path: str, model="gpt-4o-mini-2024-07-18") -> IsotopeProcessFormat | None:
    try:
        docs = load_and_process_pdf(file_path)
        embedding_model = OpenAIEmbeddings()
        vector_store = create_vector_store(docs, embedding_model, vectorstore_cls=Chroma, collection_name="isotope_docs")
        rag_pipeline = RAGPipeline(vector_store=vector_store)
        
        paper_metadata = PaperMetadata(**extract_paper_metadata(file_path))
        
        # Extract Target Material Information
        try:
            target_agent = rag_pipeline.create_agent(model=model, response_format=TargetMaterial)
            target_agent_response = target_agent.invoke({"messages": [HumanMessage(TARGET_MATERIAL_PROMPT)]})
            target_material = target_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract target material: {e}")
            target_material = TargetMaterial()
        
        # Extract Acid or Solvent Information
        try:
            acid_agent = rag_pipeline.create_agent(model=model, response_format=AcidOrSolvent)
            acid_agent_response = acid_agent.invoke({"messages": [HumanMessage(ACID_PROMPT)]})
            acid_or_solvent = acid_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract acid/solvent: {e}")
            acid_or_solvent = AcidOrSolvent()
        
        # Extract Resin or Column Information
        try:
            resin_agent = rag_pipeline.create_agent(model=model, response_format=ResinOrColumn)
            resin_agent_response = resin_agent.invoke({"messages": [HumanMessage(RESIN_PROMPT)]})
            resin_or_column = resin_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract resin/column: {e}")
            resin_or_column = ResinOrColumn()
        
        # Extract Elution Condition Information
        try:
            elution_agent = rag_pipeline.create_agent(model=model, response_format=ElutionCondition)
            elution_agent_response = elution_agent.invoke({"messages": [HumanMessage(ELUTION_PROMPT)]})
            elution_condition = elution_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract elution conditions: {e}")
            elution_condition = ElutionCondition()
        
        # Extract Final Product Information
        try:
            final_product_agent = rag_pipeline.create_agent(model=model, response_format=FinalProduct)
            final_product_agent_response = final_product_agent.invoke({"messages": [HumanMessage(FINAL_PRODUCT_PROMPT)]})
            final_product = final_product_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract final product: {e}")
            final_product = FinalProduct()
        
        # Clean up vector store after processing so only one paper is in memory at a time
        try:
            # Get all document IDs and delete them
            all_docs = vector_store._collection.get()
            if all_docs and all_docs['ids']:
                vector_store.delete(ids=all_docs['ids'])
        except Exception as e:
            print(f"  Warning: Failed to clean up vector store: {e}")
        
        del docs
        
        return IsotopeProcessFormat(
            paper_metadata=paper_metadata,
            target_materials=target_material,
            acids_and_solvents=acid_or_solvent,
            resins_or_columns=resin_or_column,
            elution_conditions=elution_condition,
            final_products=final_product,
        )
    except Exception as e:
        print(f"  Error processing file: {e}")
        try:
            # Clean up on error too
            all_docs = vector_store._collection.get()
            if all_docs and all_docs['ids']:
                vector_store.delete(ids=all_docs['ids'])
        except Exception as cleanup_error:
            print(f"  Warning: Failed to clean up vector store after error: {cleanup_error}")
        return None
    
    

if __name__ == "__main__":
    for file in os.listdir("./papers"):
        print(f"Processing file: {file}")
        result = extract_isotope_process_info(f"./papers/{file}")
        if result:
            print(f"Extraction Complete for {file}")
            with open(f"./out/{file}_structured.json", "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file}")