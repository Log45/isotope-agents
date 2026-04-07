from io import TextIOWrapper

from document_loader import load_and_process_pdf, create_vector_store, extract_paper_metadata
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.messages import HumanMessage
from agents import RAGPipeline, TARGET_MATERIAL_PROMPT, ACID_PROMPT, RESIN_PROMPT, ELUTION_PROMPT, FINAL_PRODUCT_PROMPT, SYSTEM_PROMPT
from dotenv import load_dotenv
from models import PaperMetadata, TargetMaterialList, AcidOrSolventList, ResinOrColumnList, ElutionConditionList, FinalProductList, IsotopeProcessFormat
from huggingface_hub import login
import os
import json
import gc
import torch
import traceback
from pathlib import Path
import sys
import argparse
import dataclasses

from model_config import ModelConfig, load_model_config
import yaml

load_dotenv()


def _agent_response_to_json_serializable(obj):
    """Convert agent response (may contain HumanMessage, AIMessage, etc.) to JSON-serializable form."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _agent_response_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_agent_response_to_json_serializable(i) for i in obj]
    return obj


hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)
if hf_token:
    login(hf_token)

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

def extract_isotope_process_info(
    file_path: str,
    model: str = "gpt-4o-mini-2024-07-18",
    provider: str | None = None,
    model_config: ModelConfig | None = None,
    logging: bool = False,
) -> IsotopeProcessFormat | None:
    try:
        docs = load_and_process_pdf(file_path)
        embedding_model = OpenAIEmbeddings()
        vector_store = create_vector_store(docs, embedding_model, vectorstore_cls=Chroma, collection_name="isotope_docs")
        rag_pipeline = RAGPipeline(vector_store=vector_store)
        
        paper_metadata = PaperMetadata(**extract_paper_metadata(file_path))
        
        if logging:
            log_path = f"./log/{os.path.basename(file_path)}.log"
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"Processing file: {file_path}\n")
                log_file.write(f"Extracted Paper Metadata:\n{paper_metadata.model_dump_json(indent=2)}\n\n")
        
        # Extract Target Material Information
        try:
            target_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=TargetMaterialList, config=model_config
            )
            target_agent_response = target_agent.invoke({"messages": [HumanMessage(TARGET_MATERIAL_PROMPT)]})
            if logging:
                print("logging target material response")
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("Target Material Agent Response:\n")
                    log_file.write(json.dumps(_agent_response_to_json_serializable(target_agent_response), indent=2) + "\n\n")
            target_material = target_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract target material: {e}")
            traceback.print_exc()
            target_material = TargetMaterialList()
        
        # Clean up target agent and free memory before creating new agents 
        del target_agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract Acid or Solvent Information
        try:
            acid_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=AcidOrSolventList, config=model_config
            )
            acid_agent_response = acid_agent.invoke({"messages": [HumanMessage(ACID_PROMPT)]})
            if logging:
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("Acid/Solvent Agent Response:\n")
                    log_file.write(json.dumps(_agent_response_to_json_serializable(acid_agent_response), indent=2) + "\n\n")
            acid_or_solvent = acid_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract acid/solvent: {e}")
            traceback.print_exc()
            acid_or_solvent = AcidOrSolventList()
            
        # Clean up acid agent and free memory before creating new agents
        del acid_agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract Resin or Column Information
        try:
            resin_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=ResinOrColumnList, config=model_config
            )
            resin_agent_response = resin_agent.invoke({"messages": [HumanMessage(RESIN_PROMPT)]})
            if logging:
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("Resin/Column Agent Response:\n")
                    log_file.write(json.dumps(_agent_response_to_json_serializable(resin_agent_response), indent=2) + "\n\n")
            resin_or_column = resin_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract resin/column: {e}")
            traceback.print_exc()
            resin_or_column = ResinOrColumnList()
            
        del resin_agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract Elution Condition Information
        try:
            elution_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=ElutionConditionList, config=model_config
            )
            elution_agent_response = elution_agent.invoke({"messages": [HumanMessage(ELUTION_PROMPT)]})
            if logging:
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("Elution Condition Agent Response:\n")
                    log_file.write(json.dumps(_agent_response_to_json_serializable(elution_agent_response), indent=2) + "\n\n")
            elution_condition = elution_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract elution conditions: {e}")
            traceback.print_exc()
            elution_condition = ElutionConditionList()
            
        del elution_agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Extract Final Product Information
        try:
            final_product_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=FinalProductList, config=model_config
            )
            final_product_agent_response = final_product_agent.invoke({"messages": [HumanMessage(FINAL_PRODUCT_PROMPT)]})
            if logging:
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write("Final Product Agent Response:\n")
                    log_file.write(json.dumps(_agent_response_to_json_serializable(final_product_agent_response), indent=2) + "\n\n")
            final_product = final_product_agent_response['structured_response']
        except Exception as e:
            print(f"  Warning: Failed to extract final product: {e}")
            traceback.print_exc()
            final_product = FinalProductList()
        
        # Clean up vector store after processing so only one paper is in memory at a time
        try:
            # Get all document IDs and delete them
            all_docs = vector_store._collection.get()
            if all_docs and all_docs['ids']:
                vector_store.delete(ids=all_docs['ids'])
        except Exception as e:
            print(f"  Warning: Failed to clean up vector store: {e}")
            traceback.print_exc()
        
        del final_product_agent
        del docs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        traceback.print_exc()
        try:
            # Clean up on error too
            all_docs = vector_store._collection.get()
            if all_docs and all_docs['ids']:
                vector_store.delete(ids=all_docs['ids'])
        except Exception as cleanup_error:
            print(f"  Warning: Failed to clean up vector store after error: {cleanup_error}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

def get_save_folder(root: str = "out") -> str:
    root_path = Path(root)
    root_path.mkdir(exist_ok=True)

    # find existing experiment folders
    exp_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("exp")]

    if not exp_dirs:
        exp_name = "exp"
    else:
        nums = []
        for d in exp_dirs:
            suffix = d.name.replace("exp", "")
            nums.append(int(suffix) if suffix.isdigit() else 1)

        exp_name = f"exp{max(nums) + 1}"

    exp_path = root_path / exp_name
    exp_path.mkdir(exist_ok=True)

    return str(exp_path)

def set_log_path(exp_path: str) -> TextIOWrapper:
    log_file = open(f"{exp_path}/log", "w")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file

def close_log_file(log_file: TextIOWrapper) -> None:
    log_file.close()
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__

def main(cfg: ModelConfig | None = None):
    exp_dir = get_save_folder()
    log_file = set_log_path(exp_dir)
    # save prompts to experiment folder
    os.mkdir(f"{exp_dir}/prompts")
    with open(f"{exp_dir}/prompts/target.md", "w") as f:
        f.write(TARGET_MATERIAL_PROMPT)
    with open(f"{exp_dir}/prompts/acid.md", "w") as f:
        f.write(ACID_PROMPT)
    with open(f"{exp_dir}/prompts/elution.md", "w") as f:
        f.write(ELUTION_PROMPT)
    with open(f"{exp_dir}/prompts/products.md", "w") as f:
        f.write(FINAL_PRODUCT_PROMPT)
    with open(f"{exp_dir}/prompts/resin.md", "w") as f:
        f.write(RESIN_PROMPT)
    with open(f"{exp_dir}/prompts/system.md", "w") as f:
        f.write(SYSTEM_PROMPT)
    with open(f"{exp_dir}/config.yaml", "w") as f:
        # ModelConfig is a dataclass:
        yaml.dump(dataclasses.asdict(cfg), f, default_flow_style=False)
    for file in os.listdir("./papers"):
        print(f"Processing file: {file}")
        result = extract_isotope_process_info(f"./papers/{file}", model_config=cfg, logging=False)
        if result:
            print(f"Extraction Complete for {file}")
            with open(f"{exp_dir}/{file}.json", "w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file}")
    close_log_file(log_file)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML (supports OpenAI and HuggingFace).",
    )
    args = parser.parse_args()
    
    if args.config:
        cfg = load_model_config(args.config)
        main(cfg)
    else:
        for cfg in os.listdir("config"):
            if "hf" in cfg:
                cfg = load_model_config(f"config/{cfg}")
                main(cfg)