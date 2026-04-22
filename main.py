from io import TextIOWrapper

from document_loader import load_and_process_pdf, create_vector_store, extract_paper_metadata
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.messages import HumanMessage
from agents import RAGPipeline, TARGET_MATERIAL_PROMPT, ACID_PROMPT, RESIN_PROMPT, ELUTION_PROMPT, FINAL_PRODUCT_PROMPT, SYSTEM_PROMPT, EXTRACT_ALL_PROMPT, SUMMARIZE_PAPER_PROMPT, SUMMARIZE_SECTION_PROMPT, SummarizerAgent, IsotopeProcessExtractionAgent, release_model_registry
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
from time import perf_counter

from model_config import ModelConfig, load_model_config
import yaml

load_dotenv()
DEFAULT_VRAM_SOFT_LIMIT_MB = int(os.environ.get("VRAM_SOFT_LIMIT_MB", "0") or "0")


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


def _to_yaml_serializable(obj):
    """Convert runtime objects (e.g., torch dtypes/configs) to YAML-safe values."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, torch.dtype):
        # Keep readable values such as "bfloat16" / "float16"
        return str(obj).replace("torch.", "")
    if dataclasses.is_dataclass(obj):
        return _to_yaml_serializable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_yaml_serializable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _to_yaml_serializable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "model_dump"):
        try:
            return _to_yaml_serializable(obj.model_dump())
        except Exception:
            pass
    return str(obj)


def _log_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[gpu] {label}: cuda not available")
        return
    snapshots = []
    for idx in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(idx) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(idx) / (1024 * 1024)
        snapshots.append(f"gpu{idx}: alloc={allocated:.0f}MB reserved={reserved:.0f}MB")
    print(f"[gpu] {label}: " + " | ".join(snapshots))


def _check_vram_soft_limit(limit_mb: int, label: str) -> bool:
    """Return True when memory is over soft limit."""
    if limit_mb <= 0 or not torch.cuda.is_available():
        return False
    over_limit = False
    for idx in range(torch.cuda.device_count()):
        reserved = torch.cuda.memory_reserved(idx) / (1024 * 1024)
        if reserved > limit_mb:
            over_limit = True
    if over_limit:
        print(
            f"[gpu] Soft limit exceeded after {label}: "
            f"reserved memory over {limit_mb}MB. "
            "Skipping non-essential phases to avoid OOM."
        )
    return over_limit


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
            
def extract_info_full_summary( file_path: str,
    model: str = "gpt-4o-mini-2024-07-18",
    provider: str | None = None,
    model_config: ModelConfig | None = None,
    ) -> IsotopeProcessFormat:
    try:
        _log_gpu_memory("before full summary pipeline")
        sections = load_and_process_pdf(file_path)
        summarizer = SummarizerAgent(model=model, provider=provider, config=model_config)
        _log_gpu_memory("after summarizer init (full)")
        summaries = summarizer.summarize_full(sections)
        
        isotope_agent = IsotopeProcessExtractionAgent(model=model, provider=provider, config=model_config)
        isotope_response = isotope_agent.extract_from_summaries(summaries)
        paper_metadata = PaperMetadata(**extract_paper_metadata(file_path))
        isotope_response.paper_metadata = paper_metadata
        return isotope_response
    except Exception as e:
        print(f"  Error in full summary extraction: {e}")
        traceback.print_exc()
        return IsotopeProcessFormat(paper_metadata=PaperMetadata(**extract_paper_metadata(file_path)), target_materials=TargetMaterialList(), acids_and_solvents=AcidOrSolventList(), resins_or_columns=ResinOrColumnList(), elution_conditions=ElutionConditionList(), final_products=FinalProductList())
            
def extract_info_sections_summary(
    file_path: str,
    model: str = "gpt-4o-mini-2024-07-18",
    provider: str | None = None,
    model_config: ModelConfig | None = None,
    ) -> IsotopeProcessFormat:
    try:
        _log_gpu_memory("before section summary pipeline")
        sections = load_and_process_pdf(file_path)
        summarizer = SummarizerAgent(model=model, provider=provider, config=model_config)
        _log_gpu_memory("after summarizer init (sections)")
        summaries = summarizer.summarize_sections(sections)
        
        isotope_agent = IsotopeProcessExtractionAgent(model=model, provider=provider, config=model_config)
        isotope_response = isotope_agent.extract_from_summaries(summaries)
        paper_metadata = PaperMetadata(**extract_paper_metadata(file_path))
        isotope_response.paper_metadata = paper_metadata
        return isotope_response
    except Exception as e:
        print(f"  Error in section summary extraction: {e}")
        traceback.print_exc()
        return IsotopeProcessFormat(paper_metadata=PaperMetadata(**extract_paper_metadata(file_path)), target_materials=TargetMaterialList(), acids_and_solvents=AcidOrSolventList(), resins_or_columns=ResinOrColumnList(), elution_conditions=ElutionConditionList(), final_products=FinalProductList())

def extract_isotope_process_info(
    file_path: str,
    model: str = "gpt-4o-mini-2024-07-18",
    provider: str | None = None,
    model_config: ModelConfig | None = None,
    logging: bool = False,
) -> IsotopeProcessFormat | None:
    try:
        print("Starting extraction process")
        _log_gpu_memory("before rag pipeline")
        docs = load_and_process_pdf(file_path)
        embedding_model = OpenAIEmbeddings()
        vector_store = create_vector_store(docs, embedding_model, vectorstore_cls=Chroma, collection_name="isotope_docs")
        rag_pipeline = RAGPipeline(vector_store=vector_store)
        _log_gpu_memory("after rag pipeline init")
        
        paper_metadata = PaperMetadata(**extract_paper_metadata(file_path))
        
        if logging:
            log_path = f"./log/{os.path.basename(file_path)}.log"
            with open(log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"Processing file: {file_path}\n")
                log_file.write(f"Extracted Paper Metadata:\n{paper_metadata.model_dump_json(indent=2)}\n\n")
        
        # Extract Target Material Information
        try:
            print("Extracting target materials...")
            target_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=TargetMaterialList, config=model_config
            )
            _log_gpu_memory("after target agent init")
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
            print("Extracting acid information...")
            acid_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=AcidOrSolventList, config=model_config
            )
            _log_gpu_memory("after acid agent init")
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
            print("Extracting resin information...")
            resin_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=ResinOrColumnList, config=model_config
            )
            _log_gpu_memory("after resin agent init")
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
            print("Extracting elution information...")
            elution_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=ElutionConditionList, config=model_config
            )
            _log_gpu_memory("after elution agent init")
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
            print("Extracting final product(s)...")
            final_product_agent = rag_pipeline.create_agent(
                model=model, provider=provider, response_format=FinalProductList, config=model_config
            )
            _log_gpu_memory("after final-product agent init")
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
        cfg_payload = _to_yaml_serializable(cfg) if cfg is not None else {}
        yaml.safe_dump(cfg_payload, f, default_flow_style=False, sort_keys=False)
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
    
def test_sections_and_full_summary(cfg: ModelConfig | None = None):
    exp_dir = get_save_folder()
    log_file = set_log_path(exp_dir)
    # save prompts to experiment folder
    os.mkdir(f"{exp_dir}/prompts")
    os.mkdir(f"{exp_dir}/section_summaries")
    os.mkdir(f"{exp_dir}/full_summary")
    os.mkdir(f"{exp_dir}/rag_output")
    with open(f"{exp_dir}/prompts/summarize_section.md", "w") as f:
        f.write(SUMMARIZE_SECTION_PROMPT)
    with open(f"{exp_dir}/prompts/summarize_paper.md", "w") as f:
        f.write(SUMMARIZE_PAPER_PROMPT)
    with open(f"{exp_dir}/config.yaml", "w") as f:
        cfg_payload = _to_yaml_serializable(cfg) if cfg is not None else {}
        yaml.safe_dump(cfg_payload, f, default_flow_style=False, sort_keys=False)
    for file in os.listdir("./papers"):
        print(f"Processing file: {file}")
        summarize_sections_result = extract_info_sections_summary(f"./papers/{file}", model_config=cfg)
        if summarize_sections_result:
            print(f"Section Summary Extraction Complete for {file}")
            with open(f"{exp_dir}/section_summaries/{file}.json", "w", encoding="utf-8") as f:
                f.write(summarize_sections_result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file} section summaries")
        summarize_full_result = extract_info_full_summary(f"./papers/{file}", model_config=cfg)
        if summarize_full_result:
            print(f"Full Summary Extraction Complete for {file}")
            with open(f"{exp_dir}/full_summary/{file}.json", "w", encoding="utf-8") as f:
                f.write(summarize_full_result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file} full summary")
        break # just test one paper
    close_log_file(log_file)


def test_section_summary_only(cfg: ModelConfig | None = None):
    exp_dir = get_save_folder()
    log_file = set_log_path(exp_dir)
    os.mkdir(f"{exp_dir}/prompts")
    os.mkdir(f"{exp_dir}/section_summaries")
    with open(f"{exp_dir}/prompts/summarize_section.md", "w") as f:
        f.write(SUMMARIZE_SECTION_PROMPT)
    with open(f"{exp_dir}/config.yaml", "w") as f:
        cfg_payload = _to_yaml_serializable(cfg) if cfg is not None else {}
        yaml.safe_dump(cfg_payload, f, default_flow_style=False, sort_keys=False)
    for file in os.listdir("./papers"):
        print(f"Processing file: {file}")
        summarize_sections_result = extract_info_sections_summary(f"./papers/{file}", model_config=cfg)
        if summarize_sections_result:
            with open(f"{exp_dir}/section_summaries/{file}.json", "w", encoding="utf-8") as f:
                f.write(summarize_sections_result.model_dump_json(indent=2))
        break
    close_log_file(log_file)


def test_full_summary_only(cfg: ModelConfig | None = None):
    exp_dir = get_save_folder()
    log_file = set_log_path(exp_dir)
    os.mkdir(f"{exp_dir}/prompts")
    os.mkdir(f"{exp_dir}/full_summary")
    with open(f"{exp_dir}/prompts/summarize_paper.md", "w") as f:
        f.write(SUMMARIZE_PAPER_PROMPT)
    with open(f"{exp_dir}/config.yaml", "w") as f:
        cfg_payload = _to_yaml_serializable(cfg) if cfg is not None else {}
        yaml.safe_dump(cfg_payload, f, default_flow_style=False, sort_keys=False)
    for file in os.listdir("./papers"):
        print(f"Processing file: {file}")
        summarize_full_result = extract_info_full_summary(f"./papers/{file}", model_config=cfg)
        if summarize_full_result:
            with open(f"{exp_dir}/full_summary/{file}.json", "w", encoding="utf-8") as f:
                f.write(summarize_full_result.model_dump_json(indent=2))
        break
    close_log_file(log_file)
    
def test_all(cfg: ModelConfig | None = None, vram_soft_limit_mb: int = DEFAULT_VRAM_SOFT_LIMIT_MB):
    exp_dir = get_save_folder()
    log_file = set_log_path(exp_dir)
    # save prompts to experiment folder
    os.mkdir(f"{exp_dir}/prompts")
    os.mkdir(f"{exp_dir}/section_summaries")
    os.mkdir(f"{exp_dir}/full_summary")
    os.mkdir(f"{exp_dir}/rag_output")
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
    with open(f"{exp_dir}/prompts/extract_all.md", "w") as f:
        f.write(EXTRACT_ALL_PROMPT)
    with open(f"{exp_dir}/prompts/summarize_section.md", "w") as f:
        f.write(SUMMARIZE_SECTION_PROMPT)
    with open(f"{exp_dir}/prompts/summarize_paper.md", "w") as f:
        f.write(SUMMARIZE_PAPER_PROMPT)
    with open(f"{exp_dir}/config.yaml", "w") as f:
        cfg_payload = _to_yaml_serializable(cfg) if cfg is not None else {}
        yaml.safe_dump(cfg_payload, f, default_flow_style=False, sort_keys=False)
    for file in os.listdir("./papers"):
        print(f"Processing file: {file}")
        file_path = f"./papers/{file}"
        _log_gpu_memory("start file benchmark")
        sections_t1 = perf_counter()
        summarize_sections_result = extract_info_sections_summary(file_path, model_config=cfg)
        sections_t2 = perf_counter()
        if summarize_sections_result:
            print(f"Extraction Complete for {file} section summaries")
            with open(f"{exp_dir}/section_summaries/{file}.json", "w", encoding="utf-8") as f:
                f.write(summarize_sections_result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file} section summaries")
        release_model_registry()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_gpu_memory("after section summary phase")
        if _check_vram_soft_limit(vram_soft_limit_mb, "section summary phase"):
            print(f"Skipping remaining phases for {file} due to VRAM soft limit.")
            continue
        full_summary_t1 = perf_counter()
        summarize_full_result = extract_info_full_summary(file_path, model_config=cfg)
        full_summary_t2 = perf_counter()
        if summarize_full_result:
            print(f"Extraction Complete for {file} full summary")
            with open(f"{exp_dir}/full_summary/{file}.json", "w", encoding="utf-8") as f:
                f.write(summarize_full_result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file} full summary")
        release_model_registry()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_gpu_memory("after full summary phase")
        if _check_vram_soft_limit(vram_soft_limit_mb, "full summary phase"):
            print(f"Skipping RAG phase for {file} due to VRAM soft limit.")
            continue
        rag_t1 = perf_counter()
        rag_result = extract_isotope_process_info(f"./papers/{file}", model_config=cfg, logging=False)
        rag_t2 = perf_counter()
        if rag_result:
            print(f"RAG Extraction Complete for {file} with RAG")
            with open(f"{exp_dir}/rag_output/{file}.json", "w", encoding="utf-8") as f:
                f.write(rag_result.model_dump_json(indent=2))
        else:
            print(f"Failed to extract from {file} with RAG")
        print(f"Timing for {file}: Section Summary: {sections_t2 - sections_t1:.2f}s, Full Summary: {full_summary_t2 - full_summary_t1:.2f}s, RAG: {rag_t2 - rag_t1:.2f}s")
        # save times to file for benchmarking
        with open(f"{exp_dir}/timing.json", "a", encoding="utf-8") as f:
            json.dump({
                "file": file,
                "section_summary_time": sections_t2 - sections_t1,
                "full_summary_time": full_summary_t2 - full_summary_t1,
                "rag_time": rag_t2 - rag_t1
            }, f, indent=2)
    close_log_file(log_file)


def run_mode(mode: str, cfg: ModelConfig, vram_soft_limit_mb: int = DEFAULT_VRAM_SOFT_LIMIT_MB):
    if mode == "rag":
        main(cfg)
        return
    if mode == "section_summary":
        test_section_summary_only(cfg)
        return
    if mode == "full_summary":
        test_full_summary_only(cfg)
        return
    if mode == "all":
        print("Warning: mode=all is high-memory; phases run sequentially with registry releases.")
        test_all(cfg, vram_soft_limit_mb=vram_soft_limit_mb)
        return
    raise ValueError(f"Unknown mode: {mode}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML (supports OpenAI and HuggingFace).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("rag", "section_summary", "full_summary", "all"),
        default="rag",
        help="Run mode. Use 'all' only for full benchmark (high memory).",
    )
    parser.add_argument(
        "--vram-soft-limit-mb",
        type=int,
        default=DEFAULT_VRAM_SOFT_LIMIT_MB,
        help="Optional soft VRAM reserve limit. If exceeded in mode=all, later phases are skipped.",
    )
    args = parser.parse_args()
    
    if args.config:
        cfg = load_model_config(args.config)
        run_mode(args.mode, cfg, vram_soft_limit_mb=args.vram_soft_limit_mb)
    else:
        for cfg in os.listdir("config"):
            if "30B" in cfg and "4bit" not in cfg: # skip QWEN 30B that is not quantized due to CUDA memory error.
                continue
            if "hf" in cfg.lower() or "openai" in cfg.lower():
                cfg = load_model_config(f"config/{cfg}")
                run_mode(args.mode, cfg, vram_soft_limit_mb=args.vram_soft_limit_mb)