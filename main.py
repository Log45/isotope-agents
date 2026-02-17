from document_loader import load_and_process_pdf
import os

if __name__ == "__main__":
    for file in os.listdir("./papers"):
        docs = load_and_process_pdf(f"./papers/{file}")
        out_string = ""
        for doc in docs:
            out_string += f"Section: {doc.metadata['section']}, Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}, Source: {doc.metadata.get('source', 'N/A')}\n"
            out_string += doc.page_content + "\n" 
            out_string += "-" * 80 + "\n"
        with open(f"./out/{file}.txt", "w", encoding="utf-8") as f:        
            f.write(out_string)