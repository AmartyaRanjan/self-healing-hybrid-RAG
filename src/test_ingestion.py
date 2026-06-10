import os
import sys
from dotenv import load_dotenv

# Ensure the root project directory is on the system path for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.chunker import MultimodalStructuralChunker
from src.database.chroma_client import ChromaVectorClient

# FIX: import the vision model so it can be passed into the chunker
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    # Load sensitive environment runtime keys
    load_dotenv(dotenv_path="./config/.env")
    if not os.getenv("GOOGLE_API_KEY"):
        print("[!] Execution Aborted: GOOGLE_API_KEY is missing from config/.env")
        return

    # 1. Path to your sample test document
    # Place any multi-modal PDF containing text, headers, and tables here
    sample_pdf = "./sample_document.pdf" 
    
    if not os.path.exists(sample_pdf):
        print(f"[!] Please drop a test file named 'sample_document.pdf' into: {os.path.abspath('.')}")
        print("[*] Waiting for sample target file...")
        return

    try:
        print("\n=== STARTING SELF-HEALING RAG INGESTION TEST ===\n")

        # 2. Initialise the vision model that will be used to summarise diagrams/images
        # gemini-2.0-flash is multimodal and accepts image_url content blocks
        vision_model = ChatGoogleGenerativeAI(
            model=os.getenv("VISION_MODEL", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # 3. Initialize our adaptive, structural layout parsing chunker
        # FIX: was MultimodalStructuralChunker() — vision_model is a required argument
        chunker = MultimodalStructuralChunker(vision_model=vision_model)
        
        # 4. Process layout elements and synthesize chunks
        finalized_chunks = chunker.process_document(sample_pdf)
        
        print(f"\n[+] Extraction Complete! Generated {len(finalized_chunks)} structural parent chunks.")
        
        # --- Let's inspect a sample chunk to verify layout fidelity ---
        if finalized_chunks:
            print("\n--- Inspecting Sample Chunk Payload Architecture ---")
            sample_index = 0 if len(finalized_chunks) == 1 else 1
            test_chunk = finalized_chunks[sample_index]
            print(f"Metadata Tagging: {test_chunk['metadata']}")
            print("Content Snippet Preview:")
            print("-" * 50)
            # Print first 500 characters to verify markdown and layout alignment
            print(test_chunk['page_content'][:500] + "\n...[Truncated]...")
            print("-" * 50)

        # 5. Initialize our Chroma Vector client wrapper
        chroma_client = ChromaVectorClient()
        
        # 6. Execute vectorization pass and save into local disk structures
        chroma_client.add_documents(finalized_chunks)
        
        print("\n=== INGESTION & SEMANTIC VECTORIZATION TEST SUCCESSFUL ===\n")

    except Exception as e:
        print(f"\n[!] Pipeline Execution Failed. Diagnostic Error Trace: {str(e)}")

if __name__ == "__main__":
    main()