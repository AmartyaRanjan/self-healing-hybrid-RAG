import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.nodes import PrecisionRetrievalEngine

def main():
    load_dotenv(dotenv_path="./config/.env")
    
    # Define a test question relevant to the sample_document.pdf you ingested earlier
    test_query = "What is the pipeline architecture for our system?"
    
    try:
        print("\n=== STARTING PRECISION RETRIEVAL SYSTEM TEST ===\n")
        
        # Initialize the retrieval pipeline engine
        engine = PrecisionRetrievalEngine()
        
        # Execute the multi-stage extraction pass
        final_context_frames = engine.retrieve_and_rerank(test_query)
        
        print("\n--- Inspecting Top Re-Ranked Output Context Frames ---")
        for idx, frame in enumerate(final_context_frames):
            print(f"\n[Rank {idx + 1}] (Cross-Encoder Score: {frame['score']:.4f})")
            print(f"Source Heading: {frame['metadata'].get('title', 'Unknown')}")
            print(f"Chunk ID: {frame['metadata'].get('chunk_id', 'Unknown')}")
            print("-" * 40)
            # Preview first 200 characters of each top frame
            print(frame['page_content'][:200] + " ... [Truncated]")
            print("-" * 40)
            
        print("\n=== RETRIEVAL & PRECISION RE-RANKING SUCCESSFUL ===\n")
        
    except Exception as e:
        print(f"\n[!] Retrieval system test failed. Error diagnostics: {str(e)}")

if __name__ == "__main__":
    main()