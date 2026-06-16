import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Expand python runtime lookups
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.database.chroma_client import ChromaVectorClient
from src.ingestion.chunker import MultimodalStructuralChunker  # Points to the chunker code you shared

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/.env")
load_dotenv(dotenv_path=os.path.normpath(dotenv_path))

class DocumentIngestionPipeline:
    def __init__(self):
        print("================================================================")
        print("[*] INITIALIZING ENTERPRISE MULTIMODAL INGESTION PIPELINE...")
        print("================================================================")
        
        # Instantiate Gemini 2.5 Flash as the Vision LLM for parsing image diagrams
        self.vision_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Connect to our active chunker and our vector client server over port 8001
        self.chunker = MultimodalStructuralChunker(vision_model=self.vision_model)
        self.chroma_client = ChromaVectorClient()

    def ingest_all_pdfs_from_root(self):
        """
        Scans the project root directory for any newly added PDF assets,
        parses them structure-by-structure, and uploads vectors to Chroma Server.
        """
        # Scan current project root path
        root_dir = Path(".")
        pdf_files = list(root_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("[!] Warning: No raw PDF assets detected inside your project directory folder.")
            return

        print(f"[+] Identified {len(pdf_files)} PDF asset targets awaiting layout vectorization.\n")
        
        total_chunks_processed = 0
        
        for pdf_path in pdf_files:
            print(f"----------------------------------------------------------------")
            print(f"[*] Processing Pipeline Target: {pdf_path.name}")
            print(f"----------------------------------------------------------------")
            
            try:
                # 1. Deconstruct PDF layout structure dynamically via Docling + Vision LLM
                finalized_chunks = self.chunker.process_document(str(pdf_path))
                
                if not finalized_chunks:
                    print(f"[!] Target skip: Chunker returned empty payload arrays for {pdf_path.name}")
                    continue
                
                # 2. Upload structured arrays directly to your live distributed Chroma Server
                print(f"[+] Parsing complete. Streaming chunks to your distributed Chroma container...")
                self.chroma_client.add_documents(finalized_chunks)
                
                total_chunks_processed += len(finalized_chunks)
                print(f"[+] Successfully ingested document asset: {pdf_path.name}")
                
            except Exception as e:
                print(f"[!] Critical structural failure processing document asset {pdf_path.name}: {str(e)}")
                continue

        print("\n================================================================")
        print(f"[+] INGESTION COMPLETE. Indexed {total_chunks_processed} total chunks across cluster volumes.")
        print("================================================================")

if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest_all_pdfs_from_root()
