import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.database.graph_client import Neo4jGraphClient

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/.env")
load_dotenv(dotenv_path=os.path.normpath(dotenv_path))

class KnowledgeGraphMiner:
    def __init__(self):
        print("[*] Initializing Knowledge Graph Mining Layer...")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # FIXED: Swapped local path to distributed server HTTP client container
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", 8000))
        print(f"[*] Binding Knowledge Graph Miner to Distributed Chroma Server at http://{chroma_host}:{chroma_port}...")
        self.chroma_raw_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        
        self.collection_name = "self_healing_rag_docs"
        self.graph_client = Neo4jGraphClient()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert knowledge graph extraction engine designed for technical documentation.\n"
                "Your task is to analyze the provided documentation text frame and extract structural entities and their relationships.\n\n"
                "=== SCHEMA EXCLUSIONS ===\n"
                "1. Node Types must be exactly one of: 'Module', 'Concept', 'Algorithm'.\n"
                "2. Relationship Types must be exactly one of: 'HAS_STRUCTURE', 'DEPENDS_ON', 'IMPLEMENTS', 'INTERACTS_WITH'.\n\n"
                "=== OUTPUT FORMAT ===\n"
                "You must respond strictly with a valid JSON object. Do not include markdown code wrappers (like ```json). "
                "The JSON object must match this exact format:\n"
                "{{\n"
                "  \"entities\": [\n"
                "    {{\"name\": \"EntityName\", \"type\": \"Module|Concept|Algorithm\", \"description\": \"Brief context\"}}\n"
                "  ],\n"
                "  \"relationships\": [\n"
                "    {{\"source\": \"EntityName\", \"target\": \"TargetName\", \"type\": \"DEPENDS_ON|IMPLEMENTS|INTERACTS_WITH\"}}\n"
                "  ]\n"
                "}}\n"
                "Ensure every relationship references entities listed inside the entities array."
            )),
            ("human", "=== DOCUMENTATION TEXT FRAME ===\n{text_chunk}\n\nExtract graph layout now:")
        ])
        self.chain = self.prompt | self.llm

    def commit_to_neo4j(self, doc_title: str, graph_data: Dict[str, Any]):
        doc_query = "MERGE (d:Document {title: $doc_title}) RETURN d"
        self.graph_client.execute_query(doc_query, {"doc_title": doc_title})
        
        for entity in graph_data.get("entities", []):
            name = entity.get("name")
            node_type = entity.get("type", "Concept")
            desc = entity.get("description", "")
            if not name or node_type not in ["Module", "Concept", "Algorithm"]:
                continue
            entity_query = f"""
            MERGE (n:{node_type} {{name: $name}})
            ON CREATE SET n.description = $desc
            ON MATCH SET n.description = $desc
            RETURN n
            """
            self.graph_client.execute_query(entity_query, {"name": name, "desc": desc})
            
            if node_type == "Module":
                link_doc_query = """
                MATCH (d:Document {title: $doc_title}), (m:Module {name: $m_name})
                MERGE (d)-[:HAS_STRUCTURE]->(m)
                """
                self.graph_client.execute_query(link_doc_query, {"doc_title": doc_title, "m_name": name})

        for rel in graph_data.get("relationships", []):
            source = rel.get("source")
            target = rel.get("target")
            rel_type = rel.get("type")
            if not source or not target or rel_type not in ["DEPENDS_ON", "IMPLEMENTS", "INTERACTS_WITH"]:
                continue
            rel_query = f"""
            MATCH (s {{name: $source}}), (t {{name: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            RETURN r
            """
            self.graph_client.execute_query(rel_query, {"source": source, "target": target})

    def run_mining_pipeline(self, doc_title: str):
        print(f"\n[*] Pulling raw document layout entries from server collection: {self.collection_name}...")
        try:
            collection = self.chroma_raw_client.get_collection(name=self.collection_name)
            results = collection.get()
            documents = results.get("documents", [])
        except Exception as e:
            print(f"[!] Critical Error accessing Chroma data store collection: {str(e)}")
            return
        
        if not documents:
            print("[!] Error: Server vector store collection matches, but it contains 0 text frames.")
            return

        print(f"[+] Located {len(documents)} context chunks inside the server. Initiating extraction passes...")
        for idx, chunk in enumerate(documents):
            print(f"[*] Processing Chunk {idx + 1}/{len(documents)}...")
            try:
                response = self.chain.invoke({"text_chunk": chunk})
                cleaned_content = response.content.strip().replace("```json", "").replace("```", "").strip()
                graph_json = json.loads(cleaned_content)
                self.commit_to_neo4j(doc_title, graph_json)
            except Exception as e:
                print(f"[!] Target extraction skip on frame {idx + 1}: {str(e)}")
                continue
                
        print("\n[+] Graph extraction complete! All architectural edges mapped to local database instance.")
        self.graph_client.close()

if __name__ == "__main__":
    miner = KnowledgeGraphMiner()
    miner.run_mining_pipeline(doc_title="Self-Healing Hybrid RAG Core Engine Specification")
