import os
import sys
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# Expand python runtime lookups
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.database.graph_client import Neo4jGraphClient

load_dotenv(dotenv_path="config/.env")

class KnowledgeGraphMiner:
    def __init__(self):
        print("[*] Initializing Knowledge Graph Mining Layer...")
        # Using gemini-2.5-pro for complex structural extraction tasks
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Direct clean instantiation of the chromadb client to guarantee lookups
        chroma_path = os.path.abspath("./chroma_storage")
        print(f"[*] Binding direct persistent Chroma DB client at: {chroma_path}...")
        self.chroma_raw_client = chromadb.PersistentClient(path=chroma_path)
        
        # FIXED: Updated to match your exact active collection name string
        self.collection_name = "self_healing_rag_docs"
        
        self.graph_client = Neo4jGraphClient()
        
        # Rigorous System Prompt enforcing schema parameters and returning clean JSON
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert knowledge graph extraction engine designed for technical documentation.\n"
                "Your task is to analyze the provided documentation text frame and extract structural entities and their relationships.\n\n"
                "=== SCHEMA EXCLUSIONS ===\n"
                "1. Node Types must be exactly one of: 'Module', 'Concept', 'Algorithm'.\n"
                "   - 'Module': Structural blocks, services, or engines (e.g., 'PrecisionRetrievalEngine').\n"
                "   - 'Concept': Underlying technologies, theories, or frameworks (e.g., 'ChromaDB', 'Cross-Encoders').\n"
                "   - 'Algorithm': Mathematical routines, procedures, or audits (e.g., 'Soft-Thresholding Filter').\n"
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
        """
        Generates and executes parameterized Cypher scripts to inject nodes and edges.
        """
        # Step 1: Ensure the root document anchor node exists
        doc_query = "MERGE (d:Document {title: $doc_title}) RETURN d"
        self.graph_client.execute_query(doc_query, {"doc_title": doc_title})
        
        # Step 2: Inject Entities safely
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
            
            # Auto-link Module blocks straight back to their originating Document layout frame
            if node_type == "Module":
                link_doc_query = """
                MATCH (d:Document {title: $doc_title}), (m:Module {name: $m_name})
                MERGE (d)-[:HAS_STRUCTURE]->(m)
                """
                self.graph_client.execute_query(link_doc_query, {"doc_title": doc_title, "m_name": name})

        # Step 3: Inject Intersecting Edges
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
        """
        Extracts chunks directly from ChromaDB via raw handles, pipes them through Gemini, 
        and updates the Neo4j visualization graph space.
        """
        print(f"\n[*] Pulling raw document layout entries from vector store collection: {self.collection_name}...")
        
        try:
            collection = self.chroma_raw_client.get_collection(name=self.collection_name)
            results = collection.get()
            documents = results.get("documents", [])
        except Exception as e:
            print(f"[!] Critical Error accessing Chroma data store collection: {str(e)}")
            return
        
        if not documents:
            print("[!] Error: Vector store collection matches, but it contains 0 text frames.")
            return

        print(f"[+] Located {len(documents)} context chunks. Initiating relationship extraction passes...")
        
        for idx, chunk in enumerate(documents):
            print(f"[*] Processing Chunk {idx + 1}/{len(documents)}...")
            try:
                response = self.chain.invoke({"text_chunk": chunk})
                cleaned_content = response.content.strip().replace("```json", "").replace("```", "").strip()
                graph_json = json.loads(cleaned_content)
                
                # Commit variables safely to Neo4j
                self.commit_to_neo4j(doc_title, graph_json)
                
            except Exception as e:
                print(f"[!] Target extraction skip on frame {idx + 1}: {str(e)}")
                continue
                
        print("\n[+] Graph extraction complete! All architectural edges mapped to local database instance.")
        self.graph_client.close()

if __name__ == "__main__":
    miner = KnowledgeGraphMiner()
    miner.run_mining_pipeline(doc_title="Self-Healing Hybrid RAG Core Engine Specification")
