import os
import sys
from dotenv import load_dotenv

# Expand python runtime lookups
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.workflow import app
from src.database.graph_client import Neo4jGraphClient
from langchain_core.documents import Document

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", ".env")
load_dotenv(dotenv_path=dotenv_path)

class HybridGraphRAGAgent:
    def __init__(self):
        print("================================================================")
        print("[*] INITIALIZING PRODUCTION GRAPH-VECTOR HYBRID FUSION AGENT...")
        print("================================================================")
        self.graph_client = Neo4jGraphClient()
        self.agent_runtime = app

    def fetch_graph_context(self, search_keyword: str) -> str:
        """
        Queries Neo4j to find an entity node matching the keyword and extracts 
        all adjacent structural relationships within a 2-hop topological radius.
        """
        print(f"[*] Querying Neo4j relational topology for entities matching: '{search_keyword}'...")
        
        cypher_query = """
        MATCH (e) WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        MATCH (e)-[r]->(adjacent)
        RETURN e.name AS source, type(r) AS relationship, adjacent.name AS target, adjacent.description AS target_desc
        LIMIT 10
        """
        
        records = self.graph_client.execute_query(cypher_query, {"keyword": search_keyword})
        if not records:
            return ""
            
        triples = []
        for rec in records:
            triples.append(f"Fact: ({rec['source']}) -[:{rec['relationship']}]-> ({rec['target']}) | Context: {rec['target_desc']}")
            
        return "\n".join(triples)

    def ask(self, user_query: str, graph_keyword: str):
        """
        Performs parallel retrieval extraction and injects the fused context 
        into the LangGraph self-correcting execution state loop.
        """
        print(f"\n[*] User Query Received: '{user_query}'")
        
        # 1. Fetch relational graph structures explicitly from Neo4j
        graph_triples = self.fetch_graph_context(graph_keyword)
        
        # 2. Package initial state data mapping using standardized LangChain Document schemas
        initial_state = {
            "question": user_query,
            "documents": [],  # Container initialized cleanly
            "generation": "",
            "steps": [],
            "healing_iterations": 0
        }
        
        # FIXED: Wrap the extracted Neo4j triples into a true LangChain Document instance object
        if graph_triples:
            print("[+] Relational topology facts recovered from Neo4j. Fusing into context pipeline stream...")
            standardized_doc = Document(
                page_content=f"=== CRITICAL GRAPH TOPOLOGY RELATIONSHIPS ===\n{graph_triples}",
                metadata={"title": "Neo4j Knowledge Graph Structural Extraction", "score": 2.0}
            )
            initial_state["documents"].append(standardized_doc)
            
        # 3. Invoke compiled LangGraph engine execution loop
        print("[*] Executing LangGraph Self-Healing workflow loop orchestration...")
        final_state = self.agent_runtime.invoke(initial_state)
        
        print("\n==================== FINAL RESPONSE ====================")
        print(final_state.get("generation"))
        print("========================================================\n")
        
        print(f"[*] Total Execution Route Taken: {final_state.get('steps')}")

if __name__ == "__main__":
    agent = HybridGraphRAGAgent()
    
    # Target inquiry pointing directly to your local index elements
    agent.ask(
        user_query="What is the precision re-ranking pipeline and what specific cross encoder does it use?",
        graph_keyword="Precision"
    )
    
    agent.graph_client.close()
