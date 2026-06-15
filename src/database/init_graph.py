import os
import sys

# Add root folder to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.database.graph_client import Neo4jGraphClient

def initialize_schema_constraints():
    client = Neo4jGraphClient()
    
    # Define production-grade unique constraint Cypher queries for each node type
    constraints = [
        "CREATE CONSTRAINT unique_document IF NOT EXISTS FOR (d:Document) REQUIRE d.title IS UNIQUE",
        "CREATE CONSTRAINT unique_module IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT unique_concept IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT unique_algorithm IF NOT EXISTS FOR (a:Algorithm) REQUIRE a.name IS UNIQUE"
    ]
    
    print("\n[*] Injecting structural identity uniqueness constraints into Neo4j database...")
    for constraint in constraints:
        try:
            client.execute_query(constraint)
            print(f"[+] Constraint registered cleanly.")
        except Exception as e:
            print(f"[!] Warning: Constraint configuration skipped/failed: {str(e)}")
            
    # Add an indexing layer over metadata attributes to boost hybrid search speeds later
    print("[*] Creating high-performance performance property indexes...")
    client.execute_query("CREATE INDEX module_type_idx IF NOT EXISTS FOR (m:Module) ON (m.type)")
    print("[+] Structural indexing deployment complete.")
    
    client.close()

if __name__ == "__main__":
    initialize_schema_constraints()
