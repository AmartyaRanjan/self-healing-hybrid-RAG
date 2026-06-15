import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

class Neo4jGraphClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password123")
        
        print(f"[*] Initializing connection driver to Neo4j instance at {self.uri}...")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            print("[+] Neo4j Core Driver connected successfully.")
        except Exception as e:
            print(f"[!] Critical Connection Error linking to Neo4j: {str(e)}")
            raise e

    def close(self):
        """Safely close the driver stream connection pool."""
        if hasattr(self, "driver"):
            self.driver.close()
            print("[*] Neo4j Connection Driver safely closed.")

    def execute_query(self, cypher_query: str, parameters: dict = None) -> list:
        """
        Executes an isolation-safe Cypher database transaction query.
        Returns a list of raw transaction records.
        """
        if parameters is None:
            parameters = {}
            
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query, parameters)
                return [record for record in result]
            except Exception as e:
                print(f"[!] Cypher execution transaction failed: {str(e)}")
                print(f"[!] Faulty Query: {cypher_query}")
                return []
                
    def clear_database(self):
        """Wipes the entire database graph canvas clean (Useful for re-indexing iterations)."""
        print("[!] Warning: Purging entire Neo4j database graph space...")
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        print("[+] Database wiped cleanly.")
