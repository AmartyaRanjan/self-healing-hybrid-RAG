import os
import sys
from dotenv import load_dotenv

# Ensure the root project directory is on the system path for local module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph_engine.workflow import app

def main():
    # Load environment keys
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../config/.env")
    load_dotenv(dotenv_path=os.path.normpath(dotenv_path))
    if not os.getenv("GOOGLE_API_KEY"):
        print("[!] Execution Aborted: GOOGLE_API_KEY is missing from config/.env")
        return

    print("\n" + "="*60)
    print("      SELF-HEALING STATE GRAPH AGENT - CONSOLE INTERFACE")
    print("="*60)
    
    # Prompt user for input query
    question = input("\nEnter your query: ").strip()
    if not question:
        print("[!] Query cannot be empty.")
        return

    # Initialize the centralized graph memory state payload
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "steps": []
    }

    try:
        print("\n[*] Invoking Compiled LangGraph State Machine Workflow...")
        
        # Stream the graph updates as it executes nodes sequentially
        # The 'config' argument can be expanded for thread checkpointing / cross-talk memory later
        config = {"configurable": {"thread_id": "console_session_1"}}
        
        # Execute runtime execution
        final_state = app.invoke(initial_state, config=config)
        
        # Print out the deterministic trace path history audited by our state tracking keys
        print("\n" + "="*20 + " AGENT EXECUTION TRACE " + "="*20)
        print(f"Nodes Visited sequentially: {final_state.get('steps', [])}")
        print("="*63)

        # Output the ultimate truth-grounded result
        print("\n" + "="*20 + " FINAL RE-RANKED RESPONSE " + "="*20)
        print(final_state.get("generation", "No generation produced."))
        print("="*66 + "\n")

    except Exception as e:
        print(f"\n[!] Critical Runtime State Error: {str(e)}")

if __name__ == "__main__":
    main()