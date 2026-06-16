"""
Self-Healing Hybrid RAG — FastAPI Backend
==========================================
Endpoints:
  GET  /                   — Serves the frontend UI (frontend/index.html)
  POST /ingest             — Upload a PDF; background job ingests into Chroma + Neo4j graph
  GET  /ingest/status/{id} — Poll ingestion job status
  GET  /ingest/jobs        — List all ingestion jobs
  POST /ask                — Stream Q&A as Server-Sent Events (SSE), one event per pipeline step
  GET  /health             — Health / readiness check
  GET  /documents          — List documents currently in Chroma
"""

import os
import sys
import uuid
import asyncio
import json
import time
import shutil
import threading
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import contextmanager
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# ─── Bootstrap ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", ".env")
load_dotenv(dotenv_path=dotenv_path)

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Self-Healing Hybrid RAG API",
    description="FastAPI backend for the Self-Healing Hybrid RAG system with PDF ingestion and LangGraph-powered Q&A",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Serve Frontend Static Files ─────────────────────────────────────────────
FRONTEND_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the main frontend UI."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return JSONResponse({"error": "Frontend not found. Ensure frontend/index.html exists."}, status_code=404)

# ─── In-Memory Job Store ──────────────────────────────────────────────────────
# job_id -> { status, filename, steps, error, chunks, start_time, end_time }
INGESTION_JOBS: Dict[str, Dict[str, Any]] = {}
UPLOADS_DIR = Path("./uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

# ─── Request / Response Models ────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    graph_keyword: Optional[str] = ""

class JobStatusResponse(BaseModel):
    job_id: str
    status: str   # pending | running | chroma_done | graph_done | done | error
    filename: str
    steps: list
    chunks_indexed: int
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


# ─── Thread-Safe Log Capture ────────────────────────────────────────────────
class ThreadLocalStdout:
    """
    A process-wide proxy for sys.stdout that delegates stdout writing to
    thread-local capture configurations if registered.
    """
    def __init__(self, original_stdout):
        self._original = original_stdout
        self._local = threading.local()

    def register_capture(self, capture):
        self._local.current_capture = capture

    def unregister_capture(self):
        if hasattr(self._local, "current_capture"):
            del self._local.current_capture

    def write(self, text: str):
        capture = getattr(self._local, "current_capture", None)
        if capture:
            capture.write_captured(text)
        self._original.write(text)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)

sys_stdout_proxy = ThreadLocalStdout(sys.stdout)
sys.stdout = sys_stdout_proxy


class StepCapture:
    """
    Thread-safe context manager that registers itself with the global ThreadLocalStdout proxy
    to capture print statements executed on the current thread.
    """
    def __init__(self, queue: Optional[asyncio.Queue] = None, job_steps: Optional[list] = None, loop=None):
        self.queue = queue
        self.job_steps = job_steps
        self.loop = loop  # pre-captured event loop for thread-safe puts

    def __enter__(self):
        sys_stdout_proxy.register_capture(self)
        return self

    def write_captured(self, text: str):
        if text.strip():
            if self.job_steps is not None:
                self.job_steps.append(text.strip())
            if self.queue is not None and self.loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.queue.put({"type": "log", "message": text.strip()}),
                        self.loop
                    )
                except Exception:
                    pass

    def __exit__(self, *args):
        sys_stdout_proxy.unregister_capture()


# ─── Lazy Singleton for the Hybrid Agent ─────────────────────────────────────
_agent = None
_agent_lock = threading.Lock()

def get_agent():
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                from ask_hybrid_agent import HybridGraphRAGAgent
                _agent = HybridGraphRAGAgent()
    return _agent


# ─── Background Ingestion Task ────────────────────────────────────────────────
def run_ingestion(job_id: str, pdf_path: str, filename: str):
    """
    Full ingestion pipeline:
      1. Chunk the PDF via MultimodalStructuralChunker
      2. Upload chunks to Chroma
      3. Mine entities/relationships into Neo4j
    """
    job = INGESTION_JOBS[job_id]
    job["status"] = "running"
    job["start_time"] = datetime.utcnow().isoformat()

    try:
        # ── Phase 1: Chroma Ingestion ──────────────────────────────────────
        job["status"] = "chunking"
        job["steps"].append("[PHASE 1] Starting PDF parsing and Chroma ingestion...")

        from langchain_google_genai import ChatGoogleGenerativeAI
        from src.ingestion.chunker import MultimodalStructuralChunker
        from src.database.chroma_client import ChromaVectorClient

        vision_model = ChatGoogleGenerativeAI(
            model=os.getenv("VISION_MODEL", "gemini-2.5-flash"),
            temperature=0.0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        chunker = MultimodalStructuralChunker(vision_model=vision_model)
        chroma_client = ChromaVectorClient()

        with StepCapture(job_steps=job["steps"]):
            chunks = chunker.process_document(pdf_path)

        if not chunks:
            raise ValueError("Chunker returned 0 chunks — the PDF may be empty or unreadable.")

        job["steps"].append(f"[PHASE 1] Chunking complete: {len(chunks)} structural chunks extracted.")
        job["chunks_indexed"] += len(chunks)

        with StepCapture(job_steps=job["steps"]):
            chroma_client.add_documents(chunks)

        job["status"] = "chroma_done"
        job["steps"].append("[PHASE 1] ✓ Chroma vector index updated successfully.")

        # ── Phase 2: Knowledge Graph Mining ───────────────────────────────
        job["status"] = "graph_mining"
        job["steps"].append("[PHASE 2] Starting Neo4j knowledge graph mining...")

        from src.database.extract_to_graph import KnowledgeGraphMiner
        doc_title = Path(filename).stem

        miner = KnowledgeGraphMiner()
        with StepCapture(job_steps=job["steps"]):
            miner.run_mining_pipeline(doc_title=doc_title)

        job["status"] = "graph_done"
        job["steps"].append("[PHASE 2] ✓ Knowledge graph entities and relationships committed to Neo4j.")

        job["status"] = "done"
        job["steps"].append("✓ Full ingestion pipeline complete.")

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["steps"].append(f"[ERROR] {str(exc)}")

    finally:
        job["end_time"] = datetime.utcnow().isoformat()
        # Clean up uploaded temp file
        try:
            os.remove(pdf_path)
        except Exception:
            pass


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/ingest")
async def ingest_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a PDF to ingest into Chroma + Neo4j."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    job_id = str(uuid.uuid4())
    safe_name = f"{job_id}_{file.filename}"
    save_path = UPLOADS_DIR / safe_name

    # Save upload to disk
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    INGESTION_JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "filename": file.filename,
        "steps": [f"Received PDF: {file.filename}"],
        "chunks_indexed": 0,
        "error": None,
        "start_time": None,
        "end_time": None,
    }

    background_tasks.add_task(run_ingestion, job_id, str(save_path), file.filename)

    return {"job_id": job_id, "filename": file.filename, "status": "pending"}


@app.get("/ingest/status/{job_id}", response_model=JobStatusResponse)
def ingest_status(job_id: str):
    if job_id not in INGESTION_JOBS:
        raise HTTPException(status_code=404, detail="Job not found.")
    return INGESTION_JOBS[job_id]


@app.get("/ingest/jobs")
def list_jobs():
    return list(INGESTION_JOBS.values())


@app.get("/documents")
def list_documents():
    """Return list of documents/chunks currently in Chroma."""
    try:
        import chromadb
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", 8000))
        client = chromadb.HttpClient(host=host, port=port)
        collection = client.get_collection("self_healing_rag_docs")
        results = collection.get()
        metadatas = results.get("metadatas", [])
        titles = list({m.get("title", "Unknown") for m in metadatas if m})
        return {"count": len(metadatas), "titles": sorted(titles)}
    except Exception as e:
        return {"count": 0, "titles": [], "error": str(e)}


# ─── Streaming Q&A via SSE ────────────────────────────────────────────────────

async def rag_event_stream(question: str, graph_keyword: str) -> AsyncGenerator[str, None]:
    """
    Runs the hybrid RAG agent and emits Server-Sent Events for:
      - each workflow node activation
      - the final answer
      - any errors
    """
    queue: asyncio.Queue = asyncio.Queue()
    # Capture the running loop HERE (in async context) before spawning any threads.
    # asyncio.get_event_loop() inside a thread raises RuntimeError — use this instead.
    loop = asyncio.get_running_loop()

    # Node-to-label mapping for the frontend pipeline visualizer
    NODE_LABELS = {
        "decompose_query":   "Query Decomposition",
        "hyde_optimization": "HyDE Optimization",
        "transform_query":   "Query Transformation",
        "retrieve_docs":     "Hybrid Retrieval",
        "generate_response": "Response Generation",
        "fallback_web_search": "Web Search Fallback",
        "grade":             "Grounding Audit",
    }

    def emit(event_type: str, data: Any):
        """Thread-safe: put an SSE-formatted string onto the queue using the captured loop."""
        payload = json.dumps({"type": event_type, "data": data})
        asyncio.run_coroutine_threadsafe(queue.put(payload), loop)

    def _run_agent():
        """Blocking call — runs inside a thread."""
        try:
            # Patch workflow nodes to emit events
            import src.graph_engine.workflow as wf

            original_funcs = {}

            def make_wrapper(node_name, original_fn):
                def wrapped(state):
                    emit("node_start", {
                        "node": node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                        "timestamp": time.time(),
                    })
                    result = original_fn(state)
                    emit("node_end", {
                        "node": node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                        "timestamp": time.time(),
                    })
                    return result
                return wrapped

            # Wrap each node
            node_fn_map = {
                "decompose_query":   wf.decompose_query_node,
                "hyde_optimization": wf.hyde_node,
                "transform_query":   wf.transform_query_node,
                "retrieve_docs":     wf.retrieve_node,
                "generate_response": wf.generate_node,
                "fallback_web_search": wf.fallback_search_node,
            }
            for name, fn in node_fn_map.items():
                original_funcs[name] = fn
                setattr(wf, f"{name}_node" if name != "decompose_query" else "decompose_query_node", make_wrapper(name, fn))

            # Rebuild app with patched nodes
            from langgraph.graph import StateGraph, START, END
            from src.graph_engine.state import RAGState
            from src.graph_engine.edges import decide_to_generate, grade_generation_v_documents

            patched_workflow = StateGraph(RAGState)
            patched_workflow.add_node("decompose_query",    make_wrapper("decompose_query",    wf.decompose_query_node))
            patched_workflow.add_node("hyde_optimization",  make_wrapper("hyde_optimization",  wf.hyde_node))
            patched_workflow.add_node("transform_query",    make_wrapper("transform_query",    wf.transform_query_node))
            patched_workflow.add_node("retrieve_docs",      make_wrapper("retrieve_docs",      wf.retrieve_node))
            patched_workflow.add_node("generate_response",  make_wrapper("generate_response",  wf.generate_node))
            patched_workflow.add_node("fallback_web_search",make_wrapper("fallback_web_search",wf.fallback_search_node))

            patched_workflow.add_edge(START, "decompose_query")
            patched_workflow.add_edge("decompose_query", "hyde_optimization")
            patched_workflow.add_edge("hyde_optimization", "transform_query")
            patched_workflow.add_edge("transform_query", "retrieve_docs")
            patched_workflow.add_conditional_edges(
                "retrieve_docs", decide_to_generate,
                {"generate": "generate_response", "web_search": "fallback_web_search"}
            )
            patched_workflow.add_edge("fallback_web_search", "generate_response")

            # Wrap grade edge too
            def grading_wrapper(state):
                emit("node_start", {"node": "grade", "label": "Grounding Audit", "timestamp": time.time()})
                result = grade_generation_v_documents(state)
                emit("node_end", {"node": "grade", "label": "Grounding Audit",
                                  "result": result, "timestamp": time.time()})
                return result

            patched_workflow.add_conditional_edges(
                "generate_response", grading_wrapper,
                {"useful": END, "not useful": "retrieve_docs"}
            )
            patched_app = patched_workflow.compile()

            # Fetch graph context from Neo4j
            emit("status", {"message": "Fetching Neo4j graph context...", "phase": "graph"})
            agent = get_agent()
            graph_triples = agent.fetch_graph_context(graph_keyword) if graph_keyword else ""

            from langchain_core.documents import Document
            initial_state = {
                "question": question,
                "documents": [],
                "generation": "",
                "steps": []
            }
            if graph_triples:
                initial_state["documents"].append(Document(
                    page_content=f"=== GRAPH TOPOLOGY ===\n{graph_triples}",
                    metadata={"title": "Neo4j Knowledge Graph", "score": 2.0}
                ))

            emit("status", {"message": "Invoking LangGraph workflow...", "phase": "workflow"})

            # Capture prints — pass the pre-captured loop so thread-safe puts work
            captured_logs = []
            with StepCapture(queue=queue, job_steps=captured_logs, loop=loop):
                final_state = patched_app.invoke(initial_state)

            answer = final_state.get("generation", "")
            steps = final_state.get("steps", [])

            emit("answer", {
                "text": answer,
                "steps": steps,
                "logs": captured_logs[:50],  # cap log size
            })

        except Exception as exc:
            emit("error", {"message": str(exc)})
        finally:
            # Sentinel to close the stream — use pre-captured loop, not get_event_loop()
            asyncio.run_coroutine_threadsafe(queue.put("__DONE__"), loop)

    # Run blocking agent in thread pool using the already-captured loop
    loop.run_in_executor(None, _run_agent)

    # Yield SSE events as they arrive
    while True:
        item = await queue.get()
        if item == "__DONE__":
            yield "data: {\"type\": \"done\"}\n\n"
            break
        yield f"data: {item}\n\n"


@app.post("/ask")
async def ask_question(req: AskRequest):
    """
    Ask a question. Response is a text/event-stream of Server-Sent Events.
    Each event is a JSON object with a `type` field:
      - status     { message, phase }
      - node_start { node, label, timestamp }
      - node_end   { node, label, timestamp, result? }
      - log        { message }
      - answer     { text, steps, logs }
      - error      { message }
      - done       (stream terminator)
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        rag_event_stream(req.question.strip(), req.graph_keyword or ""),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Neo4j Graph Explorer ─────────────────────────────────────────────────────
@app.get("/graph")
async def get_graph(search: Optional[str] = None, limit: int = 80):
    """
    Query Neo4j and return nodes + relationships for the frontend graph visualizer.
    Optional `search` param filters by node name (case-insensitive substring).
    """
    try:
        from src.database.graph_client import Neo4jGraphClient
        client = Neo4jGraphClient()

        if search and search.strip():
            node_query = (
                "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($search) "
                "RETURN id(n) AS id, labels(n) AS labels, n.name AS name, "
                "n.description AS description LIMIT $limit"
            )
            rel_query = (
                "MATCH (a)-[r]->(b) "
                "WHERE toLower(a.name) CONTAINS toLower($search) "
                "   OR toLower(b.name) CONTAINS toLower($search) "
                "RETURN id(a) AS source, id(b) AS target, type(r) AS type, "
                "r.description AS description LIMIT $limit"
            )
            params = {"search": search.strip(), "limit": limit}
        else:
            node_query = (
                "MATCH (n) RETURN id(n) AS id, labels(n) AS labels, "
                "n.name AS name, n.description AS description LIMIT $limit"
            )
            rel_query = (
                "MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target, "
                "type(r) AS type, r.description AS description LIMIT $limit"
            )
            params = {"limit": limit}

        raw_nodes = client.execute_query(node_query, params)
        raw_rels  = client.execute_query(rel_query, params)
        client.close()

        nodes = [
            {
                "id": str(rec["id"]),
                "label": (rec["labels"] or ["Node"])[0],
                "name": rec["name"] or f"Node {rec['id']}",
                "description": rec["description"] or "",
            }
            for rec in raw_nodes
        ]

        # Build set of node ids for edge filtering
        node_ids = {n["id"] for n in nodes}
        edges = [
            {
                "source": str(rec["source"]),
                "target": str(rec["target"]),
                "type": rec["type"] or "RELATED_TO",
                "description": rec["description"] or "",
            }
            for rec in raw_rels
            if str(rec["source"]) in node_ids and str(rec["target"]) in node_ids
        ]

        return JSONResponse({
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        })

    except Exception as exc:
        return JSONResponse(
            {"nodes": [], "edges": [], "error": str(exc), "total_nodes": 0, "total_edges": 0},
            status_code=200  # Return 200 so frontend can display the error gracefully
        )
