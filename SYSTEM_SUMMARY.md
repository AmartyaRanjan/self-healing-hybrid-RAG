# Self-Healing Hybrid RAG System — Architectural Summary

## 1) High-level concept

This repository implements a **hybrid retrieval augmented generation (Hybrid RAG)** system that:
- Ingests documents by **structural parsing** (headers/sections, tables, and diagrams/images).
- Stores text chunks in a **Chroma vector database** using **Gemini embeddings**.
- Optionally mines structured entities/relationships from the stored chunks into a **Neo4j knowledge graph**.
- At query time, runs a **self-healing LangGraph workflow**:
  - Query decomposition + HyDE-style hypothetical answer generation
  - Query transformation for vector retrieval
  - **Asynchronous vector retrieval**
  - **Cross-encoder re-ranking**
  - **Keyword/BM25 lexical “highway” recall**
  - **Context fusion**
  - Deterministic response generation via Gemini
  - **Grounding guardrail** that grades whether the answer is supported by retrieved context
  - If grounding/viability is insufficient, it falls back to **live web search**

The overall design is intentionally modular and “self-correcting” at runtime via:
- Context scoring / routing (generate vs web_search)
- Answer grounding auditing (useful vs not useful)
- Re-entry into retrieval when the answer fails grounding

---

## 2) Major components

### A. Ingestion & chunking (multimodal structural chunker)
**File:** `src/ingestion/chunker.py`

**Core class:** `MultimodalStructuralChunker`

**What it does**
1. Uses **Docling** to convert and iterate through a PDF document structure.
2. Splits content into chunks anchored by **section headers**.
3. Collects:
   - `TextItem` → appended to the chunk’s `text_buffer`
   - `TableItem` → exported to markdown and appended to chunk’s `tables`
   - `PictureItem` (diagrams/images) → handled via multimodal summarization:
     - Converts image to base64
     - Sends **(text prompt + image_url)** to a provided vision-capable LangChain chat model
     - Appends the produced visual analysis into `text_buffer`

**Chunk output schema**
`process_document(pdf_path)` returns a list of dictionaries:
- `page_content`: combined structured text
  - Includes `Section Title: <title>` + text_buffer + tables markdown
- `metadata`:
  - `title`: section heading text
  - `chunk_id`: `doc_chunk_{idx}`

**Data shape example**
```json
{
  "page_content": "Section Title: Introduction\n\n<text...>\n\n<table markdown...>",
  "metadata": { "title": "Introduction", "chunk_id": "doc_chunk_0" }
}
```

**Test entrypoint**
**File:** `src/test_ingestion.py`
- Builds a vision model (Gemini multimodal)
- Runs the chunker on `./sample_document.pdf`
- Pushes chunks to Chroma

---

### B. Vector indexing & retrieval (Chroma + Gemini embeddings)
**File:** `src/database/chroma_client.py`

**Core class:** `ChromaVectorClient`

**What it does**
- Connects to a **distributed Chroma server** via HTTP:
  - `CHROMA_HOST` (default `localhost`)
  - `CHROMA_PORT` (default `8000`)
- Creates embeddings using `GoogleGenerativeAIEmbeddings`:
  - `EMBEDDING_MODEL` (default `gemini-embedding-2-preview`)
- Uses LangChain’s `Chroma` wrapper and a configured collection name:
  - default `self_healing_rag_docs`

**Indexing**
- `add_documents(finalized_chunks)`:
  - Embeds chunk `page_content`
  - Stores vectors with `metadatas` and `ids`
  - IDs are taken from `metadata["chunk_id"]`

**Retrieval**
- `get_retriever(search_kwargs={"k": 5})` returns a LangChain retriever.

---

### C. Knowledge graph mining (Neo4j)
This is an optional but integrated layer: chunks can be mined into nodes/edges.

#### 1) Neo4j schema constraints
**File:** `src/database/init_graph.py`

- Establishes Neo4j uniqueness constraints for:
  - `:Document.title`
  - `:Module.name`
  - `:Concept.name`
  - `:Algorithm.name`
- Adds an index:
  - `:Module(type)` → `module_type_idx`

#### 2) Neo4j client
**File:** `src/database/graph_client.py`

**Core class:** `Neo4jGraphClient`
- Creates a Neo4j driver using env vars:
  - `NEO4J_URI` (default `neo4j://localhost:7687`)
  - `NEO4J_USERNAME` (default `neo4j`)
  - `NEO4J_PASSWORD` (default `password123`)
- `execute_query(cypher_query, parameters)` runs parameterized Cypher.
- `clear_database()` purges all nodes/edges.

#### 3) Mining pipeline
**File:** `src/database/extract_to_graph.py`

**Core class:** `KnowledgeGraphMiner`

**What it does**
- Reads all stored text frames from a **persistent Chroma** directory:
  - `./chroma_storage` (hard-coded in code)
- For each chunk, uses **Gemini** (`gemini-2.5-pro`) to extract:
  - Entities of types: `Module | Concept | Algorithm`
  - Relationships of types: `DEPENDS_ON | IMPLEMENTS | INTERACTS_WITH`
- Enforces strict JSON output format via prompt schema.
- Commits results into Neo4j:
  - Ensures a `:Document {title: doc_title}` node exists
  - `MERGE` inserts nodes and sets descriptions
  - Links `:Document -[:HAS_STRUCTURE]-> :Module`
  - Creates typed relationships between entities by matching `{name}`

> Note: the mining layer is separate from the query-time hybrid retrieval; the system can incorporate graph facts in `ask_hybrid_agent.py`.

---

### D. Query-time hybrid retrieval + generation (LangGraph self-healing workflow)

#### 1) LangGraph state schema
**File:** `src/graph_engine/state.py`

`RAGState` includes:
- `question: str`
- `documents: List[Dict[str, Any]]`
- `generation: str`
- `steps: List[str]`

The workflow mutates this state as it moves between nodes.

#### 2) Runtime entrypoints
- **Console runner:** `src/graph_engine/run_graph_agent.py`
  - Prompts for `question`
  - Builds initial state
  - Invokes compiled LangGraph app

- **“Production” hybrid fusion agent:** `ask_hybrid_agent.py`
  - Initializes `Neo4jGraphClient`
  - Calls `fetch_graph_context(graph_keyword)` using a Cypher query:
    - Finds `(:Entity)` nodes where `name CONTAINS keyword` OR `description CONTAINS keyword`
    - Collects 1-hop outgoing relationships:
      - Returns `(source, relationship, adjacent.target, adjacent.description)`
  - Wraps Neo4j triples into a LangChain `Document`
  - Injects it into `initial_state["documents"]`
  - Invokes the LangGraph workflow via `app.invoke(initial_state)`

#### 3) Workflow orchestration (the self-healing loop)
**File:** `src/graph_engine/workflow.py`

This file compiles the LangGraph state machine:

**Nodes**
1. `decompose_query`  
   - Calls `QueryDecompositionEngine.decompose(question)`
   - (Currently the decomposition result is not directly used later; it mainly records a step)

2. `hyde_optimization`  
   - Calls `HyDEGenerator.generate_hypothetical_document(question)`
   - Produces a “document frame” but does not directly affect retrieval in the current pipeline (it returns it as `documents`)

3. `transform_query`  
   - Calls `QueryTransformEngine.transform(raw_question)`
   - Updates `state["question"]` with an optimized query

4. `retrieve_docs` (main hybrid retrieval + fusion)
   - Runs asynchronous vector retrieval:
     - `retrieval_engine.async_vector_retrieve(question)`
   - If vector context exists and BM25 index is uninitialized:
     - `lexical_highway.initialize_index(chroma_context_frames)`
   - Runs BM25 keyword search:
     - `lexical_highway.search(question, top_n=2)`
   - Runs cross-encoder re-ranking:
     - `retrieval_engine.rerank_cache(question, chroma_context_frames)`
   - Fuses context:
     - `existing_documents + ranked_chroma + lexical_hits`
   - Sorts by `metadata["score"]` descending
   - Keeps top 4
   - Converts each document frame into a markdown wrapper via `harmonize_context_to_markdown`

5. `generate_response`
   - Calls `ContextualGenerationEngine.generate_response(question, documents)`
   - Uses Gemini with the provided reference context

6. `fallback_web_search`
   - Calls `DynamicWebSearchEngine.search(question)`
   - Produces web result document frame(s) with a high score (10.0)

**Edges / routing logic**
- Start → `decompose_query` → `hyde_optimization` → `transform_query` → `retrieve_docs`
- After `retrieve_docs`, conditional:
  - `decide_to_generate(state)` → choose:
    - `generate_response` OR `fallback_web_search`
- After `generate_response`, conditional:
  - `grade_generation_v_documents(state)`:
    - `useful` → END
    - `not useful` → back to `retrieve_docs`

This yields a **self-healing loop**: if generation is not grounded, the system retrieves again (potentially including updated context).

#### 4) Retrieval + ranking + generation engines
**File:** `src/graph_engine/nodes.py`

Key engines:

**(i) Query Decomposition**
- `QueryDecompositionEngine`
- Uses Gemini to output a JSON array of 2–3 sub-queries.
- Parsing is strict JSON (with fallback to raw query).

**(ii) HyDE Optimization**
- `HyDEGenerator`
- Uses Gemini to create a hypothetical ideal passage to improve retrieval semantics.

**(iii) Query Transformation**
- `QueryTransformEngine`
- Uses Gemini to output:
  - `optimized_query`
  - `reasoning`
- Updates `state["question"]` with optimized query.

**(iv) Precision Retrieval (Vector + Cross-Encoder)**
- `PrecisionRetrievalEngine`
  - Uses `ChromaVectorClient.get_retriever(k=15)`
  - Asynchronous execution via `asyncio.get_event_loop().run_in_executor`
  - Cross-Encoder reranker:
    - HuggingFace model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `rerank_cache`:
    - scores each pair `[question, doc.page_content]`
    - stores results in `doc.metadata["score"]`

**(v) Lexical Highway (BM25)**
- `LexicalHighwayEngine`
  - Builds BM25 index from retrieved docs’ `page_content` tokens
  - `search(query, top_n=2)`:
    - returns docs with overlap-based scoring
    - writes `doc.metadata["score"]` scaled from BM25 score (`score - 5.0`)

**(vi) Live Web Search fallback**
- `DynamicWebSearchEngine`
  - Uses DuckDuckGo via LangChain community tools
  - Returns a single `Document` containing `str(raw_web_results)`
  - Assigns `metadata.score = 10.0`

**(vii) Contextual Generation**
- `ContextualGenerationEngine`
  - Uses Gemini chat model
  - Prompt includes:
    - system instruction to answer using provided context
    - reference context inserted into `{context}`
  - Output is plain text.

#### 5) Graph edges / guardrails
**File:** `src/graph_engine/edges.py`

**(i) `decide_to_generate`**
- If `documents` is empty → route to `web_search`
- Otherwise evaluates the top doc’s `metadata.score` (supports both dict-typed docs and LangChain `Document` objects)
- Uses a score floor:
  - `CRITICAL_SCORE_FLOOR = -3.0`
- If best score < floor → route to `web_search`

**(ii) `grade_generation_v_documents`**
- Guardrail designed as an **anti-hallucination / grounding audit**
- Sends:
  - reference context (`context_text`) assembled from retrieved documents
  - the generated answer
- Gemini returns JSON:
  - `{ "score": "yes" | "no" }`
- If yes → `useful`
- If no → `not useful` and the system loops back to `retrieve_docs`

---

## 3) End-to-end execution flow (query time)

### Step 0 — (Optional) graph-fusion bootstrap
In `ask_hybrid_agent.py`:
1. `fetch_graph_context(graph_keyword)` queries Neo4j for topology facts.
2. Wraps facts into a `Document` placed in `initial_state["documents"]`.

### Step 1 — LangGraph workflow execution
In `src/graph_engine/workflow.py`:
1. `decompose_query`
2. `hyde_optimization`
3. `transform_query` (updates the question)
4. `retrieve_docs`
   - vector retrieval (async, k=15)
   - rerank with cross-encoder
   - lexical recall with BM25 over retrieved frames
   - fuse and pick top 4 contexts
   - harmonize to markdown wrappers
5. `generate_response` (Gemini with context)
6. `grade_generation_v_documents`
   - if grounded → END
   - else → loop to `retrieve_docs`
7. If initial context is weak → `fallback_web_search`

---

## 4) Data schemas & key metadata fields

### Document frames (LangChain `Document`)
During retrieval/fusion, each context item is a `Document` with:
- `page_content`:
  - either text chunk content or a markdown-wrapped context block
- `metadata`:
  - `title`: source label for provenance
  - `score`: retrieval confidence / rerank score
  - `chunk_id`: typically present from ingestion into Chroma

### Neo4j mined structures
- Nodes:
  - `:Document {title}`
  - `:Module {name, description}`
  - `:Concept {name, description}`
  - `:Algorithm {name, description}`
- Relationships:
  - `(:Document)-[:HAS_STRUCTURE]->(:Module)`
  - `(:Module|Concept|Algorithm)-[:DEPENDS_ON|IMPLEMENTS|INTERACTS_WITH]->(...)`

---

## 5) Environment configuration (from `config/.env`)

The code references these environment variables:

### LLM / embeddings
- `GOOGLE_API_KEY` (required for Gemini / embeddings)
- `GENERATION_MODEL` (default `gemini-2.5-flash`)
- `VISION_MODEL
