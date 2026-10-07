# Implementation notes

This fork extends the upstream project with Codex-assisted code changes. It retains the original engine and attribution. The supplied source files are the baseline; repository HEAD must be checked before publishing to avoid overwriting newer work.

## Index correctness

- The demo database now maintains three HNSW graphs. Each graph is both built and queried using the same distance metric. Memory and insertion costs rise because vectors and graph connections are stored per index.
- KD-tree coordinate bounds are applied only to known Euclidean and Manhattan functions. Cosine and unrecognized distance functions traverse both branches. This provides exact cosine results but gives up pruning.
- HNSW neighbor selection now keeps directionally diverse candidates before filling remaining slots with nearer candidates. Reciprocal-link pruning uses the same rule. The synthetic varied-norm benchmark checks this graph-selection behavior together with metric consistency.
- Equal-distance KD-tree candidates use `(distance, ID)` ordering, matching the exact baseline.
- Deletion rebuilds HNSW from remaining source vectors in sorted ID order. This preserves valid entry-point metadata and avoids graph disconnections left by simply removing edges. It is deliberately expensive; this implementation suits a small, single-process application.
- Dimensions, finite coordinates, supported metrics, algorithm names and search bounds are checked. The original hand-written JSON field parser remains a limitation.

## Configuration

| Setting | Default | Where |
|---|---|---|
| `HNSW_M` | 16 | Environment; 2–128 |
| `HNSW_EF_CONSTRUCTION` | 200 | Environment; M–10000 |
| `efSearch` | 50 | Query argument for `/search` and `/benchmark`; JSON field for document search/ask; UI |
| `chunkWords` | 250 | `/doc/insert` JSON and UI; 1–10000 |
| `overlapWords` | 30 | `/doc/insert` JSON and UI; 0 to chunkWords−1 |
| `k` | Existing endpoint default | Search query or document JSON; 1–1000 |
| `maxDistance` | 0.7 | Document search/ask JSON and UI; cosine distance 0–2 |
| `DATA_DIR` | data | Source-vector snapshots |
| `OLLAMA_HOST`, `OLLAMA_PORT` | 127.0.0.1, 11434 | Ollama connection |
| `EMBED_MODEL`, `GEN_MODEL` | nomic-embed-text, llama3.2 | Model names |
| `PORT` | 8080 | Server port |

Environment variables must be exported by the shell or deployment platform. `.env.example` is documentation; the app does not automatically load it. The graph view defaults to cosine; `/hnsw-info?metric=euclidean` selects another graph.

## Grounded retrieval

Document insertion embeds all chunks before inserting any, so an embedding-service failure does not leave a partially inserted document. Inserts are not a general transactional batch: a later disk or process failure may still leave an incomplete batch.

Retrieval returns source titles, chunk text, cosine distances and IDs. The generation prompt requests source citations, prohibits outside facts, treats source text as data and asks for abstention if evidence is insufficient. With no retrieved chunks, the server returns a fixed abstention without calling generation. Threshold filtering happens after candidate retrieval.

A retrieved chunk does not prove answerability. Prompt rules do not guarantee citation accuracy, immunity to prompt injection, or absence of hallucination. Real-model answer quality needs evaluation with labeled answerable/unanswerable questions. Automated tests use a mock service and verify the API flow and prompt, not model behavior.

## Persistence

Versioned text snapshots preserve IDs, next ID, strings and full float precision. An insertion/deletion writes a temporary snapshot and replaces the previous file. Startup rejects corrupt/truncated snapshots and rebuilds indexes; a stored empty demo database remains empty rather than reloading examples.

These are full snapshots after each mutation: O(ND) write cost and startup rebuild cost. No WAL, multi-process writer coordination, crash-safe transactions, encryption, compression or embedding-model migration is provided. POSIX replacement is atomic, but full power-loss durability is not guaranteed without directory/file fsync. Persistence errors return a server error; mutations already made in memory are not rolled back. Keep the model unchanged for an existing document store, or create a new DATA_DIR and re-embed documents.

## Deployment

The frontend uses the page origin for the API. Opening HTML directly still uses localhost:8080; serving through the app is preferred. Compose runs app and Ollama separately, downloads models and mounts data volumes. No authentication or per-user isolation is implemented. Deploy on a private network or add access control before public document ingestion.

`make test` runs C++ regression checks and Python HTTP tests. CI runs these checks, JavaScript syntax verification and a benchmark smoke test. Tests and benchmark tooling were run on Linux; Docker orchestration and Windows compilation were not executed here.

AddressSanitizer and UndefinedBehaviorSanitizer core checks passed with leak detection disabled. LeakSanitizer could not run under the execution environment's process restrictions; no claim of leak checking is made.
