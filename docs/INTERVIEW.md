# Explain the contribution accurately

## 30-second overview

“I extended an existing educational C++ vector-search engine with metric-consistent HNSW indexes, reproducible recall/latency evaluation, configurable document retrieval and persistent snapshots. I used Codex to assist with implementation and tests. The upstream project supplies the original engine; my fork focuses on correctness, evaluation and retrieval behavior.”

Before using “I implemented” in an interview, review the changes until you can explain and modify them independently. A commit in your fork does not establish that you personally wrote every line or designed the original engine.

## Questions to practice

1. Why does changing an HNSW search metric without rebuilding its graph risk losing recall?
   Graph edges are selected with a particular distance function. Different metrics can imply different useful neighborhoods. Separate indexes make construction and query metrics consistent.
2. Why disable KD-tree pruning for cosine?
   The raw coordinate difference is not a valid lower bound for cosine distance. Exploring both branches preserves exactness at the cost of full traversal.
3. What does Recall@k measure?
   The fraction of exact brute-force top-k IDs also returned by approximate search. This runner averages that fraction across held-out synthetic queries.
4. What does efSearch change?
   The layer-zero candidate breadth. More breadth can improve recall but increases search work; improvement depends on the dataset and graph.
5. How is the benchmark reproducible?
   A fixed seed creates database vectors and independent query vectors; index seeds and insertion order are fixed. Latency is still machine- and load-dependent.
6. Why store vectors rather than raw graph memory?
   Source snapshots are simpler to validate and version. Rebuilding indexes avoids dependence on internal pointer/layout serialization but increases startup time.
7. How do you prevent hallucinations?
   Retrieval, explicit source instructions and a deterministic no-context fallback reduce unsupported answers. They do not prove correctness when generation runs.
8. Why use a neighbor-diversity heuristic?
   Choosing only the closest nodes can create redundant local links. A candidate is kept when existing selected neighbors do not already cover it more closely than the node being connected; unused slots are then filled. This can improve navigation but is still approximate.
9. What are the biggest trade-offs?
   Three metric indexes use more memory; full snapshots and deletion rebuilds cost time; coarse database locks serialize operations; JSON parsing and production access control still need work.

See benchmarks/REPORT.md for measured results. Do not describe synthetic 16D results as evidence for 768D semantic document quality or production throughput.
