The demo could search an HNSW graph with a different metric from the one used to build it, and KD-tree cosine search applied an invalid coordinate bound. This update maintains metric-specific graphs, safely traverses the KD-tree for cosine, and adds diverse HNSW neighbor selection and deterministic deletion rebuilds.

It also adds configurable index/retrieval parameters, source-vector snapshots, source-grounded RAG instructions with a deterministic no-context abstention, same-origin frontend requests, configurable Ollama connectivity, Compose setup, reproducible benchmark runners and CI.

Validation: C++ core regressions, HTTP integration tests with mock Ollama, restart recovery and frontend JavaScript syntax checks passed locally. Synthetic evaluations cover 1K/10K/50K vectors, multiple search breadths, Recall@1/5/10, M and construction breadth sweeps; measured results and limitations are in benchmarks/REPORT.md. Docker and Windows builds, real-model quality and GitHub CI have not been verified in this environment.

Trade-offs: three HNSW graphs consume more memory, full snapshots and graph rebuilds add mutation/startup costs, and prompt grounding does not guarantee factual answers. Original JSON parsing and coarse database locks remain limitations.

Implemented with Codex assistance. Upstream engine attribution is preserved; these changes extend the existing project.
