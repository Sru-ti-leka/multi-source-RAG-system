#  TaskFlow Lite -- Multi-Source RAG for Technical Support

A Retrieval-Augmented Generation (RAG) system that answers customer
support questions about a fictional software product (TaskFlow Lite)
using multiple knowledge sources:

-    Official Documentation
-    Customer Forums
-    Technical Blogs

This project demonstrates multi-source retrieval, reranking,
contradiction detection, and logging.

------------------------------------------------------------------------

##  Features

-    Multi-source semantic retrieval (Docs, Blogs, Forums)
-    Source-aware chunking strategy
-    FAISS vector search indexes (3 separate indexes)
-    Cross-encoder reranking
-    Numeric contradiction detection (MB, days, rate limits)
-    Source reliability policy (Docs \> Blogs \> Forums)
-    Structured logging of each query
-    Performance analysis (Hit@5, Hit@1)

------------------------------------------------------------------------

##  Architecture Overview

Query → Bi-Encoder Retrieval (FAISS per source)\
→ Source Weighting\
→ Cross-Encoder Reranking\
→ Contradiction Detection\
→ Source Reliability Resolution\
→ Final Answer Synthesis\
→ Logging

------------------------------------------------------------------------

## 📂 Project Structure

    TaskFlow-Project/
    │
    ├── data/
    │   ├── docs/
    │   ├── blogs/
    │   └── forums/
    │
    ├── indexes/              # Generated FAISS indexes
    ├── logs/                 # Query logs
    ├── src/
    │   ├── build_index.py    # Builds vector indexes
    │   └── rag_final.py      # Full RAG system
    │
    └── report.pdf            # Final report 

------------------------------------------------------------------------

##  Setup Instructions

###  Create Environment

``` bash
python -m venv dl_env
source dl_env/bin/activate  # or dl_env\Scripts\activate on Windows
```

###  Install Dependencies

``` bash
pip install sentence-transformers faiss-cpu transformers torch numpy
```

------------------------------------------------------------------------

##  Build Indexes

Run:

``` bash
python src/build_index.py
```

This creates: - docs.faiss - blogs.faiss - forums.faiss

------------------------------------------------------------------------

##  Run the RAG System

``` bash
python src/rag_final.py
```

Then enter a support question such as:

-   What is the maximum attachment size?
-   How long do API keys last?
-   What is the API rate limit?

The system will: - Retrieve evidence from all sources - Rerank results -
Detect contradictions - Apply reliability policy - Log the query

------------------------------------------------------------------------

##  Performance Evaluation

Metrics included: - Hit@5 - Hit@1

Reranking improves overall retrieval robustness while authoritative
selection ensures reliability.

------------------------------------------------------------------------

##  Contradiction Handling

The system detects conflicting numeric values across sources:

Examples: - Attachment size (25MB vs 50MB) - API key expiry (30 days vs
60 days) - Rate limit differences

Resolution Policy:

    Docs > Blogs > Forums

------------------------------------------------------------------------

##  Example Capabilities

-   Detects when forums provide outdated information
-   Prefers official documentation when conflicts occur
-   Logs full trace of retrieval and ranking decisions
-   Provides transparent evidence for every answer

------------------------------------------------------------------------

##  Why No External LLM?

This implementation is deterministic and retrieval-focused to: - Avoid
hallucination - Ensure reproducibility - Maintain traceable
evidence-based answers

------------------------------------------------------------------------

##  License

This project is for academic demonstration purposes.

------------------------------------------------------------------------

##  Author

Srutileka Suresh\
MS in Applied Machine Learning\
University of Maryland

------------------------------------------------------------------------

