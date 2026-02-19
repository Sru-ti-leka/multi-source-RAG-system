# TaskFlow Lite Multi-Source RAG: retrieval + weighting + reranking + contradiction handling + logging

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# Load index 
def load_index(path: Path):
    return faiss.read_index(str(path))

# Load metadata
def load_metadata(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

# Retrieval (bi-encoder)
def retrieve(query: str, embed_model, index, metadata: List[dict], top_k: int = 6) -> List[dict]:
    qvec = embed_model.encode([query], normalize_embeddings=True)
    qvec = np.asarray(qvec, dtype=np.float32)

    scores, indices = index.search(qvec, top_k)

    results: List[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


# Reranking (cross-encoder)
def rerank(query: str, candidates: List[dict], tokenizer, reranker_model, top_k: int = 5) -> List[dict]:
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )

    with torch.no_grad():
        outputs = reranker_model(**inputs)
        logits = outputs.logits
        if logits.dim() == 2 and logits.shape[1] == 2:
            scores = logits[:, 1].cpu().numpy()
        else:
            scores = logits.squeeze(-1).cpu().numpy()

    reranked = []
    for c, s in zip(candidates, scores):
        item = c.copy()
        item["rerank_score"] = float(s)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# Contradiction Handling
def _canonicalize_claim(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def extract_numeric_claims(text: str) -> List[str]:
    patterns = [
        r"\b\d+\s?MB\b",
        r"\b\d+\s?days?\b",
        r"\b\d+\s?requests?/hour\b",
        r"\b\d+\s?/hour\b",
    ]
    claims = set()
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            claims.add(_canonicalize_claim(m))
    return sorted(claims)


def detect_contradictions(results: List[dict]) -> Dict[str, Any]:

    claim_mentions = defaultdict(list)
    for r in results:
        for c in extract_numeric_claims(r["text"]):
            claim_mentions[c].append(
                {
                    "source": r["source"],
                    "chunk_id": r.get("chunk_id"),
                    "source_id": r.get("source_id"),
                    "text_preview": r["text"][:160],
                }
            )

    buckets = defaultdict(set)
    for c in claim_mentions.keys():
        if c.endswith("mb"):
            buckets["attachment_mb"].add(c)
        elif "day" in c:
            buckets["api_key_days"].add(c)
        elif "hour" in c:
            buckets["rate_limit"].add(c)

    contradictions: Dict[str, Any] = {}
    for bucket, vals in buckets.items():
        if len(vals) >= 2:
            contradictions[bucket] = [
                {"claim": v, "mentions": claim_mentions[v]} for v in sorted(vals)
            ]
    return contradictions


def choose_preferred_source(evidence: List[dict]) -> str:
    for src in ["docs", "blogs", "forums"]:
        if any(e["source"] == src for e in evidence):
            return src
    return evidence[0]["source"] if evidence else "docs"


#  Answer synthesis (no external LLM)
def synthesize_answer(query: str, evidence: List[dict], contradictions: Dict[str, Any], preferred_source: str) -> str:
    best = None
    for e in evidence:
        if e["source"] == preferred_source:
            best = e
            break
    if best is None and evidence:
        best = evidence[0]

    answer_lines = []
    answer_lines.append(f"Answer (based on {preferred_source.upper()}):")
    if best:
        answer_lines.append(best["text"].strip())

    if contradictions:
        answer_lines.append("")
        answer_lines.append("Note on conflicting information:")
        answer_lines.append("I found conflicting numeric values across sources. Policy used: DOCS > BLOGS > FORUMS.")
    return "\n".join(answer_lines)


# Main
def main():
    project_root = Path(__file__).resolve().parents[1]
    index_root = project_root / "indexes"
    logs_root = project_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    # Retrieval embeddings model
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Cross-encoder reranker
    reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(reranker_name)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_name)
    reranker_model.eval()

    # Load indexes + metadata
    docs_index = load_index(index_root / "docs.faiss")
    blogs_index = load_index(index_root / "blogs.faiss")
    forums_index = load_index(index_root / "forums.faiss")

    docs_meta = load_metadata(index_root / "docs.meta.jsonl")
    blogs_meta = load_metadata(index_root / "blogs.meta.jsonl")
    forums_meta = load_metadata(index_root / "forums.meta.jsonl")

    # Source weights (intelligent multi-source combo)
    weights = {"docs": 1.2, "blogs": 1.0, "forums": 0.8}

    query = input("Ask a TaskFlow question: ").strip()

    # Retrieve per source (top_k per source)
    docs_results = retrieve(query, embed_model, docs_index, docs_meta, top_k=6)
    blogs_results = retrieve(query, embed_model, blogs_index, blogs_meta, top_k=6)
    forums_results = retrieve(query, embed_model, forums_index, forums_meta, top_k=6)

    # Apply weights + merge
    merged = []
    for r in docs_results:
        r["source"] = "docs"
        r["weighted_score"] = r["score"] * weights["docs"]
        merged.append(r)
    for r in blogs_results:
        r["source"] = "blogs"
        r["weighted_score"] = r["score"] * weights["blogs"]
        merged.append(r)
    for r in forums_results:
        r["source"] = "forums"
        r["weighted_score"] = r["score"] * weights["forums"]
        merged.append(r)

    # Pre-sort then rerank a shortlist
    merged.sort(key=lambda x: x["weighted_score"], reverse=True)
    pre_rerank = merged[:12]

    post_rerank = rerank(query, pre_rerank, tokenizer, reranker_model, top_k=5)

    # Contradiction handling
    contradictions = detect_contradictions(post_rerank)
    preferred_source = choose_preferred_source(post_rerank)

    # Minimal answer synthesis
    final_answer = synthesize_answer(query, post_rerank, contradictions, preferred_source)

    # Print
    print("\n================= FINAL ANSWER =================\n")
    print(final_answer)

    print("\n================= EVIDENCE (TOP 5) =================\n")
    for r in post_rerank:
        print(f"[{r['source'].upper()}] rerank={r['rerank_score']:.3f} weighted={r['weighted_score']:.3f}")
        print(r["text"])
        print("-" * 60)

    if contradictions:
        print("\n!!!Potential contradictions detected:")
        for bucket, items in contradictions.items():
            print(f"\n- {bucket}:")
            for it in items:
                sources = [m["source"] for m in it["mentions"]]
                print(f"  • {it['claim']} mentioned by {sources}")
        print(f"\nConflict policy applied: {preferred_source.upper()} preferred (docs > blogs > forums).")

    # Log
    log_event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "weights": weights,
        "reranker_model": reranker_name,
        "retrieved_pre_rerank": [
            {
                "source": r["source"],
                "chunk_id": r.get("chunk_id"),
                "source_id": r.get("source_id"),
                "score": r["score"],
                "weighted_score": r["weighted_score"],
                "text_preview": r["text"][:200],
            }
            for r in pre_rerank
        ],
        "retrieved_post_rerank": [
            {
                "source": r["source"],
                "chunk_id": r.get("chunk_id"),
                "source_id": r.get("source_id"),
                "rerank_score": r["rerank_score"],
                "weighted_score": r["weighted_score"],
                "text_preview": r["text"][:200],
            }
            for r in post_rerank
        ],
        "contradictions": contradictions,
        "preferred_source_policy": "docs > blogs > forums",
        "preferred_source_selected": preferred_source,
        "final_answer_preview": final_answer[:300],
    }

    log_path = logs_root / "query_logs_final.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

    print(f"\nLogged to: {log_path}\n")


if __name__ == "__main__":
    main()
