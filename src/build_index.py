# src/build_index.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Data model
@dataclass
class Chunk:
    source_type: str          
    source_id: str           
    chunk_id: str             # unique id
    text: str
    meta: Dict[str, str]      # extra fields (title, etc.)



# Chunking utilities
def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_by_words(text: str, target_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks

#Different chuncking strategy for different sources 
def chunk_docs(text: str) -> List[str]:
    return chunk_by_words(text, target_words=220, overlap_words=30)   

def chunk_blogs(text: str) -> List[str]:
    return chunk_by_words(text, target_words=180, overlap_words=40)

def chunk_forum_post(text: str) -> List[str]:
    return chunk_by_words(text, target_words=110, overlap_words=10)


# Loaders
def load_markdown_dir(dir_path: Path) -> List[Tuple[str, str]]:
    items = []
    for p in sorted(dir_path.glob("*.md")):
        items.append((p.name, p.read_text(encoding="utf-8")))
    return items


def load_forums_jsonl(jsonl_path: Path) -> List[dict]:
    threads = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            threads.append(json.loads(line))
    return threads


# Index builder
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  #cosine similarity 
    )
    return np.asarray(emb, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine
    index.add(vectors)
    return index


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    out_root = project_root / "indexes"
    out_root.mkdir(parents=True, exist_ok=True)

    # Embedding model 
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_sources = {
        "docs": data_root / "docs",
        "blogs": data_root / "blogs",
        "forums": data_root / "forums" / "threads.jsonl",
    }

    all_chunks: List[Chunk] = []

    #docs 
    for filename, raw in load_markdown_dir(all_sources["docs"]):
        text = normalize_ws(raw)
        for i, c in enumerate(chunk_docs(text)):
            all_chunks.append(
                Chunk(
                    source_type="docs",
                    source_id=filename,
                    chunk_id=f"docs::{filename}::{i}",
                    text=c,
                    meta={"filename": filename},
                )
            )

    #blogs
    for filename, raw in load_markdown_dir(all_sources["blogs"]):
        text = normalize_ws(raw)
        for i, c in enumerate(chunk_blogs(text)):
            all_chunks.append(
                Chunk(
                    source_type="blogs",
                    source_id=filename,
                    chunk_id=f"blogs::{filename}::{i}",
                    text=c,
                    meta={"filename": filename},
                )
            )

    #forums
    threads = load_forums_jsonl(all_sources["forums"])
    for t in threads:
        thread_id = str(t.get("id", "unknown"))
        title = normalize_ws(str(t.get("title", "")))
        question = normalize_ws(str(t.get("question", "")))
        replies = t.get("replies", []) or []

        # Chunk question (as one unit)
        q_text = f"TITLE: {title} | QUESTION: {question}"
        for i, c in enumerate(chunk_forum_post(q_text)):
            all_chunks.append(
                Chunk(
                    source_type="forums",
                    source_id=thread_id,
                    chunk_id=f"forums::{thread_id}::q::{i}",
                    text=c,
                    meta={"thread_id": thread_id, "title": title, "part": "question"},
                )
            )

        # Chunk replies (each reply separately)
        for r_i, r in enumerate(replies):
            author = normalize_ws(str(r.get("author", "anon")))
            upvotes = str(r.get("upvotes", "0"))
            reply_text = normalize_ws(str(r.get("text", "")))
            r_text = f"TITLE: {title} | REPLY by {author} (upvotes={upvotes}): {reply_text}"
            for i, c in enumerate(chunk_forum_post(r_text)):
                all_chunks.append(
                    Chunk(
                        source_type="forums",
                        source_id=thread_id,
                        chunk_id=f"forums::{thread_id}::r{r_i}::{i}",
                        text=c,
                        meta={"thread_id": thread_id, "title": title, "part": f"reply_{r_i}", "author": author, "upvotes": upvotes},
                    )
                )

    # 3 separate indexes 
    by_source: Dict[str, List[Chunk]] = {"docs": [], "blogs": [], "forums": []}
    for ch in all_chunks:
        by_source[ch.source_type].append(ch)

    # Build and save each index + metadata
    for source_type, chunks in by_source.items():
        print(f"\nBuilding index for {source_type} with {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        vecs = embed_texts(model, texts)

        index = build_faiss_index(vecs)

        faiss_path = out_root / f"{source_type}.faiss"
        meta_path = out_root / f"{source_type}.meta.jsonl"

        faiss.write_index(index, str(faiss_path))

        with meta_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

        print(f"Saved: {faiss_path.name}, {meta_path.name}")

    print("\n Done.")


if __name__ == "__main__":
    main()
