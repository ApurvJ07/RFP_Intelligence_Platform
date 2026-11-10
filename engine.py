import os
import json
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from ingestion.parser import ingest_bid_folder

QUERY_EXPANSIONS = {
    "deadline": "due date submission deadline closing date solicitation due",
    "due date": "due date submission deadline closing date extend",
    "bond": "bid bond security guarantee percentage surety",
    "affidavit": "affidavit form 1295 w-9 w9 mwbe mercury contract certificate",
    "warranty": "warranty extended warranty repair replacement years oem",
    "specs": "specifications processor cpu ram memory storage display screen",
    "contact": "buyer contact representative email phone poc",
}

def chunk_documents(pages: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Section/table-aware chunking with overlap."""
    chunks = []
    for page in pages:
        text = page["text"]
        paragraphs = text.split("\n\n")
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks.append(_make_chunk(page, current_chunk, len(chunks)))
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = (overlap_text + "\n\n" + para).strip()
                if len(current_chunk) > 2500:
                    chunks.append(_make_chunk(page, current_chunk[:2500], len(chunks)))
                    current_chunk = current_chunk[2300:]
        if current_chunk:
            chunks.append(_make_chunk(page, current_chunk, len(chunks)))
    return chunks

def _make_chunk(page: dict, text: str, idx: int) -> dict:
    raw_id = f"{page['bid_id']}_{page['file_name']}_p{page['page_number']}_{idx}_{text[:30]}"
    chunk_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
    return {
        "chunk_id": chunk_id,
        "bid_id": page["bid_id"],
        "file_name": page["file_name"],
        "doc_type": page["doc_type"],
        "addendum_number": page["addendum_number"] if page["addendum_number"] is not None else -1,
        "page_number": page["page_number"],
        "text": text
    }

class HybridSearchEngine:
    def __init__(self, persist_dir: str = "data/chroma_db"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="rfp_chunks",
            embedding_function=self.emb_fn
        )
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.bm25_corpus = []
        self._load_bm25_from_chroma()

    def _load_bm25_from_chroma(self):
        existing = self.collection.get(include=["documents", "metadatas"])
        self.bm25_corpus = []
        if existing and existing["ids"]:
            for cid, doc, meta in zip(existing["ids"], existing["documents"], existing["metadatas"]):
                self.bm25_corpus.append({
                    "chunk_id": cid,
                    "text": doc,
                    **meta
                })
        if self.bm25_corpus:
            tokenized = [c["text"].lower().split() for c in self.bm25_corpus]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def index_bid_folder(self, folder_path: str) -> int:
        pages = ingest_bid_folder(folder_path)
        chunks = chunk_documents(pages)
        if not chunks:
            return 0

        ids = [c["chunk_id"] for c in chunks]
        docs = [c["text"] for c in chunks]
        metas = [{
            "bid_id": c["bid_id"],
            "file_name": c["file_name"],
            "doc_type": c["doc_type"],
            "addendum_number": c["addendum_number"],
            "page_number": c["page_number"]
        } for c in chunks]

        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        self._load_bm25_from_chroma()
        return len(chunks)

    def expand_query(self, query: str) -> str:
        expanded = query
        q_lower = query.lower()
        for key, addition in QUERY_EXPANSIONS.items():
            if key in q_lower:
                expanded += " " + addition
        return expanded

    def search(self, query: str, bid_id: str = None, doc_type: str = None,
               top_k: int = 5, mode: str = "hybrid_rerank") -> list[dict]:
        expanded_query = self.expand_query(query)

        where_clause = None
        filters = []
        if bid_id:
            filters.append({"bid_id": bid_id})
        if doc_type:
            filters.append({"doc_type": doc_type})
        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}

        vec_results = self.collection.query(
            query_texts=[expanded_query],
            n_results=max(top_k * 3, 15),
            where=where_clause
        )

        vec_Support = {}
        chunk_lookup = {}
        if vec_results["ids"] and vec_results["ids"][0]:
            for rank, (cid, doc, meta) in enumerate(zip(
                vec_results["ids"][0], vec_results["documents"][0], vec_results["metadatas"][0]
            )):
                vec_Support[cid] = 1.0 / (60 + rank + 1)
                chunk_lookup[cid] = {"chunk_id": cid, "text": doc, **meta}

        if mode == "vector":
            sorted_ids = sorted(vec_Support.keys(), key=lambda x: vec_Support[x], reverse=True)[:top_k]
            return [{**chunk_lookup[cid], "score": round(vec_Support[cid], 4)} for cid in sorted_ids]

        bm25_support = {}
        if self.bm25 and self.bm25_corpus:
            tokens = expanded_query.lower().split()
            scores = self.bm25.get_scores(tokens)
            scored_items = []
            for idx, score in enumerate(scores):
                item = self.bm25_corpus[idx]
                if bid_id and item["bid_id"] != bid_id:
                    continue
                if doc_type and item["doc_type"] != doc_type:
                    continue
                scored_items.append((score, item))
            scored_items.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, item) in enumerate(scored_items[:max(top_k * 3, 15)]):
                cid = item["chunk_id"]
                bm25_support[cid] = 1.0 / (60 + rank + 1)
                chunk_lookup[cid] = item

        all_cids = set(vec_Support.keys()) | set(bm25_support.keys())
        rrf_scores = {
            cid: vec_Support.get(cid, 0.0) + bm25_support.get(cid, 0.0)
            for cid in all_cids
        }
        top_candidates = sorted(all_cids, key=lambda x: rrf_scores[x], reverse=True)[:max(top_k * 2, 12)]

        if mode == "hybrid" or not top_candidates:
            return [{**chunk_lookup[cid], "score": round(rrf_scores[cid], 4)} for cid in top_candidates[:top_k]]

        pairs = [[query, chunk_lookup[cid]["text"]] for cid in top_candidates]
        ce_scores = self.reranker.predict(pairs)
        reranked = sorted(zip(top_candidates, ce_scores), key=lambda x: float(x[1]), reverse=True)[:top_k]

        return [{
            **chunk_lookup[cid],
            "score": round(float(score), 4)
        } for cid, score in reranked]