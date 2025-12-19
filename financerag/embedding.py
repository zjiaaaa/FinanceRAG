from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


# ===============================
# Embedding Config
# ===============================
@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"  
    batch_size: int = 32
    normalize: bool = True
    chunk_key: str = "chunks"
    text_key: Literal["text"] = "text"
    title_key: Literal["title"] = "title"
    output_dir: str = "dataset/embeddings"

    @property
    def model_slug(self) -> str:
        return self.model_name.replace("/", "__")


# Helpers
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



# Encode texts (SentenceTransformer)
def _encode_texts(
        model: SentenceTransformer,
        texts: List[str],
        batch_size: int,
        normalize: bool,
) -> np.ndarray:

    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype="float32")

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return np.asarray(vectors, dtype="float32")


# Build Corpus Entries

def _build_corpus_entries(
        corpus: Dict[str, Dict[str, Any]],
        cfg: EmbeddingConfig,
) -> Tuple[List[str], List[Dict[str, Any]]]:

    texts = []
    metadata = []

    for doc_id, doc in corpus.items():

        title = str(doc.get(cfg.title_key, "")).strip()
        base_text = str(doc.get(cfg.text_key, "")).strip()
        chunks = doc.get(cfg.chunk_key, [])

        numbers_value = doc.get("numbers_value", [])
        numbers_plain = doc.get("numbers_plain", [])
        dates = doc.get("dates", [])
        keywords = doc.get("keywords", [])

        numbers_all = doc.get("numbers_all", [])
        numbers_str = " ".join(map(str, numbers_all))
        keywords_str = " ".join(map(str, keywords))

        meta_base = {
            "doc_id": doc_id,
            "title": title,
            "numbers_value": numbers_value,
            "numbers_plain": numbers_plain,
            "dates": dates,
            "keywords": keywords
        }

        # chunk embedding
        if isinstance(chunks, list) and chunks:
            for idx, chunk in enumerate(chunks):

                chunk_text = str(chunk).strip()

                augmented = f"passage: {title} -- {chunk_text}"

                extras = []
                if numbers_all:
                    extras.append(f"NUM: {numbers_str}")
                if keywords:
                    extras.append(f"KEY: {keywords_str}")
                if extras:
                    augmented = augmented + "\n" + "\n".join(extras)

                texts.append(augmented)
                row_meta = meta_base.copy()
                row_meta["chunk_index"] = idx
                metadata.append(row_meta)

        else:
            combined = f"passage: {title} {base_text}".strip()

            extras = []
            if numbers_all:
                extras.append(f"NUM: {numbers_str}")
            if keywords:
                extras.append(f"KEY: {keywords_str}")
            if extras:
                combined += "\n" + "\n".join(extras)

            texts.append(combined)
            row_meta = meta_base.copy()
            row_meta["chunk_index"] = None
            metadata.append(row_meta)
    return texts, metadata


# Build Query Entries
def _build_query_entries(
        queries: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:

    texts = []
    metadata = []

    for qid, q in queries.items():

        if isinstance(q, dict):
            text = q.get("text", "")
            numbers_all = q.get("numbers_all", [])
            numbers_value = q.get("numbers_value", [])
            numbers_plain = q.get("numbers_plain", [])
            dates = q.get("dates", [])
            keywords = q.get("keywords", [])
        else:
            text = q
            numbers_all = []
            numbers_value = []
            dates = []
            numbers_plain = []
            keywords = []

        text = str(text).strip()

        augmented = f"query: {text}"
        extras = []
        if numbers_all:
            extras.append(f"NUM: {' '.join(map(str, numbers_all))}")
        if keywords:
            extras.append(f"KEY: {' '.join(map(str, keywords))}")
        if extras:
            augmented = augmented + "\n" + "\n".join(extras)

        texts.append(augmented)
        metadata.append({
            "query_id": qid,
            "numbers_value": numbers_value,
            "dates": dates,
            "numbers_plain": numbers_plain,
            "keywords": keywords
        })

    return texts, metadata


# Embed One Task

def embed_task(
        task_name: str,
        task_obj: Any,
        cfg: Optional[EmbeddingConfig] = None,
        model: Optional[SentenceTransformer] = None,
) -> Dict[str, Path]:

    if cfg is None:
        cfg = EmbeddingConfig()

    if model is None:
        model = SentenceTransformer(cfg.model_name)

    if not getattr(task_obj, "corpus", None):
        raise ValueError("Task has no corpus.")

    if not getattr(task_obj, "queries", None):
        raise ValueError("Task has no queries.")

    task_dir = Path(cfg.output_dir) / task_name / cfg.model_slug
    _ensure_dir(task_dir)

    # ---- Corpus ----
    corpus_texts, corpus_meta = _build_corpus_entries(task_obj.corpus, cfg)
    corpus_embeddings = _encode_texts(
        model=model,
        texts=corpus_texts,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize,
    )

    # ---- Queries ----
    query_texts, query_meta = _build_query_entries(task_obj.queries)
    query_embeddings = _encode_texts(
        model=model,
        texts=query_texts,
        batch_size=cfg.batch_size,
        normalize=cfg.normalize,
    )

    # ---- Save ----
    corpus_path = task_dir / "corpus_embeddings.npy"
    query_path = task_dir / "query_embeddings.npy"
    corpus_meta_path = task_dir / "corpus_metadata.jsonl"
    query_meta_path = task_dir / "query_metadata.jsonl"
    config_path = task_dir / "config.json"

    np.save(corpus_path, corpus_embeddings)
    np.save(query_path, query_embeddings)
    _write_jsonl(corpus_meta_path, corpus_meta)
    _write_jsonl(query_meta_path, query_meta)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    return {
        "corpus_embeddings": corpus_path,
        "query_embeddings": query_path,
        "corpus_metadata": corpus_meta_path,
        "query_metadata": query_meta_path,
        "config": config_path,
    }


# ===============================
# Embed All Tasks
# ===============================
def embed_tasks(
        tasks: Dict[str, Any],
        cfg: Optional[EmbeddingConfig] = None,
) -> Dict[str, Dict[str, Path]]:

    if cfg is None:
        cfg = EmbeddingConfig()

    model = SentenceTransformer(cfg.model_name)

    results = {}
    for name, task in tasks.items():
        results[name] = embed_task(
            task_name=name,
            task_obj=task,
            cfg=cfg,
            model=model,
        )
    return results

