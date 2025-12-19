import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import traceback
from tqdm import tqdm

from financerag.loader import load_all_tasks
# -------------------------------------
# 【第一部分】配置與輔助函式
# -------------------------------------

# 定義您想跑的模型名稱
TARGET_MODEL_NAME = "BAAI/bge-m3" 
TARGET_MODEL_DIRNAME = TARGET_MODEL_NAME.replace("/", "__")

# 設置 Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25 that lowercases and keeps word characters."""
    return re.findall(r"\w+", str(text).lower())


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize score array while ignoring non-finite entries.
    Non-finite values are set to 0 in the output to keep masking intact.
    """
    arr = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return np.zeros_like(arr, dtype=float)
    valid = arr[finite_mask]
    min_v, max_v = float(valid.min()), float(valid.max())
    if math.isclose(max_v, min_v):
        out = np.zeros_like(arr, dtype=float)
        out[finite_mask] = 1.0
        return out
    out = np.zeros_like(arr, dtype=float)
    out[finite_mask] = (valid - min_v) / (max_v - min_v)
    return out


class SimpleBM25:
    """
    Lightweight BM25 implementation to avoid extra dependencies.
    Operates on tokenized documents and returns a score per document.
    """

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_size = len(corpus_tokens)
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / (self.corpus_size or 1)
        self.k1 = k1
        self.b = b

        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []

        df_counts: Dict[str, int] = {}
        for doc in corpus_tokens:
            freqs: Dict[str, int] = {}
            self.doc_len.append(len(doc))
            for word in doc:
                freqs[word] = freqs.get(word, 0) + 1
            self.doc_freqs.append(freqs)
            for word in freqs:
                df_counts[word] = df_counts.get(word, 0) + 1

        for word, freq in df_counts.items():
            # Smooth IDF to avoid negative values for very frequent terms
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    @classmethod
    def build_from_texts(cls, texts: List[str], k1: float = 1.5, b: float = 0.75):
        tokens = [_tokenize(t) for t in texts]
        return cls(tokens, k1=k1, b=b)

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores: List[float] = []
        query_freqs: Dict[str, int] = {}
        for token in query_tokens:
            query_freqs[token] = query_freqs.get(token, 0) + 1

        for doc_idx, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[doc_idx] or 1
            for term, qf in query_freqs.items():
                if term not in freqs:
                    continue
                idf = self.idf.get(term, 0.0)
                f = freqs[term]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * f * (self.k1 + 1) / denom
            scores.append(score)
        return scores


def load_jsonl(path: Path) -> List[Dict]:
    """從 .jsonl 檔案載入數據."""
    data = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    except FileNotFoundError:
        raise FileNotFoundError(f"[載入失敗] 找不到檔案: {path.name}") 
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解碼錯誤在檔案: {path}, 錯誤: {e}")
        sys.exit(1)
    return data

def get_unique_numbers(meta_entry: Dict) -> Set[str]:
    """從 metadata 中提取所有關鍵數字與日期"""
    unique_set = set()
    
    if "dates" in meta_entry:
        for d in meta_entry.get("dates", []):
            d_str = str(d)
            if d_str:
                unique_set.add(d_str)
                if "-" in d_str:
                    unique_set.add(d_str.split("-")[0])
                
    if "numbers_value" in meta_entry:
        for n in meta_entry.get("numbers_value", []):
            try:
                n_float = float(n)
                unique_set.add(str(n_float)) 
                unique_set.add(str(int(n_float))) 
            except (ValueError, TypeError):
                continue

    if "numbers_plain" in meta_entry:
        for n in meta_entry.get("numbers_plain", []):
            if n:
                unique_set.add(str(n))

    return unique_set

def get_keywords(meta_entry: Dict) -> Set[str]:
    """將 metadata 中的關鍵字轉為小寫集合"""
    keywords = meta_entry.get("keywords", [])
    if not keywords:
        return set()
    return {str(k).strip().lower() for k in keywords if str(k).strip()}


# -------------------------------------
# 【第二部分】核心邏輯：向量 + 數字 + 標題混合檢索 (含 ID Filter)
# -------------------------------------
def run_hybrid_retrieval(
    task_output_dir: str, 
    top_k_final: int = 100,     
    top_k_candidate: int = 200, 
    number_bonus: float = 0.1,
    keyword_bonus: float = 0.1,
    title_bonus: float = 0.5,    # 🔥 [設定] 標題命中的加分權重
    save_results: bool = True,
    batch_size: int = 32,
    task_obj: Any = None,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> Dict[str, List[Dict[str, Any]]]:
    
    base_path = Path(task_output_dir)
    print(f"\n🚀 正在處理路徑: {Path(task_output_dir).resolve().relative_to(Path.cwd())}")

    # 1. 載入資料
    q_embs = np.load(base_path / "query_embeddings.npy")
    c_embs = np.load(base_path / "corpus_embeddings.npy")
    q_meta = load_jsonl(base_path / "query_metadata.jsonl")
    c_meta = load_jsonl(base_path / "corpus_metadata.jsonl")

    # =================================================
    # 🔥 [新增功能 A] 建立 ID 索引對照表 (用於快速過濾)
    # =================================================
    print("⚡ 正在建立 ID 索引對照表 (加速 Filtering)...")
    doc_id_to_indices = {}
    for idx, meta in enumerate(c_meta):
        d_id = meta['doc_id']
        if d_id not in doc_id_to_indices:
            doc_id_to_indices[d_id] = []
        doc_id_to_indices[d_id].append(idx)

    num_queries = len(q_embs)
    print(f"⚡ 開始分批混合檢索 (Querys: {num_queries}, Corpus: {len(c_embs)})")
    
    results = {}

    # 準備 BM25 (若有 task_obj 才能取得文字內容)
    bm25_engine = None
    bm25_query_tokens = {}
    if task_obj is None:
        print("ℹ️ 未提供 task_obj，BM25 將被略過，僅使用 Dense 檢索。")
    else:
        try:
            # 依 metadata 順序組合 chunk 文本，確保與向量對齊
            bm25_corpus_texts = []
            for meta in c_meta:
                doc_id = meta["doc_id"]
                chunk_idx = meta.get("chunk_index")
                doc = task_obj.corpus.get(doc_id, {})
                chunks = doc.get("chunks") or []
                if chunk_idx is None:
                    body = doc.get("text", "")
                elif 0 <= chunk_idx < len(chunks):
                    body = chunks[chunk_idx]
                else:
                    body = doc.get("text", "")
                title = doc.get("title", "")
                combined = "\n".join([t for t in [title, body] if t]).strip()
                bm25_corpus_texts.append(combined)

            bm25_engine = SimpleBM25.build_from_texts(
                bm25_corpus_texts, k1=bm25_k1, b=bm25_b
            )
            bm25_query_tokens = {
                qid: _tokenize(q_data["text"] if isinstance(q_data, dict) else str(q_data))
                for qid, q_data in task_obj.queries.items()
            }
            print(f"⚙️ 已建立 BM25 索引 (docs: {len(bm25_corpus_texts)})")
        except Exception as exc:
            print(f"⚠️ BM25 初始化失敗，改用 Dense 單一路徑：{exc}")
            bm25_engine = None

    # 2. 分批迴圈
    for start_idx in tqdm(range(0, num_queries, batch_size), desc="Retrieving Batches"):
        end_idx = min(start_idx + batch_size, num_queries)
        
        batch_q_embs = q_embs[start_idx : end_idx]
        
        # 計算純向量分數
        batch_scores = np.dot(batch_q_embs, c_embs.T)
        
        for local_i, q_row in enumerate(batch_scores):
            global_i = start_idx + local_i
            
            query_data = q_meta[global_i]
            query_id = query_data['query_id']
            q_numbers = get_unique_numbers(query_data)
            q_keywords = get_keywords(query_data)
            
            # =================================================
            # 🔥 [新增功能 B] ID Filtering 過濾邏輯
            # =================================================
            allowed_ids = query_data.get("filter_doc_ids", None) # 讀取 query 中的允許清單

            if allowed_ids:
                valid_mask = np.zeros(len(c_embs), dtype=bool)
                has_valid_doc = False
                
                for d_id in allowed_ids:
                    if d_id in doc_id_to_indices:
                        indices = doc_id_to_indices[d_id]
                        valid_mask[indices] = True
                        has_valid_doc = True
                
                # 若該 Query 有合法的目標文件，將不合法的文件分數設為 -inf
                if has_valid_doc:
                    q_row[~valid_mask] = -np.inf

            dense_scores = np.asarray(q_row, dtype=float)

            bm25_scores = None
            if bm25_engine is not None and query_id in bm25_query_tokens:
                bm25_scores = np.asarray(bm25_engine.get_scores(bm25_query_tokens[query_id]), dtype=float)
                if allowed_ids and has_valid_doc:
                    bm25_scores[~valid_mask] = -np.inf

            # 混合 Dense 與 BM25 分數
            if bm25_scores is not None:
                dense_norm = _normalize_scores(dense_scores)
                bm25_norm = _normalize_scores(bm25_scores)
                combined_scores = dense_weight * dense_norm + bm25_weight * bm25_norm
                base_for_sort = combined_scores
            else:
                base_for_sort = dense_scores

            if allowed_ids and has_valid_doc:
                base_for_sort[~valid_mask] = -np.inf
            
            # --- 檢索與重排序 ---
            # 因為上面可能改過 q_row 分數了，這裡的 argsort 會自動把 -inf 排到最後
            candidate_indices = np.argsort(base_for_sort)[-top_k_candidate:][::-1]
            scored_candidates = []
            seen_docs = set()

            for idx in candidate_indices:
                if not np.isfinite(base_for_sort[idx]):
                    continue

                doc_data = c_meta[idx]
                base_score = float(base_for_sort[idx])
                dense_score = float(dense_scores[idx])
                bm25_score = float(bm25_scores[idx]) if bm25_scores is not None else None
                final_score = base_score
                
                # 1. 數字加分
                d_numbers = get_unique_numbers(doc_data)
                match_count = 0
                matches = []
                if q_numbers:
                    common = q_numbers.intersection(d_numbers)
                    match_count = len(common)
                    matches = list(common)
                
                if match_count > 0:
                    final_score += (match_count * number_bonus)

                # 2. 關鍵字加分 (內文)
                keyword_matches = []
                if q_keywords:
                    keyword_common = q_keywords.intersection(get_keywords(doc_data))
                    keyword_matches = list(keyword_common)
                    if keyword_common:
                        final_score += (len(keyword_common) * keyword_bonus)

                # =================================================
                # 🔥 [新增功能 C] 標題關鍵字加分 (Title Bonus)
                # =================================================
                doc_title = str(doc_data.get('title', '')).lower()
                title_matches = []
                if q_keywords and doc_title:
                    for kw in q_keywords:
                        if kw in doc_title: # 檢查關鍵字是否在標題字串中
                            title_matches.append(kw)
                    
                    if title_matches:
                        final_score += (len(title_matches) * title_bonus)

                # 存入候選列表
                unique_key = f"{doc_data['doc_id']}_{doc_data['chunk_index']}"
                if unique_key not in seen_docs:
                    scored_candidates.append({
                        "doc_id": doc_data['doc_id'],
                        "chunk_index": doc_data['chunk_index'],
                        "title": doc_data.get('title', ''),
                        "score": final_score,
                        "hybrid_score": base_score,
                        "vector_score": dense_score,
                        **({"bm25_score": bm25_score} if bm25_score is not None else {}),
                        "matches": matches,
                        "keyword_matches": keyword_matches,
                        "title_matches": title_matches # 🔥 紀錄標題命中
                    })
                    seen_docs.add(unique_key)

            # 依 doc 聚合
            doc_best = {}
            for cand in scored_candidates:
                doc_id = cand["doc_id"]
                if doc_id not in doc_best or cand["score"] > doc_best[doc_id]["score"]:
                    doc_best[doc_id] = cand

            aggregated = sorted(doc_best.values(), key=lambda x: x['score'], reverse=True)
            results[query_id] = aggregated[:top_k_final]

    # 3. 存檔
    if save_results:
        save_path = base_path / "retrieval_results.json" 
        logger.info(f"[Auto-Save] 正在存檔至: {save_path.name}")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"存檔失敗: {e}")

    return results

# -------------------------------------
# 【第三部分】執行邏輯
# -------------------------------------
if __name__ == "__main__":
    
    root_dir_path = Path("dataset/embeddings") 
    if not root_dir_path.exists():
        candidate = Path(__file__).resolve().parent.parent / root_dir_path
        if candidate.exists():
            root_dir_path = candidate
    
    if not root_dir_path.exists():
        print(f"致命錯誤：找不到根目錄 {root_dir_path.name}！")
        sys.exit(1)

    all_task_dirs = [p.parent for p in root_dir_path.rglob("config.json")]
    target_task_dirs = [
        d for d in all_task_dirs if d.name in {TARGET_MODEL_NAME, TARGET_MODEL_DIRNAME}
    ]
    
    if not target_task_dirs:
        print(f"❌ 錯誤：找不到模型 '{TARGET_MODEL_NAME}' 的結果資料夾。")
        sys.exit(1)

    print(f"✅ 找到 {len(target_task_dirs)} 個 '{TARGET_MODEL_NAME}' 任務資料夾，準備處理...")

    try:
        all_tasks = load_all_tasks()
    except Exception as e:
        print(f"❌ 無法載入任務資料，BM25 會失效：{e}")
        all_tasks = {}

    for task_dir in target_task_dirs:
        print(f"\n======== [處理中] {task_dir.parent.name} / {task_dir.name} ========")
        task_obj = all_tasks.get(task_dir.parent.name)
        
        try:
            # 1. 執行混合檢索 (記得調整 title_bonus 權重)
            final_results = run_hybrid_retrieval(
                task_output_dir=str(task_dir),
                top_k_final=50,      
                number_bonus=0.15,
                keyword_bonus=0.1,
                title_bonus=0.5,    # 🔥 設定標題命中權重
                task_obj=task_obj,
            )

            # 2. 列印檢查結果
            print("\n" + "="*60)
            print(f"👀 檢索結果範例 (前 2 個 Query)")
            print("="*60)
            
            for qid, docs in list(final_results.items())[:2]:
                print(f"\n❓ Query ID: {qid}")
                for rank, doc in enumerate(docs[:3]): 
                    # 顯示詳細命中資訊
                    extras = []
                    if doc.get('matches'): extras.append(f"數:{len(doc['matches'])}")
                    if doc.get('title_matches'): extras.append(f"標:{doc['title_matches']}")
                    extra_str = f"| {' '.join(extras)}" if extras else ""

                    bm25_part = f", BM25: {doc['bm25_score']:.4f}" if 'bm25_score' in doc else ""
                    hybrid_part = doc.get('hybrid_score', doc.get('vector_score', doc['score']))
                    print(
                        f"   [{rank+1}] Final: {doc['score']:.4f} "
                        f"(Hybrid: {hybrid_part:.4f}, Vec: {doc.get('vector_score', 0):.4f}{bm25_part}) "
                        f"{extra_str} | {doc['doc_id']}"
                    )
            
            print("\n" + "-"*30 + "\n")

        except Exception as e:
            print(f"處理 {task_dir.name} 時發生錯誤: {e}")
            traceback.print_exc()
