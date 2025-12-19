import json
import torch
import numpy as np
import csv
import gc
import sys
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from tqdm import tqdm
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from financerag.loader import load_all_tasks

TARGET_RETRIEVAL_MODEL = "BAAI/bge-m3"
TARGET_RETRIEVAL_DIRNAME = TARGET_RETRIEVAL_MODEL.replace("/", "__")

# ==========================================
# 1. Reranker 類別 (核心模型)
# ==========================================
class Reranker:
    def __init__(self, model_name: str, batch_size: int = 4):
        print(f" [Init] Loading Reranker model: {model_name} ...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        # 強制開啟 fp16 半精度以節省記憶體
        self.model = CrossEncoder(
            model_name, 
            max_length=512, 
            device=device,
            trust_remote_code=True,
            automodel_args={"torch_dtype": torch.float16} 
        )
        self.batch_size = batch_size

    def rerank(self, query_text: str, candidates: List[Dict], corpus_data: Dict) -> List[Dict]:
        if not candidates:
            return []

        pairs = []
        valid_indices = []

        for i, doc_info in enumerate(candidates):
            doc_id = doc_info['doc_id']
            chunk_idx = doc_info['chunk_index']
            
            if doc_id not in corpus_data:
                continue 
            
            doc_obj = corpus_data[doc_id]

            # 提取標題與內文
            title = str(doc_obj.get("title", "")).strip()
            content = ""
            
            if chunk_idx is not None and "chunks" in doc_obj and doc_obj["chunks"]:
                if 0 <= chunk_idx < len(doc_obj["chunks"]):
                    content = doc_obj["chunks"][chunk_idx]
                else:
                    content = doc_obj.get("text", "")
            else:
                content = doc_obj.get("text", "")

            # 組合 Title + Content
            if title:
                doc_text = f"{title}\n{content}"
            else:
                doc_text = content

            pairs.append([query_text, str(doc_text)])
            valid_indices.append(i)

        if not pairs:
            return []

        # 預測分數
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        # 更新分數
        reranked_results = []
        for j, list_idx in enumerate(valid_indices):
            original_doc = candidates[list_idx].copy()
            original_doc['rerank_score'] = float(scores[j])
            reranked_results.append(original_doc)

        # 排序
        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_results 

# ==========================================
# 2. 輔助工具：轉 Kaggle CSV
# ==========================================
def save_kaggle_submission(rerank_results: Dict, output_dir: Path):
    csv_path = output_dir / "submission.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "corpus_id"])
        for qid, docs in rerank_results.items():
            for doc in docs:
                writer.writerow([qid, doc['doc_id']])
    print(f"   🏆 Kaggle submission generated: {csv_path.name}")

# ==========================================
# 3. [新增] 執行單一任務的封裝函式 (給 main.py 呼叫用)
# ==========================================
def run_reranking(
    task_obj: Any, 
    retrieval_results: Dict, 
    output_dir: Path, 
    top_k_rerank: int = 10,
    model_name: str = "BAAI/bge-reranker-base",
    pre_rerank_limit: int = 200,
) -> Dict:
    
    # 1. 初始化模型 (每次呼叫都重新建立，確保乾淨)
    # batch_size 設為 2 比較保險
    reranker = Reranker(model_name=model_name, batch_size = 4)
    
    final_output = {}
    print(f" Reranking {len(retrieval_results)} queries...")
    
    # 2. 執行迴圈
    for qid, docs in tqdm(retrieval_results.items(), desc="Reranking"):
        if qid in task_obj.queries:
            q_data = task_obj.queries[qid]
            query_text = q_data["text"] if isinstance(q_data, dict) else q_data

            # 先截斷候選，避免浪費 CrossEncoder 計算
            prefiltered_docs = docs[:pre_rerank_limit] if pre_rerank_limit else docs
            
            ranked_docs = reranker.rerank(query_text, prefiltered_docs, task_obj.corpus)
            final_output[qid] = ranked_docs[:top_k_rerank]

    # 3. 存檔 (JSON + CSV)
    save_path = output_dir / "rerank_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    save_kaggle_submission(final_output, output_dir)
    
    # 4. 清理記憶體 (非常重要)
    del reranker
    torch.cuda.empty_cache()
    gc.collect()
    
    return final_output

# ==========================================
# 4. 輔助：合併多個檢索結果 (多模型並集)
# ==========================================
def merge_retrieval_results(retrieval_sets: List[Dict], pre_rerank_limit: int = 200) -> Dict:
    """
    將不同檢索模型的結果合併，針對同一 doc/chunk 保留最高分，最後截斷至 pre_rerank_limit。
    """
    merged: Dict[str, Dict[Any, Dict]] = {}

    for retrieval_results in retrieval_sets:
        for qid, docs in retrieval_results.items():
            qslot = merged.setdefault(qid, {})
            for doc in docs:
                key = (doc.get("doc_id"), doc.get("chunk_index"))
                score = doc.get("score", doc.get("vector_score", 0.0))
                keep = qslot.get(key)
                if (keep is None) or (score > keep.get("score", 0.0)):
                    new_doc = doc.copy()
                    new_doc["score"] = score  # 確保後續排序有值
                    qslot[key] = new_doc

    trimmed: Dict[str, List[Dict]] = {}
    for qid, doc_map in merged.items():
        merged_list = sorted(doc_map.values(), key=lambda x: x.get("score", 0.0), reverse=True)
        trimmed[qid] = merged_list[:pre_rerank_limit] if pre_rerank_limit else merged_list
    return trimmed

# ==========================================
# 4. 自動批次執行邏輯 (單獨執行此檔案時)
# ==========================================
if __name__ == "__main__":
    
    # --- 定義任務與模型對應表 ---
    MODEL_MAP = {
        "MultiHiertt": "jinaai/jina-reranker-v2-base-multilingual",
        "FinQABench": "jinaai/jina-reranker-v2-base-multilingual",
        "FinanceBench": "jinaai/jina-reranker-v2-base-multilingual",
        "FinQA": "Alibaba-NLP/gte-multilingual-reranker-base",
        "TATQA": "BAAI/bge-reranker-v2-m3",
        "ConvFinQA": "BAAI/bge-reranker-v2-m3",
        "FinDER": "BAAI/bge-reranker-v2-m3"
    }
    
    DEFAULT_MODEL = "BAAI/bge-reranker-base"

    print("\n=== [1/2] Scanning for Retrieval Results ===")
    root_dir_path = Path("dataset/embeddings")
    if not root_dir_path.exists():
        candidate = Path(__file__).resolve().parent.parent / root_dir_path
        if candidate.exists():
            root_dir_path = candidate

    all_json_files = list(root_dir_path.rglob("retrieval_results.json"))

    # 依任務聚合所有模型的檢索結果
    task_to_files: Dict[str, List[Path]] = {}
    for f in all_json_files:
        task_name = f.parent.parent.name
        task_to_files.setdefault(task_name, []).append(f)
      
    if not task_to_files:
        print(" 找不到任何 retrieval_results.json，請先執行 Retrieval 步驟！")
        exit()
        
    print(f"🔍 找到 {len(task_to_files)} 個任務待處理。")

    print("\n=== [2/2] Start Batch Reranking ===")

    for task_name, files in task_to_files.items():
        # 選擇輸出目錄：優先用 BGE 路徑，否則取第一個
        output_dir = None
        for f in files:
            if f.parent.name in {TARGET_RETRIEVAL_MODEL, TARGET_RETRIEVAL_DIRNAME}:
                output_dir = f.parent
                break
        if output_dir is None:
            output_dir = files[0].parent

        # 檢查是否已完成 (有 CSV 就跳過)
        # csv_output_path = output_dir / "submission.csv"
        # if csv_output_path.exists():
        #     print(f"⏩ [Skip] Task: {task_name} 已經有 submission.csv，跳過。")
        #     continue 
        
        print(f"\n🚀 [Processing] Task: {task_name}")
        model_name = MODEL_MAP.get(task_name, DEFAULT_MODEL)
        
        try:
            # 記憶體安全載入
            print("   ⏳ Loading dataset...")
            all_tasks = load_all_tasks()
            
            if task_name not in all_tasks:
                print(f"   ⚠️ 跳過：Dataset 中找不到 {task_name}")
                del all_tasks
                gc.collect()
                continue
                
            task_obj = all_tasks[task_name]
            del all_tasks
            gc.collect()

            # 讀取並合併所有模型的 Retrieval 結果
            retrieval_sets = []
            for json_file in files:
                with open(json_file, "r", encoding="utf-8") as f:
                    retrieval_sets.append(json.load(f))

            merged_retrieval = merge_retrieval_results(retrieval_sets, pre_rerank_limit=200)

            # 👇 直接呼叫我們剛剛寫好的 run_reranking 函式
            run_reranking(
                task_obj=task_obj,
                retrieval_results=merged_retrieval,
                output_dir=output_dir,
                top_k_rerank=10,
                model_name=model_name,
                pre_rerank_limit=200,
            )
            
            print(f"   ✅ Done: {task_name}")

            # 雙重清理 (run_reranking 裡面清過一次，這裡再清一次確保 task_obj 也清掉)
            del task_obj
            gc.collect()

        except Exception as e:
            print(f"   ❌ Error processing {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== 🎉 All Reranking Tasks Completed! ===")
