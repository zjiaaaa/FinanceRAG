import json
import gc
import torch
from pathlib import Path
from financerag.loader import load_all_tasks
from financerag.preprocess import preprocess_task
from financerag.embedding import embed_tasks
from financerag.retrieval.retrieval import run_hybrid_retrieval
from financerag.rerank.rerank import run_reranking

def main():
    print("=== Loading datasets ===")
    loaded_tasks = load_all_tasks()

    print("=== Preprocessing datasets ===")
    for task_name, task_obj in loaded_tasks.items():
        preprocess_task(task_name, task_obj)

    print("=== Embedding datasets ===")
    embed_results = embed_tasks(loaded_tasks)

    print("=== Retrieving Documents ===")
    for task_name, paths in embed_results.items():
        print(f"\n🔍 Processing retrieval for task: {task_name}")
        
        output_dir = paths['config'].parent
        
        retrieval_results = run_hybrid_retrieval(
            task_output_dir=str(output_dir),
            top_k_final=50,
            number_bonus=0.1,
            task_obj=loaded_tasks[task_name],
        )
        print("   >> Running BGE Reranker...")
        
        final_results = run_reranking(
            task_obj=loaded_tasks[task_name], # 關鍵：傳入原始物件以查閱文字
            retrieval_results=retrieval_results,
            output_dir=output_dir,
            top_k_rerank=10       # 最終只留精華的 10 筆
        )
        
        # 簡單驗證
        first_qid = next(iter(final_results))
        print(f"   Task Done! Sample Doc: {final_results[first_qid][0]['doc_id']}")

        # 每次跑完一個任務的 Rerank，手動清一下 GPU 記憶體
        # (雖然 loaded_tasks 還在 RAM 裡，但 GPU VRAM 可以清空)
        torch.cuda.empty_cache()
        gc.collect()

    print(f"   --> 檢索完成，共處理 {len(retrieval_results)} 個 Query。")

    print("\n=== All Pipeline Done ===")


    print("=== Done ===")


if __name__ == "__main__":
    main()
