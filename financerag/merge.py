import csv
from pathlib import Path

def merge_all_submissions(root_dir: str, output_file: str = "submission.csv"):
    root_path = Path(root_dir)

    # If running from a subfolder, fall back to path relative to repo root
    if not root_path.exists():
        candidate = Path(__file__).resolve().parent.parent / root_path
        if candidate.exists():
            root_path = candidate
    
    csv_files = list(root_path.rglob("submission.csv"))
    
    # 排除掉自己 (如果 output_file 也在同目錄下，避免無線迴圈)
    csv_files = [f for f in csv_files if f.name == "submission.csv" and f.resolve() != Path(output_file).resolve()]

    if not csv_files:
        print(f"❌ 在 {root_path} 底下找不到任何 submission.csv 檔案！")
        print("   請確認 Rerank 步驟是否已執行完畢並生成了 CSV。")
        return

    print(f"🔍 找到 {len(csv_files)} 個提交檔，準備合併...")
    for f in csv_files:
        # 顯示任務名稱 (通常是上兩層資料夾名)
        task_name = f.parent.parent.name 
        print(f"   ➕ 加入: {task_name} ({f})")

    # 2. 開始合併
    total_rows = 0
    query_ids_seen = set()
    
    # 開啟輸出的檔案
    with open(output_file, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        
        # 寫入 Kaggle 規定的唯一標頭 (Header)
        writer.writerow(["query_id", "corpus_id"])
        
        # 逐一讀取每個小 CSV
        for csv_file in csv_files:
            try:
                with open(csv_file, "r", encoding="utf-8") as f_in:
                    reader = csv.reader(f_in)
                    header = next(reader, None) 

                    # 寫入資料
                    for row in reader:
                        if not row: continue # 跳過空行
                        
                        qid = row[0]
                        query_ids_seen.add(qid)
                        
                        writer.writerow(row)
                        total_rows += 1
                        
            except Exception as e:
                print(f"   ❌ 讀取 {csv_file} 時發生錯誤: {e}")

    print("="*40)
    print(f" 合併完成！")
    print(f"📄 輸出檔案: {Path(output_file).absolute()}")
    print(f"📊 總資料筆數: {total_rows}")
    print(f"🔢 總 Query 數: {len(query_ids_seen)}")
    print("🚀 請將此檔案上傳至 Kaggle！")

if __name__ == "__main__":
    # 設定您的總目錄 (通常是 dataset/embeddings)
    target_root = "dataset/embeddings" 
    
    merge_all_submissions(target_root)
