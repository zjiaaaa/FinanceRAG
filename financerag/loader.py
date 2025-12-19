# loader.py

import os
from financerag.common.loader import HFDataLoader
from financerag.tasks import (
    ConvFinQA, FinDER, FinQA, FinQABench,
    FinanceBench, MultiHiertt, TATQA
)


task_configs = {
    "FinDER": FinDER,
    "FinQA": FinQA,
    "TATQA": TATQA,
    "ConvFinQA": ConvFinQA,
    "FinQABench": FinQABench,
    "FinanceBench": FinanceBench,
    "MultiHiertt": MultiHiertt,
}


def load_all_tasks():
    project_root = os.getcwd()
    data_dir = os.path.join(project_root, "dataset", "data")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到資料夾 {data_dir}")

    loaded_tasks = {}

    for subset_name, TaskClass in task_configs.items():
        print(f"\n👉 初始化資料集：{subset_name}")

        class LocalTask(TaskClass):
            def load_data(self):
                loader = HFDataLoader(
                    data_folder=data_dir,
                    subset=subset_name,
                    keep_in_memory=False,
                )
                corpus, queries = loader.load()
                self.corpus = {
                    d["id"]: {"title": d["title"], "text": d["text"]} for d in corpus
                }
                self.queries = {q["id"]: q["text"] for q in queries}

        try:
            task = LocalTask()
            loaded_tasks[subset_name] = task

            print("✔ 成功載入")
            print(f"   corpus: {len(task.corpus):,} 筆")
            print(f"   queries: {len(task.queries):,} 筆")

        except Exception as e:
            print(f"失敗：{e}")

    print(f"\n=== 已成功載入 {len(loaded_tasks)} 個 dataset ===")
    return loaded_tasks
