# FinanceRAG: 金融領域檢索增強生成框架

FinanceRAG 是一個專為金融領域設計的 RAG (Retrieval-Augmented Generation) 評測與執行框架。本專案整合了多種金融數據集（如 MultiHiertt, FinQA, TATQA 等），並提供從多模型檢索合併到雙階段重排序（Reranking）。


### Environment Setup

```bash
# 複製專案
git clone https://github.com/zjiaaaa/FinanceRAG
cd FinanceRAG

To begin, install the necessary dependencies:

# Set up a virtual environment
python -m venv financerag_env
source financerag_env/bin/activate  # Windows: financerag_env\Scripts\activate

# Install the required packages
pip install -r requirements.txt

