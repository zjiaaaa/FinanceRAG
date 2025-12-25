## FinanceRAG

FinanceRAG is a Retrieval-Augmented Generation (RAG) evaluation and execution framework designed specifically for the financial domain.

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/zjiaaaa/FinanceRAG
cd FinanceRAG

# Set up a virtual environment
python -m venv financerag_env
source financerag_env/bin/activate  # Windows: financerag_env\Scripts\activate

# Install the required packages
pip install -r requirements.txt

### Dataset Preparation

Datasets are not included in this repository and must be downloaded manually from Kaggle.

# Install Kaggle API
pip install kaggle

# Download the FinanceRAG dataset (ICAIF 2024 Finance RAG Challenge)
kaggle competitions download -c icaif-24-finance-rag-challenge

# Unzip the dataset
unzip icaif-24-finance-rag-challenge.zip
