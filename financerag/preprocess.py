import os
import re
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 只有這兩個 task 要做 summarization
SUMM_TASKS = {"FinanceBench", "FinQABench"}

# BART large CNN 摘要模型
SUMM_MODEL = "facebook/bart-large-cnn"
# 可用環境變數關閉摘要，預設開啟
USE_SUMMARIZATION = os.getenv("USE_SUMM", "1").lower() not in {"0", "false", "off"}

_SUMMARIZER_CACHE = None

MAX_SUMM_CHARS = 2000      # 摘要輸入最多保留多少字元
TARGET_SUMM_LEN = 128      # 摘要輸出長度上限
MIN_SUMM_LEN = 40          # 摘要輸出長度下限
FINANCIAL_TERMS = [
    "revenue", "sales", "net income", "operating income", "operating profit",
    "gross profit", "gross margin", "operating margin", "ebitda", "ebit",
    "eps", "earnings per share", "cash flow", "free cash flow", "dividend",
    "guidance", "forecast"
]

# 常見的非關鍵首字大寫字，避免將疑問詞等視為公司名
COMMON_CAP_STOPWORDS = {
    "what", "which", "when", "where", "who", "why", "how",
    "is", "are", "was", "were", "will", "do", "does", "did",
    "in", "on", "at", "for", "with", "from", "to", "of", "the", "a", "an"
}

# 清洗文本
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)  # 移除 HTML tags
    text = text.replace("\n", " ")       # 換行 → 空白
    text = re.sub(r"\s+", " ", text)     # 多空白合併
    text = re.sub(r"[^\x00-\x7F]+", " ", text) 
    return text.strip()

# 10K / 10M / 3B 轉換
def convert_money_unit(token: str):
    unit_map = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    match = re.match(r"(\d+(?:\.\d+)?)([KMB])", token)
    if not match:
        return None
    number, unit = match.groups()
    return float(number) * unit_map[unit]

# 日期抽取
def extract_dates(text: str):
    pattern = r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b"
    return re.findall(pattern, text)

# K/M/B 抽取
def extract_money_units(text: str):
    pattern = r"\b\d+(?:\.\d+)?[KMB]\b"
    return re.findall(pattern, text)

# 一般數字
def extract_plain_numbers(text: str, dates, numbers_raw):
    t = text
    for d in dates:
        t = t.replace(d, " ")
    for m in numbers_raw:
        t = t.replace(m, " ")
    t = re.sub(r"(\d),(?=\d{3})", r"\1", t)
    pattern = r"\d+(?:\.\d+)?"
    return re.findall(pattern, t)

# 數字整合
def extract_all_numbers(clean_t: str):
    dates = extract_dates(clean_t)
    numbers_raw = extract_money_units(clean_t)

    numbers_value = []
    for token in numbers_raw:
        v = convert_money_unit(token)
        if v is not None:
            numbers_value.append(v)

    plain_numbers = extract_plain_numbers(clean_t, dates, numbers_raw)
    return dates, numbers_raw, numbers_value, plain_numbers

def extract_keywords(text: str) -> List[str]:
    if not text:
        return []

    keywords = []
    seen = set()
    lower_text = text.lower()

    def add_kw(token: str):
        key = token.strip().lower()
        if key and key not in seen:
            seen.add(key)
            keywords.append(key)

    # 股票代碼或全大寫縮寫
    for ticker in re.findall(r"\b[A-Z]{2,5}\b", text):
        add_kw(ticker)

    # 可能的公司名稱 (多個首字大寫的詞組)
    for phrase in re.findall(r"\b(?:[A-Z][a-zA-Z&]+(?:\s+[A-Z][a-zA-Z&]+)+)\b", text):
        add_kw(phrase)

    # 單個首字大寫詞
    for token in re.findall(r"\b[A-Z][a-zA-Z&]+\b", text):
        if token.lower() not in COMMON_CAP_STOPWORDS:
            add_kw(token)

    # 年份
    for year in re.findall(r"\b(?:19|20)\d{2}\b", text):
        add_kw(year)

    # 財務術語
    for term in FINANCIAL_TERMS:
        if term in lower_text:
            add_kw(term)

    return keywords

# chunk 切割
def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += (chunk_size - overlap)
    return chunks

# MultiHiertt 表格 → 轉文字
def flatten_multihiertt_table(table: Dict[str, Any]) -> str:
    """
    將 MultiHiertt 的階層式表格展平：
    header = ["Category", "Item", "Value"]
    rows = [["Electronics","", ""], ["","iPhone","1200"]]
    會變成：
    Category: Electronics
    Item: iPhone, Value: 1200
    """
    header = table.get("header", [])
    rows = table.get("rows", [])

    lines = []
    for row in rows:
        pairs = []
        for h, v in zip(header, row):
            if v and str(v).strip():     # 避免空字串
                pairs.append(f"{h}: {v}")
        if pairs:
            lines.append(", ".join(pairs))
    return "\n".join(lines)

def build_summarization_input(title: str, text: str) -> str:
    """
    將 title + text 合併後做基本截斷，避免太長。
    """
    t = " ".join([x for x in [title, text] if x]).strip()
    return t[:MAX_SUMM_CHARS]


def get_summarizer() -> Optional[Any]:
    """
    懶加載摘要模型；自動偵測 GPU，找不到就用 CPU。
    """
    global _SUMMARIZER_CACHE

    if _SUMMARIZER_CACHE is not None:
        return _SUMMARIZER_CACHE

    if not USE_SUMMARIZATION:
        return None

    try:
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMM_MODEL)
        _SUMMARIZER_CACHE = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
    except Exception as exc:
        print(f"[WARN] Summarizer init failed, skip summarization: {exc}")
        _SUMMARIZER_CACHE = None
    return _SUMMARIZER_CACHE


def summarize_for_task(task_name: str, doc_id: str, title: str, text: str) -> str:
    """
    只在指定的 task 上做 summarization。
    失敗時回傳空字串，不影響整個前處理流程。
    """
    if (not USE_SUMMARIZATION) or (task_name not in SUMM_TASKS):
        return ""

    full = build_summarization_input(title, text)
    if not full:
        return ""

    summarizer = get_summarizer()
    if summarizer is None:
        return ""

    try:
        out = summarizer(
            full,
            max_length=TARGET_SUMM_LEN,
            min_length=MIN_SUMM_LEN,
            do_sample=False,
            truncation=True,
        )
        summ = out[0]["summary_text"]
        return clean_text(summ)
    except Exception as e:
        print(f"[WARN] Summarization failed for {task_name} doc {doc_id}: {e}")
        return ""

# 整合 MultiHiertt 表格的 preprocess_task
def preprocess_task(task_name: str, task_obj: Any):

    for doc_id, doc in task_obj.corpus.items():

        text = doc.get("text", "")
        title = doc.get("title", "")

        clean_t = clean_text(text)
        clean_title = clean_text(title)

        if task_name == "MultiHiertt" and "table" in doc:
            table_text = flatten_multihiertt_table(doc["table"])
            doc["table_text"] = table_text

            # 把表格文字併入主文本，讓 chunk 與 embedding 都能用到
            clean_t = clean_t + "\n" + table_text

        # 數字抽取
        dates, numbers_raw, numbers_value, plain_numbers = extract_all_numbers(clean_t)
        numbers_all = dates + numbers_raw + plain_numbers
        # 🔹 新增：只在 FinanceBench / FinQABench 做 summarization
        summary_text = summarize_for_task(task_name, doc_id, clean_title, clean_t)
        if summary_text:
            # 存一份摘要版本，後續 embedding 想用就用這個欄位
            doc["summary_text"] = summary_text
            # 也可以幫摘要切 chunk，方便後面選擇用 summary_chunks
            doc["summary_chunks"] = chunk_text(summary_text, chunk_size=220, overlap=40)
        else:
            # 沒做摘要或失敗，就不要放這兩個欄位
            doc["summary_text"] = None
            doc["summary_chunks"] = []

        keywords = extract_keywords(f"{clean_title} {clean_t}")

        # chunk 切段
        chunks = chunk_text(clean_t, chunk_size=220, overlap=40)

        # 寫回 corpus
        doc["text"] = clean_t
        doc["title"] = clean_title
        # 摘要在前、原文在後，讓 embedding / rerank 都能使用摘要版本
        doc["chunks"] = (doc["summary_chunks"] or []) + chunks

        doc["dates"] = dates
        doc["numbers_raw"] = numbers_raw
        doc["numbers_value"] = numbers_value
        doc["numbers_plain"] = plain_numbers
        doc["numbers_all"] = numbers_all
        doc["keywords"] = keywords

    # Debug 第一筆
    first_doc_id = next(iter(task_obj.corpus))
    print(f"\n Debug {task_name} Document Sample")
    print(task_obj.corpus[first_doc_id])

    # 處理 queries
    new_queries = {}

    for qid, q_text in task_obj.queries.items():
        clean_q = clean_text(q_text)

        q_dates, q_numbers_raw, q_numbers_value, q_plain = extract_all_numbers(clean_q)
        q_numbers_all = q_dates + q_numbers_raw + q_plain
        q_keywords = extract_keywords(clean_q)

        new_queries[qid] = {
            "text": clean_q,
            "dates": q_dates,
            "numbers_raw": q_numbers_raw,
            "numbers_value": q_numbers_value,
            "numbers_plain": q_plain,
            "numbers_all": q_numbers_all,
            "keywords": q_keywords,
        }

    task_obj.queries = new_queries

    print(f"✔ {task_name} 前處理完成！")
