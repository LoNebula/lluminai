from typing import List, Dict, Any, Optional
import uuid

import chromadb
from sentence_transformers import SentenceTransformer
import requests

# ================================
# 1. Markdown → chunk 分割
# ================================

def split_markdown(md: str) -> List[Dict[str, str]]:
    """
    Markdown文字列を「タイトル＋本文」のチャンクに分割する。
    見出し行（#, ##, ###）をタイトルとみなし、それ以外を本文としてまとめる。
    """
    lines = md.splitlines()
    chunks: List[Dict[str, str]] = []
    buffer: List[str] = []
    current_title = ""

    def flush():
        nonlocal buffer, current_title
        if buffer:
            content = "\n".join(buffer).strip()
            if content:
                chunks.append({
                    "title": current_title,
                    "content": content
                })
            buffer = []

    for line in lines:
        if line.startswith("#"):  # 見出し
            flush()
            current_title = line.lstrip("#").strip()
        else:
            buffer.append(line)

    flush()
    return chunks


# ================================
# 2. Embedding モデル（bge-m3）
# ================================

# NOTE: 初回ロードは数秒かかります
_EMBEDDER = SentenceTransformer("BAAI/bge-m3")

def embed_text(text: str):
    """テキストをベクトル（list[float]）に変換"""
    return _EMBEDDER.encode(text, normalize_embeddings=True)


# ================================
# 3. ChromaDB セットアップ
# ================================

_client = chromadb.Client()
_collection = _client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)


def index_markdown(md: str, source_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Markdownをチャンク化してChromaDBに登録する。
    - source_id が与えられなければ自動生成する。
    戻り値: {"source_id": str, "num_chunks": int}
    """
    if source_id is None or not source_id.strip():
        source_id = f"doc_{uuid.uuid4().hex[:8]}"

    chunks = split_markdown(md)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, ch in enumerate(chunks):
        content = ch["content"].strip()
        if not content:
            continue

        emb = embed_text(content)
        ids.append(f"{source_id}_{i}")
        embeddings.append(emb)
        documents.append(content)
        metadatas.append({
            "title": ch["title"],
            "source": source_id,
        })

    if ids:
        _collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    return {"source_id": source_id, "num_chunks": len(ids)}


def search(query: str, k: int = 5) -> Dict[str, Any]:
    """
    ユーザクエリを埋め込み、近傍チャンクを検索する。
    ChromaDB の生の query 結果を返す。
    """
    q_emb = embed_text(query)
    results = _collection.query(
        query_embeddings=[q_emb],
        n_results=k,
    )
    return results


# ================================
# 4. LLM（Ollama）で回答生成
# ================================

OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL_NAME = "gpt-oss:20b"  # ← 好きなテキストモデル名に変更してください


def call_llm(prompt: str) -> str:
    """
    Ollama の /api/generate を叩いてテキストを生成する。
    """
    payload = {
        "model": LLM_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(OLLAMA_API_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def answer_with_context(query: str, k: int = 5) -> Dict[str, Any]:
    """
    - クエリで ChromaDB から上位k件を取得
    - それらをコンテキストとして LLM に投げる
    - 回答と、参照したコンテキストをまとめて返す
    """
    ctx = search(query, k=k)
    docs = ctx.get("documents", [[]])[0]
    metadatas = ctx.get("metadatas", [[]])[0]

    # コンテキストをLLMに渡すためのテキストに整形
    context_block = ""
    for i, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
        title = meta.get("title") or ""
        source = meta.get("source") or ""
        header = f"[{i}] source={source} title={title}".strip()
        context_block += f"{header}\n{doc}\n\n"

    prompt = f"""
あなたはRAGシステムの回答エンジンです。
ユーザーの質問に対して、以下のコンテキストだけを根拠にして日本語で回答してください。
わからない場合は「手元のナレッジだけでは確実な回答ができません」と答えてください。

# 質問
{query}

# コンテキスト
{context_block}
"""

    answer = call_llm(prompt)

    return {
        "answer": answer,
        "contexts": {
            "documents": docs,
            "metadatas": metadatas,
        },
    }