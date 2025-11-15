import time
import tempfile
from pathlib import Path
import subprocess

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ollama_client import analyze_image_with_ollama
from rag_pipeline import index_markdown, answer_with_context

app = FastAPI(title="Multimodal RAG Pipeline")

# CORS (必要に応じて調整)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル (index.html, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")


def check_ollama_mode():
    """
    Ollama が GPU / CPU のどちらを使っているかをざっくり表示（任意）
    Windows のデフォルトパス例 C:\\ollama\\ollama.exe を想定
    """
    try:
        OLLAMA_CMD = r"C:\ollama\ollama.exe"

        result = subprocess.run(
            [OLLAMA_CMD, "list", "hardware"],
            capture_output=True,
            text=True,
        )
        out = result.stdout.lower()

        if "nvidia" in out or "cuda" in out:
            return "GPU"
        return "CPU"
    except Exception as e:
        print("⚠️ GPU/CPU 判定に失敗:", e)
        return "不明"


@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI started")
    mode = check_ollama_mode()
    print(f"💡 Ollama is using: {mode} mode")


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    メイン画面（ブラウザUI）
    """
    return Path("static/index.html").read_text(encoding="utf-8")


# ========== VLM: 画像 → Markdown → RAGインデックス ==========

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    画像を解析して Markdown を返しつつ、そのまま RAG にインデックスする。
    戻り値:
    - markdown: 生成されたMarkdown
    - exec_time_sec: VLM実行時間
    - source_id: ベクトルDBに保存したドキュメントID
    - num_chunks: 保存されたチャンク数
    """
    start_time = time.time()

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        # 1. VLMでMarkdown生成
        md = analyze_image_with_ollama(tmp_path)

        # 2. そのままRAGに投入（source_idには元ファイル名を使う）
        index_info = index_markdown(md, source_id=file.filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)

    exec_time = time.time() - start_time
    print(f"⏱️ VLM+Index Execution time: {exec_time:.2f} sec")

    return {
        "markdown": md,
        "exec_time_sec": exec_time,
        "source_id": index_info["source_id"],
        "num_chunks": index_info["num_chunks"],
    }


# ========== RAG: 質問 → 回答 ==========

class QueryBody(BaseModel):
    question: str
    top_k: int = 5


@app.post("/api/query")
async def query_rag(body: QueryBody):
    """
    RAG に質問するエンドポイント。
    - body.question: 質問文（日本語でOK）
    - body.top_k   : 取得するコンテキスト数（デフォルト5）

    戻り値:
    - answer: LLMによる最終回答
    - contexts: 参照したチャンク（documents, metadatas）
    """
    try:
        result = answer_with_context(body.question, k=body.top_k)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ========== 開発用: uvicorn から直接起動する場合 ==========

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app_full:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
