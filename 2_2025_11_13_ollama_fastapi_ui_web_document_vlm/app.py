import time
import tempfile
from pathlib import Path
import subprocess

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ollama_client import analyze_image_with_ollama

app = FastAPI()

# 静的ファイル
app.mount("/static", StaticFiles(directory="static"), name="static")

def check_ollama_mode():
    try:
        OLLAMA_CMD = r"C:\ollama\ollama.exe"

        result = subprocess.run(
            [OLLAMA_CMD, "list", "hardware"],
            capture_output=True,
            text=True
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
    return Path("static/index.html").read_text(encoding="utf-8")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """画像を解析して Markdown を返す"""
    start_time = time.time()

    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        md = analyze_image_with_ollama(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        tmp_path.unlink(missing_ok=True)

    exec_time = time.time() - start_time
    print(f"⏱️ Execution time: {exec_time:.2f} sec")

    return {
        "markdown": md,
        "exec_time_sec": exec_time
    }
