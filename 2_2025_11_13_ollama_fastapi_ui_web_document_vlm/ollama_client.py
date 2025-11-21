import base64
import requests
from pathlib import Path

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llava:13b"

def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_vlm_prompt() -> str:
    return """
あなたはドキュメント画像を読み取り、その内容をMarkdownとして構造化するアシスタントです。

出力ルール:
- 見出しは `#`, `##`, `###`
- 箇条書きは `-`
- 表はMarkdownテーブル記法
- 余計な説明は不要、Markdownのみ
- 出力は必ず日本語で行う
"""

def analyze_image_with_ollama(image_path: Path) -> str:
    prompt = build_vlm_prompt()
    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False
    }

    resp = requests.post(OLLAMA_API_URL, json=payload)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()
