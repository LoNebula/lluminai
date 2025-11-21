import os
import requests
from dotenv import load_dotenv

load_dotenv()

API = "https://api.ai.sakura.ad.jp/v1/chat/completions"
TOKEN = os.getenv("SAKURA_TOKEN")

if not TOKEN:
    raise RuntimeError("SAKURA_TOKENが設定されていません")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}

def sakura_chat(messages, model="gpt-oss-120b"):
    payload = {
        "model": model,
        "messages": messages
    }
    res = requests.post(API, headers=HEADERS, json=payload)
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"]