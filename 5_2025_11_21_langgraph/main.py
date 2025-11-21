import os
from typing import Optional, TypedDict, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import requests

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# =========================
# 0. 環境変数ロード
# =========================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY が .env に設定されていません。")
if not NOTION_TOKEN:
    raise RuntimeError("NOTION_TOKEN が .env に設定されていません。")

# Notion DB ID（.env）
NOTION_DB_MEETING = os.getenv("NOTION_DB_MEETING")
NOTION_DB_WORK = os.getenv("NOTION_DB_WORK")

# =========================
# 1. LangGraph State 定義
# =========================

class BotState(TypedDict):
    content: Optional[str]
    category: Optional[str]
    page_id: Optional[str]
    result_message: Optional[str]

# =========================
# 2. OpenRouter (gpt-oss:20b) 設定
# =========================

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# =========================
# 3. Notion API Utility 関数
# =========================

def fetch_page_list(database_id: str) -> List[Dict]:
    """特定DB内のページ一覧を取得して返す"""
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json={}, headers=headers)
    results = resp.json().get("results", [])

    pages = []
    for page in results:
        title_prop = page["properties"].get("Name", {}).get("title", [])
        title = title_prop[0]["plain_text"] if title_prop else "(無題)"
        pages.append({
            "title": title,
            "page_id": page["id"],
        })
    return pages


def fetch_page_content(page_id: str) -> str:
    """ページ本文を抽出してテキスト化"""
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
    }
    resp = requests.get(url, headers=headers)
    results = resp.json().get("results", [])

    texts = []
    for blk in results:
        if blk["type"] == "paragraph":
            for t in blk["paragraph"]["rich_text"]:
                texts.append(t.get("plain_text", ""))

    return "\n".join(texts)


# =========================
# 4. CLI：ページ選択機能
# =========================

def select_page_interactively(database_id: str) -> Dict:
    """DB内ページ一覧を CLI で選択 → page_id + content を返す"""
    print("\n=== Notion ページ一覧 ===")
    pages = fetch_page_list(database_id)

    if not pages:
        print("⚠️ ページがありません")
        return None

    for i, p in enumerate(pages):
        print(f"{i}: {p['title']}")

    idx = int(input("\n処理したいページ番号を入力してください: "))
    selected = pages[idx]

    page_id = selected["page_id"]
    content = fetch_page_content(page_id)

    print(f"\n▶ 選択ページ: {selected['title']} ({page_id})")
    print(f"内容:\n{content}")

    return {"page_id": page_id, "content": content}


# =========================
# 5. LangGraph ノード定義
# =========================

def classify_node(state: BotState) -> Dict:
    content = state.get("content", "")

    prompt = f"""
あなたは日本語テキストの分類器です。
次のメモを、以下のカテゴリのいずれか1つに分類してください。

候補カテゴリ:
- 会議メモ
- 仕事

出力はカテゴリ名のみ。

---
メモ本文:
{content}
---
"""
    res = llm.invoke(prompt)
    category = res.content.strip()

    allowed = ["会議メモ", "仕事"]
    if category not in allowed:
        category = "アイデア"

    print(f"[classify_node] → {category}")
    return {"category": category}


def update_notion_node(state: BotState) -> Dict:
    page_id = state.get("page_id")
    category = state.get("category", "アイデア")

    database_map = {
        "会議メモ": NOTION_DB_MEETING,
        "仕事": NOTION_DB_WORK
    }

    db_id = database_map.get(category)

    payload = {
        "parent": {"database_id": db_id},
        "properties": {
            "Name": {"title": [{"text": {"content": f"{category}｜Auto-Sorted"}}]}
        }
    }

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    resp = requests.patch(
        f"https://api.notion.com/v1/pages/{page_id}",
        json=payload,
        headers=headers,
    )

    if not resp.ok:
        raise RuntimeError(f"Notion API Error: {resp.text}")

    msg = f"Notion page {page_id} updated to {category}"
    print(f"[update_notion_node] {msg}")
    return {"result_message": msg}


# =========================
# 6. LangGraph グラフ構築
# =========================

workflow = StateGraph(BotState)
workflow.add_node("classify", classify_node)
workflow.add_node("update_notion", update_notion_node)
workflow.add_edge("classify", "update_notion")
workflow.set_entry_point("classify")
agent = workflow.compile()


# =========================
# 7. FastAPI Webhook
# =========================

app = FastAPI(title="LangGraph Notion Bot")

class WebhookPayload(BaseModel):
    page_id: str
    content: str

@app.post("/webhook")
def handle_webhook(payload: WebhookPayload):
    initial_state: BotState = {
        "page_id": payload.page_id,
        "content": payload.content,
        "category": None,
        "result_message": None,
    }
    result = agent.invoke(initial_state)
    return {"status": "ok", "result": result}


# =========================
# 8. CLI 実行
# =========================

def debug_run():

    print("\n=== DB を選択 ===")
    print("0: 会議メモDB")
    print("1: 仕事DB")

    choice = int(input("番号を入力: "))

    DB_MAP = {
        0: NOTION_DB_MEETING,
        1: NOTION_DB_WORK,
    }

    database_id = DB_MAP.get(choice)

    selected = select_page_interactively(database_id)

    # 🆕 ここが重要！！
    if not selected:
        print("❌ 選択されたDBにページがないため、処理を終了します。")
        return

    initial_state: BotState = {
        "page_id": selected["page_id"],
        "content": selected["content"],
        "category": None,
        "result_message": None,
    }

    print("\n=== LangGraph 実行 ===")
    result_state = agent.invoke(initial_state)

    print("\n=== 結果 ===")
    for k, v in result_state.items():
        print(f"{k}: {v}")



if __name__ == "__main__":
    debug_run()
