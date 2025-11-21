# LangGraph × Notion × OpenRouter  
## 自動分類＆ページ移動 Notion Bot（CLIページ選択式）

このリポジトリは  
**LangGraph × OpenRouter(gpt-oss:20b) × Notion API**  
を利用して構築した、

# 👉 **「Notionページを自動分類し、対応DBへ自動移動するBot」**

CLI で Notion DB のページ一覧を表示し、  
番号を選ぶだけで

1. ページ本文の取得  
2. LLM によるカテゴリ分類  
3. Notion DB 移動（自動整理）

がワンステップで実行されます。

---

# ✨ 特徴

### 🚀 LangGraph による状態管理フロー
- Node1: LLMでカテゴリ分類  
- Node2: Notion API でページの DB 移動  
- State: page_id / content / category / result_message を一元管理  

### 📒 Notionページを CLI で選択できる
- DB 内のページ一覧をタイトル付きで出力  
- 番号で選ぶだけで page_id + content を自動取得  
- 開発中のテストが爆速

### 🤖 OpenRouter（gpt-oss-20b）で分類
- 高速 & 安価  
- 「会議メモ / 仕事」の2分類に対応  
- プロンプト変更でカテゴリ拡張が簡単

---

# 📁 ファイル構成

```

main.py        # Bot の全処理（CLI / LangGraph / Notion API）

```

---

# 🛠️ セットアップ

## 1. リポジトリを clone

```bash
git clone https://github.com/<your-name>/<repo>.git
cd <repo>
```

## 2. 必要ライブラリのインストール

```bash
pip install langgraph langchain-openai fastapi uvicorn python-dotenv requests
```

## 3. `.env` を作成

以下をコピーして `.env` を作成：

```env
OPENROUTER_API_KEY=your_openrouter_api_key
NOTION_TOKEN=your_notion_api_token

# Notion DB ID（URLのID部分）
NOTION_DB_MEETING=xxxxxxxxxxxxxxxx
NOTION_DB_WORK=yyyyyyyyyyyyyyyyyy
```

Notion DB ID の取得方法：

```
https://www.notion.so/Workspace/<DATABASE_ID>?v=xxxxx
```

の `<DATABASE_ID>` 部分をコピー。

---

# 🚀 実行方法（CLI モード）

```bash
python main.py
```

すると以下のように CLI UI が表示されます：

```
=== DB を選択 ===
0: 会議メモDB
1: 仕事DB
番号を入力: 1
```

### DB 内のページ一覧が表示：

```
=== Notion ページ一覧 ===
0: 出張
1: コード編集
処理したいページ番号を入力してください: 0
```

### ページ内容が表示される：

```
▶ 選択ページ: 出張 (xxxxxx)
内容:
来週、カナダに出張．
```

### LangGraph が分類 → DB自動移動！

```
=== LangGraph 実行 ===
[classify_node] → 仕事
[update_notion_node] Notion page ... updated to 仕事

=== 結果 ===
content: 来週、カナダに出張．
category: 仕事
page_id: xxxxx
result_message: Notion page ... updated to 仕事
```

---

# 🧠 アーキテクチャ

```
CLI（ページ選択）
      ↓
fetch_page_list()
      ↓
fetch_page_content()
      ↓
LangGraph State
      ↓
[classify_node]  — LLM（OpenRouter gpt-oss:20b）
      ↓
[update_notion_node] — Notion API
      ↓
DB へ自動移動
```

---

# 📌 main.py の機能概要

### ✔ fetch_page_list()

Notion DB 内のページ一覧を取得し
`[{title, page_id}, ...]` の形式に変換。

### ✔ fetch_page_content()

ページの本文（paragraphブロック）を抽出 → テキストへ。

### ✔ select_page_interactively()

CLI でページ一覧を表示 → 番号入力 → `page_id + content` を返す。

### ✔ classify_node()

OpenRouter に “会議メモ / 仕事” のいずれかを分類させる。

### ✔ update_notion_node()

分類結果に応じて
`parent.database_id = (会議メモDB / 仕事DB)`
に切り替える = **Notion 上で実質「移動」**。

---

# 🌐 Webhook モード（FastAPI）

POST `/webhook` に以下の JSON を送ると分類→移動が動きます。

```json
{
  "page_id": "xxxx",
  "content": "メモ本文"
}
```

FastAPI としても使える設計。

---

# 🧩 カスタマイズ案

* 分類カテゴリを増やす（研究 / 日記 / アイデア…）
* 会議メモ→自動要約ノードを追加
* TODO抽出→タスクDBへ登録
* Slack / Gmail → Notion の同期
* Web UI（FastAPI + HTML）でページ選択をGUI化

---

# ⭐️ ぜひスターをお願いします！

Notion 自動整理 Bot が役立ったら、
GitHub の ⭐️ を押していただけると励みになります！
