# 🧠 LangGraph × Supabase × Ollama で作る「永続記憶 AI 秘書」

このリポジトリは、**LangGraph × Supabase × Ollama** を組み合わせて構築した  
「**永続記憶を持つ AI 秘書エージェント**」の実装です。

会話の中から AI が自動的に **覚えるべき情報を抽出し、Supabase に永続記憶として保存**。  
次回以降の会話でその情報を参照しながら回答します。

実行環境はすべて **ローカル**（Ollama）で完結するため：

- APIコストゼロ  
- プライバシー安全  
- 応答が高速（1秒台）  

というメリットがあります。

---

# 🚀 Features

- 🔹 **LangGraph による状態管理（短期記憶）**
- 🔹 **Supabase（Postgres + pgvector）による長期記憶**
- 🔹 **Ollama でローカル LLM 推論**
- 🔹 AI が自ら「覚えるべき情報」を判断（MEM_TO_SAVEプロトコル）
- 🔹 Embedding 生成もローカル（nomic-embed-text）

---

# 📁 ファイル構成

```

project/
│── agent.py         # LangGraph ノード（retrieve & think）
│── app.py           # アプリ実行（エージェント起動）
│── config.py        # Supabase 読み込み
│── memory.py        # Embedding / 検索 / 追加
│── .env             # Supabase URL / KEY
└── requirements.txt

```

---

# 🔧 事前準備

## 1. 📦 Python ライブラリをインストール

```

pip install -r requirements.txt

```

## 2. 🐘 Supabase を準備

1. Supabase プロジェクトを作成  
2. SQL Editor に以下を貼って実行（テーブル + RPC 関数）:

```sql
create extension if not exists vector;

create table memory_records (
  id uuid primary key default gen_random_uuid(),
  user_id text,
  content text,
  embedding vector(768),
  created_at timestamptz default now()
);

create or replace function match_memory (
  query_embedding vector(768),
  match_threshold float,
  match_count int
)
returns table (
  id uuid,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    id,
    content,
    1 - (memory_records.embedding <=> query_embedding) as similarity
  from memory_records
  where 1 - (memory_records.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
end;
$$;
```

## 3. 🔑 `.env` を準備

```
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-service-role-key
```

> ※ anon-key ではなく **service_role** が必要です（INSERT するため）

## 4. 🤖 Ollama モデルを取得

```
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

---

# ▶️ 実行方法

`src/app.py` を実行：

```
python app.py
```

初期状態では以下の入力が送信されます：

```python
"明日は何する日？？データを参照して答えて．"
```

出力は次のようになります：

```
明日は〇〇をする予定です。（保存されている記憶に基づいて回答）
```

---

# 🧩 主要コード

## `agent.py`

LangGraph の **retrieve（記憶検索）** と **think（推論）** を定義


## `app.py`

エージェントを起動し、状態付き実行をします


## `config.py`

Supabase の URL / KEY を `.env` から読み込む


## `memory.py`

* Embedding
* 記憶追加
* 類似検索（pgvector + RPC）
  を担当


## `requirements.txt`

依存パッケージ


---

# 🧠 アーキテクチャ概要

```
User Input
     │
     ▼
[LangGraph State] ——（短期記憶）
     │
     ├─ retrieve_memory() → Supabase（長期記憶）から類似記憶を取得
     ▼
 think() → LLM（Ollama）
     │
     ├─ 必要なら MEM_TO_SAVE: を抽出し Supabase へ保存
     ▼
 Assistant Response
```