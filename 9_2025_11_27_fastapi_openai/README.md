# 🚀 LLM モデル自動切替 API & UI  
FastAPI × OpenAI × Streamlit

このリポジトリは、**タスク種別に応じて最適な OpenAI モデルを自動選択して推論を行う API** と、その API を操作するための **Streamlit UI** を提供します。

目的は、プロダクト開発における  
「どの LLM モデルを使うべきか問題」を API 側で吸収し、  
クライアントは単に “task” を指定するだけで最適モデルが自動選ばれるようにすることです。

---

# ✨ 特徴

- **FastAPI** による軽量＆高速な推論 API
- **ModelSelector** によるタスク別の LLM 自動切替  
  （chat / summarize / classify / reasoning）
- **OpenAI Responses API** に対応（最新仕様）
- **Streamlit UI** で直感的に操作可能
- 拡張しやすいアーキテクチャ（Claude/Gemini対応も容易）

---

# 🖼️ UI スクリーンショット

以下のような UI からプロンプトを入力し、  
モデルが自動選ばれて推論結果が返ってきます。

（※ GitHub にアップロード後、画像パスを貼り替えてください）

![LLM モデル自動切替 UIのスクリーンショット](./screenshot.png)

---

# 📁 ディレクトリ構成

```

project/
├── main.py            # FastAPI: API本体
├── selector.py        # モデル切替ロジック
├── services.py        # OpenAI呼び出し（Responses API）
├── streamlit_app.py   # Streamlit UI
├── .env               # OPENAI_API_KEY を格納
└── README.md

````

---

# 🔧 セットアップ

## 1. リポジトリをクローン

```bash
git clone https://github.com/yourname/yourrepo.git
cd yourrepo
````

## 2. パッケージをインストール

```bash
pip install -r requirements.txt
```

（requirements.txt が無い場合）

```bash
pip install fastapi uvicorn openai python-dotenv streamlit requests
```

---

# 🔑 .env を作成

`OPENAI_API_KEY` を設定します。

```env
OPENAI_API_KEY=sk-********************************
```

---

# 🚀 起動方法

## 1. FastAPI（バックエンド）

```bash
uvicorn main:app --reload
```

成功すると：

```
http://127.0.0.1:8000
```

で動作します。

---

## 2. Streamlit（フロントエンド）

別ターミナルで：

```bash
streamlit run streamlit_app.py
```

`http://localhost:8501` を開くと UI が表示されます。

---

# ⚙️ API 仕様

### POST `/inference`

FastAPI 本体は `main.py` に定義されています。
API は task と prompt を受け取ります。

（参照：`main.py`）

#### リクエスト（JSON）

```json
{
  "task": "summarize",
  "prompt": "Explain quantum computing in simple terms."
}
```

#### レスポンス（JSON）

```json
{
  "model_used": "gpt-4o-mini",
  "output": "Quantum computing is..."
}
```

---

# 🧠 モデル自動切替の仕組み

自動切替は `selector.py` の `ModelSelector` が担当します。
（参照：`selector.py`）

```python
class ModelSelector:
    def choose(self, task: TaskType):
        if task in (TaskType.CLASSIFY, TaskType.SUMMARIZE):
            return "gpt-4o-mini"

        if task == TaskType.CHAT:
            return "gpt-4o"

        if task == TaskType.REASONING:
            return "o1"

        return "gpt-4o"
```

タスク名（文字列→Enum）が入るだけで
最適なモデルが自動的に割り当てられます。

---

# 🤖 OpenAI Responses API の使用

推論処理は `services.py` に実装されています。
（参照：`services.py`）

```python
response = client.responses.create(
    model=model,
    input=prompt
)

return {
    "model_used": model,
    "output": response.output_text
}
```

OpenAI の最新API（Responses API）を使用しているため、
`chat.completions` より安定的で将来仕様にも強い構成になっています。

---

# 🎨 Streamlit UI

UI は `streamlit_app.py` に実装されています。
（参照：`streamlit_app.py`）

* task のセレクトボックス
* プロンプト入力欄
* 実行ボタン
* 自動選ばれたモデルの表示
* 出力結果のレンダリング

直感的に API をテストできます。
