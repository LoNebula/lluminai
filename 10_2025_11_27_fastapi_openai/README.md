# 🚀 Learning-based LLM Model Router (FastAPI × LightGBM × Streamlit)

**学習可能な LLM モデルルーター（Learning-based Model Router）** のサンプル実装です。

- FastAPI → 推論API  
- LightGBM → モデル選択ルーター  
- SQLite (logs.db) → 推論ログ蓄積  
- Streamlit → 推論UI  
- `python main.py --train` → 推論ログからルーターを学習  
- router_model.pkl → 学習済みルーター

プロンプトの特徴（タスク種別・文字数・コード含有・数式含有など）を学習し、  
**複数LLMの中から最適なモデルを自動選択**できる仕組みを構築しています。

---

# ✨ Features

- 🔍 **FeatureExtractor**  
  - コード含有 / 数式含有 / プロンプト長 / タスク種別  
- 🔄 **ModelRouter（LightGBM）**  
  - 「if 文ルーター」から「学習可能ルーター」へ進化  
- 🧠 **Active Learning Loop**  
  - 推論ログ → 再学習 → ルーターが賢くなる  
- 🗄️ SQLite (logs.db) 自動生成  
- 🌐 FastAPI で推論サービス  
- 🖥️ Streamlit UI で簡易クライアント  

---

# 🏗 Architecture

```mermaid
graph TD;

A[Streamlit UI] --> B[FastAPI /inference];
B --> C[FeatureExtractor];
C --> D[ModelRouterML];
D -->|model selected| E[OpenAI API];
B --> F[logs.db];
F --> G[LGBM Trainer --train];
G --> H[router_model.pkl];
H --> D;
````

---

# 📦 Requirements

* Python 3.9+
* FastAPI
* uvicorn
* lightgbm
* scikit-learn
* pandas
* joblib
* requests
* streamlit
* openai

```bash
pip install -r requirements.txt
```

（必要なら requirements.txt を生成可能）

---

# 🚀 Run FastAPI

```bash
python main.py
```

起動ログ：

```
🔥 Starting FastAPI server...
📦 logs.db Ready.
```

API docs → [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

# 🖥️ Run Streamlit

```bash
streamlit run streamlit_ui.py
```

ブラウザが開きます：

* タスク選択（chat / summarize / classify / reasoning）
* プロンプト入力
* 実行するとモデル選択 + 推論結果が表示されます

---

# 🧪 Train Router (LightGBM)

推論ログがたまったら以下を実行：

```bash
python main.py --train
```

成功すると：

```
🔍 Loading data...
⚙️ Training LightGBM...
🎉 Router model saved: router_model.pkl
🎉 Training complete!
```

次回 FastAPI 起動時に：

```
📦 ML Router loaded: router_model.pkl
```

と表示され、
**学習済みルーターが実際に動作し始めます。**

---

# 📁 Directory Structure

```
.
├── main.py           # FastAPI + Router + Trainer
├── streamlit_ui.py   # Web UI
├── logs.db           # 推論ログ（自動生成）
├── router_model.pkl  # 学習済みルーター（学習後に生成）
└── README.md
```

---

# 🧩 Code

## main.py

> FastAPI / Model Router / LightGBM Trainer
> → ****

## streamlit_ui.py

> Streamlit UI クライアント
> → ****

---

# 🔍 How Routing Works

### 1. 特徴抽出

* prompt_length
* contains_code
* contains_math
* task (chat/summarize/classify/reasoning)

### 2. Fallback Router（初期状態）

データ不足なら if 文で選択：

```python
if task == "summarize":
    return "gpt-4o-mini"
if task == "reasoning":
    return "o1"
```

### 3. 学習済み Router

LightGBM + LabelEncoder で分類器化：

```python
task → task_encoded
best_model → best_model_encoded
```

推論ごとに：

```
特徴量 → LightGBM → 最適モデル
```

---

# 🔥 Example Routing Result

| 入力タスク     | 内容         | ルーター出力      |
| --------- | ---------- | ----------- |
| chat      | Hi!        | gpt-4o-mini |
| summarize | 長文要約       | gpt-4o      |
| reasoning | 三段論法       | o1          |
| code      | Pythonデバッグ | gpt-4.1     |

---

# 🧩 Notes

* LightGBM の Warning はデータが少ないときの仕様
  （正常動作、気にしなくてOK）
* SQLite は絶対パス指定で確実に動くように設定済み
* router_model.pkl は joblib で保存されます

---

# 📈 Future Improvements

* 特徴量の追加（embedding / complexity / token ratio）
* Multi-Provider Routing

  * OpenAI, Anthropic, Google Gemini を統合
* 強化学習によるコスト最適化ルーター
* SHAP による判断根拠の可視化
* バッチ学習 + モデル更新スケジューラ

