import os
import re
import sqlite3
import joblib
import pandas as pd
import lightgbm as lgb
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import dotenv
from sklearn.preprocessing import LabelEncoder

dotenv.load_dotenv()

###############################################################
# 0. パス設定（絶対パス）
###############################################################
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "logs.db")
MODEL_PATH = os.path.join(BASE_DIR, "router_model.pkl")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

print("🔥 Using logs.db at:", DB_PATH)
print("🔥 Using router_model.pkl at:", MODEL_PATH)

###############################################################
# 1. DB 初期化
###############################################################
def init_db():
    print("💾 init_db() called")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT,
            prompt_length INTEGER,
            contains_code INTEGER,
            contains_math INTEGER,
            used_model TEXT,
            best_model TEXT,
            latency_ms REAL,
            cost REAL
        );
    """)
    conn.commit()
    conn.close()
    print("📦 logs.db Ready.")
    print("💾 DB PATH:", DB_PATH)

init_db()

###############################################################
# 2. 特徴抽出器
###############################################################
class FeatureExtractor:
    def contains_code(self, text):
        return int(bool(re.search(r"```|class |def |\{.*\}", text)))

    def contains_math(self, text):
        return int(bool(re.search(r"\$.*\$|\d+ \+ \d+", text)))

    def extract(self, task: str, prompt: str):
        return {
            "task": task,
            "prompt_length": len(prompt),
            "contains_code": self.contains_code(prompt),
            "contains_math": self.contains_math(prompt),
        }

###############################################################
# 3. モデルルーター（ML or Fallback）
###############################################################
class ModelRouterML:
    def __init__(self):
        self.extractor = FeatureExtractor()

        if os.path.exists(MODEL_PATH):
            print("📦 ML Router loaded:", MODEL_PATH)
            data = joblib.load(MODEL_PATH)
            self.model = data["model"]
            self.le_task = data["le_task"]
            self.le_model = data["le_model"]
            self.use_ml = True
        else:
            print("⚠️ No ML Router found → Using fallback rules.")
            self.use_ml = False

    def choose(self, task, prompt):
        if not self.use_ml:
            return self.fallback_rule(task, prompt)

        features = self.extractor.extract(task, prompt)

        # LightGBM が使う task_encoded に変換
        task_encoded = self.le_task.transform([task])[0]

        df = pd.DataFrame([{
            "task_encoded": task_encoded,
            "prompt_length": features["prompt_length"],
            "contains_code": features["contains_code"],
            "contains_math": features["contains_math"],
        }])

        pred_encoded = self.model.predict(df)[0]
        prob = max(self.model.predict_proba(df)[0])

        # best_model をラベルから元に戻す
        pred_label = self.le_model.inverse_transform([pred_encoded])[0]

        return pred_label, float(prob)

###############################################################
# 4. 推論サービス（安全レスポンス）
###############################################################
class InferenceService:
    def __init__(self, router):
        self.router = router

    async def run(self, task, prompt):
        try:
            start = time.time()
            model, conf = self.router.choose(task, prompt)

            response = client.responses.create(
                model=model,
                input=prompt
            )

            latency = (time.time() - start) * 1000
            output = getattr(response, "output_text", str(response))

            self.log(task, prompt, model, latency, cost=0.0)

            return {
                "model_used": model,
                "confidence": conf,
                "latency_ms": latency,
                "output": output,
            }

        except Exception as e:
            return {
                "model_used": None,
                "confidence": 0.0,
                "latency_ms": 0,
                "output": f"Error: {str(e)}",
            }

    def log(self, task, prompt, model, latency, cost):
        fe = self.router.extractor.extract(task, prompt)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO logs (task, prompt_length, contains_code, contains_math, used_model, best_model, latency_ms, cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task,
            fe["prompt_length"],
            fe["contains_code"],
            fe["contains_math"],
            model,
            model,
            latency,
            cost
        ))
        conn.commit()
        conn.close()

###############################################################
# 5. Router Trainer（LightGBM）
###############################################################
def train_router():
    print("🔍 Loading data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()

    if df.empty:
        print("❌ No logs found. Cannot train.")
        return

    print(df.head())

    # -----------------------------------------
    # 1. カテゴリ列（task, best_model）のラベルエンコード
    # -----------------------------------------
    le_task = LabelEncoder()
    le_model = LabelEncoder()

    df["task_encoded"] = le_task.fit_transform(df["task"])
    df["best_model_encoded"] = le_model.fit_transform(df["best_model"])

    # -----------------------------------------
    # 2. 数値特徴量に限定
    # -----------------------------------------
    X = df[[
        "task_encoded",
        "prompt_length",
        "contains_code",
        "contains_math"
    ]]

    y = df["best_model_encoded"]

    print("⚙️ Training LightGBM...")

    model = lgb.LGBMClassifier()
    model.fit(X, y)

    # -----------------------------------------
    # 3. 保存（モデル＋エンコーダ両方）
    # -----------------------------------------
    joblib.dump({
        "model": model,
        "le_task": le_task,
        "le_model": le_model
    }, MODEL_PATH)

    print("🎉 Router model saved:", MODEL_PATH)

###############################################################
# 6. FastAPI
###############################################################
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

router = ModelRouterML()
service = InferenceService(router)

class RequestBody(BaseModel):
    task: str
    prompt: str

@app.post("/inference")
async def inference(req: RequestBody):
    return await service.run(req.task, req.prompt)

###############################################################
# 7. Main（--train / 通常）
###############################################################
if __name__ == "__main__":
    import sys

    if "--train" in sys.argv:
        print("🔄 Training router model...")
        train_router()
        print("🎉 Training complete!")
        sys.exit(0)

    print("🔥 Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
