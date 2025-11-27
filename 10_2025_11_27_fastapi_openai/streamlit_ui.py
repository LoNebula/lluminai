import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/inference"

st.title("🚀 学習可能モデルルーター付き 推論UI")

task = st.selectbox("タスク：", ["chat", "summarize", "classify", "reasoning"])
prompt = st.text_area("プロンプトを入力", height=150)

if st.button("実行"):
    res = requests.post(API_URL, json={"task": task, "prompt": prompt}).json()

    model = res.get("model_used", "N/A")
    conf = res.get("confidence", 0.0)
    latency = res.get("latency_ms", 0)
    output = res.get("output", "No output")

    st.success(f"使用モデル：{model} (信頼度 {conf:.2f})")
    st.write(f"レイテンシ：{latency:.2f} ms")
    st.write("### 🔽 出力結果")
    st.write(output)
