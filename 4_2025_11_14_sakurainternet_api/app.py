import streamlit as st
from sakura_client import sakura_chat

st.set_page_config(page_title="Sakura AI Chat", page_icon="💬")

st.title("💬 Sakura AI × Streamlit — 国内LLMチャットアプリ")

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。"}
    ]

# --- 入力欄 ---
user_input = st.text_input("質問を入力してください")

if st.button("送信", type="primary"):
    if user_input.strip():
        st.session_state.history.append({"role": "user", "content": user_input})

        response = sakura_chat(st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": response})
    else:
        st.warning("テキストを入力してください！")

# --- チャットの描画 ---
for message in reversed(st.session_state.history):
    role = message["role"]
    if role == "user":
        st.chat_message("user").write(message["content"])
    elif role == "assistant":
        st.chat_message("assistant").write(message["content"])

# 履歴クリア
if st.button("履歴をクリア"):
    st.session_state.history = [
        {"role": "system", "content": "あなたは親切なAIアシスタントです。"}
    ]