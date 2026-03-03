import streamlit as st
import cv2
import base64
import requests
import tempfile
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# --- ページ設定 ---
st.set_page_config(page_title="AI Video Director", page_icon="🎬", layout="wide")
st.title("🎬 AI Video Director")
st.markdown("動画をアップロードすると、VLM（視覚言語モデル）が映像監督の視点で**カメラワーク・ライティング・演出意図**を逆コンパイルします。")

# --- サイドバー（設定） ---
with st.sidebar:
    st.header("⚙️ 設定 (Settings)")
    api_key = st.text_input("OpenRouter API Key", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
    model_choice = st.selectbox("使用モデル", ["google/gemini-3.1-pro-preview", "qwen/qwen3.5-35b-a3b", "bytedance-seed/seed-2.0-mini"])
    num_frames = st.slider("抽出フレーム数", min_value=3, max_value=10, value=5, help="抽出枚数が多いほど精度は上がりますが、APIコストと時間がかかります。")

# --- メイン処理関数 ---
def extract_frames(video_path, num_frames):
    """動画から等間隔でフレームを抽出する"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []

    interval = max(1, total_frames // num_frames)
    frames_base64 = []
    frames_images = [] # UI表示用

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(i * interval, total_frames - 1))
        ret, frame = cap.read()
        if ret:
            # 処理を軽くするためリサイズ（RGB変換も行う）
            frame_resized = cv2.resize(frame, (640, 360))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_images.append(frame_rgb)
            
            # Base64エンコード
            _, buffer = cv2.imencode('.jpg', frame_resized)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            frames_base64.append(b64_str)
            
    cap.release()
    return frames_base64, frames_images

def analyze_with_vlm(frames_base64, api_key, model):
    """OpenRouter経由でVLMに解析を依頼する"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt_text = """
    提供された画像は動画の連続したフレームです。あなたはプロの映像監督です。
    以下の項目について、具体的なシーンの状況を交えながらMarkdown形式で詳細に分析・解説してください。
    
    ### 🎥 1. カメラワークと構図
    ### 💡 2. ライティングと色彩設計
    ### 🎭 3. 登場人物の感情・状況の推移
    ### 🎬 4. 総合的な演出意図（なぜこのような見せ方をしたのか）
    """
    
    content = [{"type": "text", "text": prompt_text}]
    for b64 in frames_base64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}]
    }
    
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

# --- UIレイアウト ---
uploaded_file = st.file_uploader("動画ファイルをアップロード (mp4, mov等)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 左右の2カラムレイアウト
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📼 元動画")
        st.video(uploaded_file)
        
    if st.button("🎬 映像を解析する", type="primary"):
        if not api_key:
            st.error("左のサイドバーからOpenRouter API Keyを入力してください。")
        else:
            with st.status("AIディレクターが映像を解析中...", expanded=True) as status:
                try:
                    # 1. 一時ファイルとして保存（OpenCVで読み込むため）
                    st.write("動画を読み込んでいます...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                        tfile.write(uploaded_file.read())
                        temp_video_path = tfile.name
                    
                    # 2. フレーム抽出
                    st.write("キーフレームを抽出中...")
                    frames_b64, frames_img = extract_frames(temp_video_path, num_frames)
                    
                    # UIに抽出したフレームを表示
                    st.write("抽出されたフレーム:")
                    cols = st.columns(len(frames_img))
                    for idx, img in enumerate(frames_img):
                        cols[idx].image(img, use_column_width=True)
                    
                    # 3. VLM解析
                    st.write("VLM（AI）に演出意図を質問中...")
                    analysis_result = analyze_with_vlm(frames_b64, api_key, model_choice)
                    
                    status.update(label="解析完了！", state="complete", expanded=False)
                    
                    # 一時ファイルを削除
                    os.remove(temp_video_path)
                    
                    # 結果表示
                    with col2:
                        st.subheader("📝 演出の解析結果")
                        st.markdown(analysis_result)

                except Exception as e:
                    status.update(label="エラーが発生しました", state="error")
                    st.error(f"詳細: {e}")