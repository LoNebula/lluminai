import os
import cv2
import time
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import google.generativeai as genai
from threading import Thread

# ==========================================
# ここにAPIキーを入れてね！
# ==========================================
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# モデルは爆速の "gemini-2.0-flash-exp" を指定
model = genai.GenerativeModel("gemini-2.0-flash-exp")

# AIへの指示書（システムプロンプト）
# ここを変えるとキャラ変できます。
SYSTEM_PROMPT = """
あなたは超ハイテンションな実況アナウンサーです。
目の前の映像を見て、何が起きているか、何が映っているかを
「短く」「テンポよく」「情熱的に」実況してください。
一文は20文字以内で！
"""

class RealTimeNarrator:
    def __init__(self):
        self.cap = cv2.VideoCapture(0) # Webカメラ起動
        self.latest_frame = None
        self.running = True
        self.narrative = "実況準備中..."
        # 日本語フォント設定（WindowsならMeiryoなど。MacならHiraginoなど適宜変更を！）
        # 見つからない場合はデフォルトフォントになります
        try:
            self.font = PIL.ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 32)
        except:
            self.font = PIL.ImageFont.load_default()

    def add_text_to_image(self, img_bgr, text):
        """OpenCV画像に日本語を描画するためのヘルパー関数"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)
        draw = PIL.ImageDraw.Draw(pil_img)
        
        # テキストの背景に黒帯を敷く
        draw.rectangle([(10, 10), (630, 60)], fill=(0, 0, 0, 200))
        draw.text((20, 15), text, font=self.font, fill=(255, 255, 255))
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def capture_loop(self):
        """ひたすらカメラを見て表示するスレッド"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.latest_frame = frame
            
            # テキストを重ねて表示
            display_frame = self.add_text_to_image(frame, self.narrative)
            cv2.imshow('Gemini 2.0 Eye', display_frame)
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        self.cap.release()
        cv2.destroyAllWindows()

    def analyze_loop(self):
        """定期的にAIに「これ何？」って聞くスレッド"""
        # チャットセッションを開始（履歴を覚えられる！）
        chat = model.start_chat(history=[])
        
        while self.running:
            if self.latest_frame is None:
                time.sleep(0.1)
                continue

            try:
                # 画像変換: OpenCV(BGR) -> PIL(RGB)
                img_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                pil_img = PIL.Image.fromarray(img_rgb)

                # Geminiに画像とプロンプトを投げる！
                response = chat.send_message(
                    [SYSTEM_PROMPT, pil_img],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=50, # 短文縛り
                        temperature=0.7
                    )
                )
                
                # AIの返事を更新
                self.narrative = response.text.replace("\n", "")
                print(f"🎙️ AI: {self.narrative}")
                
                # API制限考慮してちょっと待つ（1秒間隔くらいが丁度いい）
                time.sleep(1.0) 
                
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

if __name__ == "__main__":
    app = RealTimeNarrator()
    
    # AI思考回路（別スレッド）を起動
    thread_analyze = Thread(target=app.analyze_loop)
    thread_analyze.start()
    
    # メイン画面起動
    app.capture_loop()
    
    thread_analyze.join()