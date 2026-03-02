import asyncio
from playwright.async_api import async_playwright
import ollama

async def capture_and_analyze():
    async with async_playwright() as p:
        # 1. ブラウザを起動し、Zennのトップページへアクセス
        print("ブラウザを起動し、Zennのスクリーンショットを取得します...")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://zenn.dev/")
        
        # ページが読み込まれるまで少し待機
        await page.wait_for_timeout(3000)
        
        # スクリーンショットを保存
        screenshot_path = "./zenn_top.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        await browser.close()
        print("スクリーンショットの取得が完了しました！")

        prompt = """
        あなたは優秀なWebリサーチャーです。
        このスクリーンショットは技術ブログZennのトップページです。
        画像内の情報から、現在トレンドになっている（一番目立っている）記事を3つ選び、
        それぞれの「タイトル」と「推測される概要」を抽出して、日本語で分かりやすく要約してください。
        """
        
        print("ローカルの qwen3-vl モデルに画像を解析させています...（少し時間がかかる場合があります）")
        
        # 2. OllamaのローカルVLMを使って画像を解析
        response = ollama.chat(
            model='qwen3-vl:latest',
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [screenshot_path]  # ファイルパスをそのまま渡せる！
                }
            ]
        )
        
        result_text = response['message']['content']
        print("\n【解析結果】\n", result_text)
        
        return result_text

# 実行
if __name__ == "__main__":
    asyncio.run(capture_and_analyze())
