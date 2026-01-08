# 🎙️ Gemini 2.0 Flash Real-Time Narrator

Webカメラの映像をリアルタイムで解析し、Googleの最新モデル **Gemini 2.0 Flash** が「実況アナウンサー」としてコメントし続けるPythonデモアプリです。

OpenCVによる映像キャプチャと、AIによる推論（Vision-Language）を別スレッドで非同期処理することで、**映像のカクつきを防ぎながら連続的な実況**を実現しています。


## 🚀 特徴

- **爆速レスポンス**: 推論エンジンに `gemini-2.0-flash-exp` を採用。動画のような連続的な入力に対しても高速に応答します。
- **非同期アーキテクチャ**: 映像描画ループとAI推論ループを `threading` で分離。推論待ち時間中も映像がフリーズしません。
- **日本語オーバーレイ**: OpenCVの画像にPillowを使って日本語フォントを合成表示します。
- **人格カスタマイズ**: システムプロンプトを変更するだけで、「実況アナウンサー」から「関西弁のオカン」「冷静な解説者」までキャラ変可能です。

## 🛠️ 必要要件 (Requirements)

- Python 3.9+
- Webcam
- Google AI Studio API Key

## 📦 インストール

依存ライブラリをインストールします。
```bash
pip install opencv-python pillow google-generativeai numpy
```



## 🔑 セットアップ

Google AI Studio ( https://aistudio.google.com/ ) からAPIキーを取得してください。

コード内の `os.environ["GOOGLE_API_KEY"]` を書き換えるか、環境変数として設定してください。

**main.py:**

```python
# 推奨: 環境変数から読み込むか、ここに直接書き込む（公開リポジトリにはコミットしないよう注意！）
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

```

## ▶️ 使い方

```bash
python main.py

```

* **起動後**: Webカメラのウィンドウが立ち上がり、AIの実況テキストが表示されます。
* **終了**: 映像ウィンドウをアクティブにした状態で `q` キーを押してください。

## ⚙️ カスタマイズ

### キャラクター設定 (System Prompt)

`main.py` の `SYSTEM_PROMPT` 変数を編集することで、AIの振る舞いを変更できます。

```python
SYSTEM_PROMPT = """
あなたは名探偵です。
目の前の映像を見て、怪しい点や手掛かりを
ハードボイルドな口調で推理してください。
"""

```

## 🧩 技術スタック

* **AI Model**: Google Gemini 2.0 Flash Experimental
* **Vision**: OpenCV (`cv2`)
* **Image Processing**: Pillow (`PIL`)
* **Concurrency**: Python Standard Library (`threading`)

## ⚠️ 注意事項

* APIの利用枠（Rate Limit）にご注意ください。
* `gemini-2.0-flash-exp` は実験的モデルであり、挙動が変更される可能性があります。