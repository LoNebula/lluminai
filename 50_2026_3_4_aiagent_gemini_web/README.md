# 🌐 Local VLM Web Explorer (Ollama + Playwright)

**「API課金も、トークン制限も、もう怖くない。」**

本プロジェクトは、ローカルで動作するマルチモーダルAI（VLM）である **Qwen3-VL** と、ブラウザ自動化ライブラリ **Playwright** を組み合わせた、完全無料・プライバシー重視のWeb調査エージェントのプロトタイプです。

## 🚀 特徴

* **完全ローカル実行**: OpenAIやGoogleのAPIキーは不要です。Ollama上で動作するため、通信コストやプライバシーを気にせず無限に試行錯誤できます。
* **視覚ベースの解析**: HTMLのDOM構造をパースするのではなく、スクリーンショット画像から「人間のように」情報を読み取ります。UIの軽微な変更に強い堅牢なスクレイピングを実現します。
* **日本語対応**: 高性能な `qwen3-vl` モデルを採用。Zennなどの技術ブログの日本語タイトルも高精度にOCR（文字認識）可能です。

## 🛠️ セットアップ

### 1. 前提条件

* [Ollama](https://ollama.com/) がインストールされていること
* Python 3.10 以上

### 2. ライブラリのインストール

```bash
# Pythonライブラリのインストール
pip install playwright ollama

# Playwright用ブラウザのセットアップ
playwright install chromium

```

### 3. モデルの準備

Ollamaで `qwen3-vl` をプルしておきます。

```bash
ollama pull qwen3-vl

```

## 💻 使い方

1. リポジトリをクローンします。
2. `main.py` を実行します。

```bash
python main.py

```

### 動作フロー

1. **Playwright** がヘッドレスブラウザを起動し、指定したURL（デフォルトはZenn）へアクセス。
2. 画面のスクリーンショット（`zenn_top.png`）を撮影。
3. **Ollama (Qwen3-VL)** がその画像を解析し、トレンド記事などの情報を抽出。
4. ターミナルに解析結果を出力。

## 📝 実行例

実行すると、以下のような解析結果が得られます。

```text
【解析結果】
 \boxed{1. \text{【全エンジニアがClaude Codeを100%活用する】を目的にしたデッシュボードを作った} \\ 2. \text{Claude CodeやCopilotをオーケストレーションして自前でビジュアルAIアプリをつくる} \\ 3. \text{Skilledで実装する機械学習RAG}}

```

> ※ Qwenモデルの特性上、結果がLaTeX形式（数式フォーマット）で出力されることがありますが、テキスト内容は正確です。

## ⚠️ 注意事項

* ローカルVLMの推論には一定の計算リソースが必要です。GPU（NVIDIA RTX 3060以上推奨）があると快適に動作します。
* スクリーンショット取得後の推論には、マシンスペックにより数十秒〜数分かかる場合があります。

---

*Developed by Shogo Miyawaki (Lluminai)*
