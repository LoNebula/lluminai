# Visual RAG with ColPali & Qwen2-VL 👁️📄

OCR（光学文字認識）を一切使用せず、PDFを「画像」としてそのまま検索・理解する **Visual RAG** の実装デモです。

Google Colabの無料枠（T4 GPU, VRAM 16GB）でも動作するように、メモリ管理と量子化（4-bit quantization）を最適化しています。

## 🚀 Features

* **OCR-Free Retrieval**: テキスト抽出を行わず、PDFのページ全体を画像としてEmbedding化します。図表、グラフ、複雑なレイアウトを崩さずに検索可能です。
* **Vision Language Model (VLM)**: 検索結果の画像を `Qwen2-VL` が直接読み取り、視覚情報に基づいて回答を生成します。
* **Low VRAM Optimized**: RetrieverとGeneratorを直列に実行し、都度VRAMを強制解放（Garbage Collection）することで、限られたGPUリソースでも 7B モデルを動作可能にしています。

## 🛠️ Architecture

このシステムは以下のパイプラインで動作します：

1. **Index**: PDFをページごとの画像に変換し、`ColPali` でベクトル化。
2. **Retrieve**: クエリ（テキスト）に最も近いページ画像を取得。
3. **Clean Up**: **(重要)** `ColPali` モデルをメモリから削除し、VRAMを解放。
4. **Generate**: `Qwen2-VL (4-bit)` をロードし、画像とクエリを入力して回答を生成。

## 📦 Installation

Google Colabでの実行を推奨します。ローカル環境（Linux/WSL）の場合は以下をインストールしてください。

**System Dependencies (Required for PDF processing):**

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils

```

**Python Libraries:**

```bash
pip install torch transformers accelerate bitsandbytes qwen-vl-utils byaldi pdf2image

```

## 📖 Usage

### 1. Run on Google Colab

リポジトリ内のノートブックを開き、上から順にセルを実行してください。

### 2. Custom Data

`sample.pdf` 変数に、解析したいPDFのパスを指定してください。

```python
# 任意のPDFパス
pdf_path = "your_document.pdf"

```

### 3. Query

`query_text` に質問を入力します。

```python
query_text = "この図表のトレンドと、注釈に書かれている課題点は？"

```

## 🧠 Models Used

* **Retriever**: [vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)
* Based on PaliGemma. Optimized for efficient document retrieval using late interaction (ColBERT strategy).


* **Generator**: [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
* State-of-the-art visual language model. Running in 4-bit quantization via `bitsandbytes`.



## ⚠️ Limitations & Notes

* **Execution Time**: メモリ節約のためにモデルのロード/アンロードを繰り返すため、連続的なチャットには向きません（バッチ処理向き）。
* **VRAM**: 実行には最低でも 12GB 程度のVRAMが必要です（T4 16GB 推奨）。
* **Image Only**: PDFは画像として処理されるため、テキスト情報のコピー＆ペーストはできません。

---

### Acknowledgements

This implementation relies on the amazing work by:

* [Hugging Face](https://huggingface.co/)
* [Byaldi (RAGMultiModalModel)](https://github.com/AnswerDotAI/byaldi)
* [Qwen Team](https://github.com/QwenLM/Qwen2-VL)