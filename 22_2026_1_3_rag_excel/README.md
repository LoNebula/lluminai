# Visual RAG for Excel: ColPali + Qwen2-VL 📊👁️

**「PandasでExcelをパースするのはもう終わり。AIに”直接見せる”時代へ。」**

本リポジトリは、Excelファイル（`.xlsx`）をテキストデータとしてではなく、**視覚的なドキュメント**として扱う「Visual RAG (Retrieval-Augmented Generation)」の実装デモです。
従来のテキスト抽出型RAGでは無視されてしまっていた**グラフ、チャート、セルの色分け、浮動テキストボックス**などの要素を正確に認識・検索・理解します。

Google Colabの無料枠（T4 GPU, VRAM 16GB）で完走できるように、メモリ管理と量子化処理を最適化しています。

## 🌟 主な特徴

* **📈 グラフ・図表の理解**: 折れ線グラフのトレンドや棒グラフの大小比較など、画像情報から数値を解釈します。
* **🎨 レイアウトの完全保持**: **LibreOffice** を使用し、人間が見ている通りの見た目（条件付き書式や注釈を含む）でレンダリングします。
* **🚫 OCR・テキスト抽出不要**: **ColPali** を使用し、ページ画像を直接Embedding化します。`pdfminer` や `pandas` でテーブルが崩れる悩みから解放されます。
* **⚡ Colab T4 最適化**: Retriever（検索）とGenerator（生成）を直列実行し、都度VRAMを強制解放（GC）する仕組みと、4bit量子化（`bitsandbytes`）を導入することで、限られたリソースでも7Bモデルを動作可能にしています。

## 🛠️ アーキテクチャ

1. **Rendering**: ヘッドレスモードの **LibreOffice** を使い、Excel (`.xlsx`) をPDF/画像に変換。
2. **Indexing (Retriever)**: **[ColPali](https://huggingface.co/vidore/colpali-v1.2)** (Vision-Language Modelベースの検索) で画像をベクトル化。
3. **Memory Cleanup**: **(最重要)** 検索終了後にRetrieverモデルをGPUメモリから削除し、VRAMを解放。
4. **Generation (Reader)**: **[Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)** (4bit量子化版) をロードし、検索された画像を見て回答を生成。

## 📦 必要要件

### システム依存パッケージ (Linux/Colab)

PDF処理のために Poppler と LibreOffice が必要です。

```bash
sudo apt-get update
sudo apt-get install -y libreoffice fonts-ipafont poppler-utils

```

### Pythonライブラリ

```bash
pip install torch transformers accelerate bitsandbytes qwen-vl-utils byaldi pdf2image xlsxwriter

```

## 🚀 クイックスタート (Google Colab)

1. **ノートブックを開く**: 上記の "Open in Colab" バッジをクリックしてください。
2. **全セルを実行**: テスト用にグラフ付きのダミーExcelを生成するスクリプトが含まれています。
3. **質問する**:
```python
query = "売上の推移グラフを見て、最も落ち込んでいる月とその原因は？"

```



## 📖 使い方ガイド

### 1. ダミーデータの生成 (任意)

手元に手頃なExcelがない場合、付属のスクリプトを実行すると `monthly_sales_report.xlsx`（V字回復グラフと注釈付き）が生成されます。

### 2. Excelの画像化

LibreOfficeを使って高精細な画像に変換します。

```python
# 詳細はノートブックを参照
pdf_path, images = convert_excel_to_rag_ready("monthly_sales_report.xlsx")

```

### 3. 検索と回答生成

T4 GPUでOOM (Out Of Memory) を回避するため、パイプライン内でメモリ管理を自動で行います。

```python
# 1. インデックス作成
RAG.index(input_path=pdf_path, index_name="excel_index")

# 2. 検索実行
results = RAG.search("グラフのV字回復の要因を説明して", k=1)

# 3. メモリ解放 (重要！)
del RAG; torch.cuda.empty_cache()

# 4. 回答生成 (Qwen2-VL 4-bit)
output = model.generate(**inputs)

```

## 📊 出力例

**入力Excel画像:**

* 4月に底を打ち、8月に急回復する「売上推移グラフ」がある。
* グラフの横に「※5月から新マーケティング施策を開始」というテキストボックス（注釈）がある。

**ユーザーの質問:**

> "売上の推移グラフを見て、最も落ち込んでいる月とその原因と思われる注釈を教えて"

**モデルの回答:**

> "グラフによると、売上は4月に底を打ち（約200万円）、その後急激に回復して8月には過去最高の500万円に達しています。グラフの横にある吹き出しには「※5月から新マーケティング施策を開始」という注釈があります。"