# 👁️ ColPali Minimal Demo: OCR不要の文書検索 (CPU対応版)

このリポジトリは、OCR（光学文字認識）を使用せずに文書画像を直接検索する技術 **「ColPali (ColBERT + PaliGemma)」** の最小実装デモです。

論文 *[ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)* のコンセプトを、**GPUがない環境やGoogle Colabの無料枠でも動作するように最適化**しています。

## 🚀 特徴

* **OCR完全撤廃**: PDFや画像を「テキスト」に変換せず、視覚情報（レイアウト、図表、文字）をそのままベクトル化して検索します。
* **省メモリ設計**: `bfloat16` を強制指定することで、メモリ消費量を大幅に削減。一般的なPCのCPU環境や、VRAMの少ないGPUでも動作可能です。
* **堅牢な実行**: 外部画像のリンク切れが発生しても、ダミー画像を自動生成して処理を継続する安全装置を組み込んでいます。
* **Late Interaction**: クエリと画像のパッチを細粒度でマッチングさせる `MaxSim` 演算を体験できます。

## 🛠️ インストール

Python 3.10以上推奨。以下のコマンドで必要なライブラリをインストールしてください。

```bash
pip install torch requests pillow colpali-engine transformers

```

※ `colpali-engine` はHugging FaceのTransformersライブラリに依存しています。

## 💻 使い方

リポジトリをクローンし、スクリプトを実行するだけです。

```bash
python main.py

```

### 実行の流れ

1. **モデルロード**: `vidore/colpali-v1.2` (PaliGemma-3Bベース) をダウンロードします（初回のみ数分かかります）。
2. **画像準備**: サンプルの文書画像（アーキテクチャ図）と、無関係な画像（猫）をWebから取得します。
3. **推論**: 以下のクエリに対して、どちらの画像がふさわしいかスコアを計算します。
* "What is ColPali?"
* "Is there a cat?"
* "Show me the architecture."



## 📊 出力例

以下のようなスコアが表示されます。
スコアは絶対値（0~1）ではなく、**Late Interactionの総和スコア**であるため、相対比較で判断します。

```text
🚀 環境設定を確認中...
   Device: cpu (または cuda)
📥 モデルをロード中: vidore/colpali-v1.2 ...
✅ モデルロード完了

🖼️ 画像を準備中...
   Downloading: Document(ColPali Paper)... OK!
   Downloading: Cat Image... OK!

⚙️ Embedding生成とスコアリング計算中...

📊 --- 検索結果 (類似度スコア) ---

🔍 Query: 'What is ColPali?'
   📄 vs Document(ColPali Paper): Score = 22.8125
   📄 vs Cat Image: Score = 6.0938
      👉 ★ Top Match!  <-- 文書画像が正しくヒット

🔍 Query: 'Is there a cat?'
   📄 vs Document(ColPali Paper): Score = 1.1875
   📄 vs Cat Image: Score = 8.6875
      👉 ★ Top Match!  <-- 猫画像が正しくヒット

```

※ 画像URLがリンク切れ（404）の場合は、自動的に白紙の画像が生成され、ログに `Failed (...). Creating dummy image instead.` と表示されます。

## 📝 技術解説

### ColPaliとは？

従来のRAG（Retrieval-Augmented Generation）では、PDFをテキスト化（OCR/Parse）する工程で、図表やレイアウト情報が失われる課題がありました。
ColPaliは、VLM（Vision Language Model）を用いて画像をパッチごとの埋め込みベクトル（Multi-Vector）に変換し、ColBERT方式の **Late Interaction** で検索を行うことで、この課題を解決しています。

### コードのポイント

`main.py` 内の以下の設定により、CPU環境での実行を可能にしています。

```python
model = ColPali.from_pretrained(
    model_name,
    dtype=torch.bfloat16,  # <--- 重要: float32だとメモリ溢れするため軽量化
    device_map=device,
).eval()

```

## 🔗 関連リンク

* **Paper:** [arXiv:2407.01449](https://arxiv.org/abs/2407.01449)
* **Model:** [Hugging Face - vidore/colpali-v1.2](https://huggingface.co/vidore/colpali-v1.2)
* **Library:** [illuin-tech/colpali](https://github.com/illuin-tech/colpali)