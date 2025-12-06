# Minimal LLM Pre-training Pipeline (Japanese) 🧱

日本語LLMをゼロから事前学習（Pre-training）するための、**データセット作成からPyTorch Dataset定義までの一気通貫パイプライン**の最小実装です。

Zenn記事連載 **「LLM自作入門」Vol.1** の実証コードとして作成。

## 🚀 Features

このリポジトリのコードは、以下の3つのモダンなLLM開発プロセスを実装しています。

1.  **Streaming Data Loading** 🌊
    * `datasets` の `streaming=True` を活用。
    * TB級の巨大データセット（Wikipedia等）でも、ローカルマシンのメモリを圧迫せずにダウンロード＆処理が可能。
2.  **Custom BPE Tokenizer** ✂️
    * Hugging Face `tokenizers` を使用し、**Byte-Pair Encoding (BPE)** をスクラッチ学習。
    * 日本語データに特化した語彙（Vocab Size: 32,000）を構築し、未知語（Unknown Token）を理論上ゼロにする `ByteLevel` 設定を採用。
3.  **PyTorch Dataset for Pre-training** 📚
    * Next Token Predictionタスク用に、テキストを固定長（Context Length）でチャンキングして供給する `Dataset` クラスの実装。

## 🛠 Requirements

```bash
pip install torch datasets tokenizers tqdm
```

### 実行フロー

スクリプトを実行すると、以下の処理が順次行われます。

1.  **データ取得**: `izumi-lab/wikipedia-ja-20230720` から日本語Wikipediaデータをストリーミング取得し、最初の1万件を `wiki_ja_subset.txt` に保存します。
2.  **トークナイザ学習**: 保存したテキストデータを元にBPEトークナイザを学習し、`custom_tokenizer.json` を出力します。
3.  **動作検証**: 文のエンコード・デコード結果を表示し、トークナイズの挙動を確認します。
4.  **Dataset構築**: PyTorchの `Dataset` クラスをインスタンス化し、学習用テンソル（Input ID）の形状を確認します。

## 📝 Code Explanation

### 1\. Data Preparation

メモリ効率を最優先し、Hugging Face Hubからデータをストリーミング（逐次読み込み）しています。
日本語Wikipedia特有のノイズ除去（改行削除）や、短すぎる記事の足切り（100文字以下除外）などの前処理を含みます。

### 2\. Tokenizer (BPE)

GPT-2やLlamaと同様の **Byte-Level BPE** を採用しています。

```python
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
```

これにより、漢字・絵文字・特殊記号を含むあらゆるテキストをバイト列として処理し、堅牢なトークナイズを実現しています。

### 3\. Pre-training Dataset

LLMの学習（Causal Language Modeling）のために、連続したテキストデータを `max_length` ごとに分割（Chunking）します。
`__getitem__` では `x` (input) と `y` (target) として同じテンソルを返します（※Shift処理は学習ループ内で行う想定）。

## ⚠️ Limitations & Future Work

  * **Memory Usage**: 現在の `LLMPretrainDataset` は `f.read()` で全テキストをメモリに展開しています。数GB以上のデータを扱う際は `np.memmap` 等の使用を推奨します。
  * **Normalization**: 本実装では簡易的なクリーニングのみ行っています。実運用では `NFKC` 正規化などをトークナイザパイプラインに組み込むことを推奨します。
