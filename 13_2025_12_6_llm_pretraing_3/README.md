# 📉 LLM自作入門 Vol.3: 学習ループの設計と省メモリ技術 (Training Loop Optimization)

このリポジトリは、技術ブログ **「LLMをゼロから作る」連載 Vol.3** の実装コードです。
Google Colab (T4 GPU) のような限られたリソース環境でも、Llamaアーキテクチャを採用したLLMを事前学習（Pre-training）させるための実践的なエンジニアリングを実装しています。

## 📖 概要

数十億パラメータのLLMを学習させるには膨大な計算資源が必要ですが、基礎的なメカニズムは小規模な実験で理解できます。
本コードでは、**System RAM (CPUメモリ)** と **VRAM (GPUメモリ)** の両方の制約を回避しつつ、日本語Wikipediaデータセットを用いてモデルを学習させるパイプラインを構築します。

### 主な機能

  * **省メモリデータ読み込み (Lazy Loading)**: データセット全体をメモリに展開せず、必要な瞬間にディスクから読み込む `LazyLLMPretrainDataset` を実装。
  * **勾配蓄積 (Gradient Accumulation)**: 小さなバッチサイズで計算した勾配を積み重ね、実質的に大きなバッチサイズで更新を行うことでVRAM不足を解消。
  * **混合精度学習 (Mixed Precision)**: `torch.amp` (Automatic Mixed Precision) を利用し、fp16での計算による高速化とメモリ節約を実現。
  * **モダンなアーキテクチャ**: RMSNorm, RoPE (Rotary Positional Embeddings), SwiGLU を採用したLlamaベースのモデル定義。

## 🛠️ 要件 (Requirements)

  * Python 3.10+
  * PyTorch 2.0+ (CUDA推奨)
  * Datasets (Hugging Face)
  * Tokenizers (Hugging Face)

## 🚀 クイックスタート

### 1\. インストール

```bash
pip install torch datasets tqdm tokenizers
```

### 2\. 実行

※ **Google Colabでの実行を推奨します**（ランタイムのタイプを `T4 GPU` などに設定してください）。

## 📂 コード構成

コードは以下のセクションで構成されています。

1.  **データセット作成**: 日本語Wikipediaからサブセットを抽出し、テキストファイル化。
2.  **トークナイザ学習**: Byte-Pair Encoding (BPE) トークナイザを独自データで学習。
3.  **Datasetクラス実装**: `f.seek` / `f.tell` を活用した **Lazy Loading** の実装（★重要）。
4.  **モデル定義**: PyTorchによる Transformer (Llama Architecture) のフルスクラッチ実装。
5.  **学習ループ**: AdamW, Cosine Scheduler, Gradient Clipping, AMP を組み合わせた堅牢なループ。
6.  **生成テスト**: Greedy Decoding による推論テスト。

## 📊 実験結果 (Sample Output)

学習（1エポック程度）後の生成例：

```
Input: "人工知能は"
Output: "人工知能は、日本の小説家、日本の小説家、日本の小説家..."
```

*(注: 初期の学習段階では、特定の頻出パターン（Wikipediaの「〜は日本の〜である」構文など）を繰り返す傾向が見られますが、これは文法構造を学習し始めている証左です)*
