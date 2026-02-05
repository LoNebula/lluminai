# 🦾 HARR: Minimal Implementation Demo

論文 **"Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG"**  に基づく、**HARR (History-Aware Reinforced Retriever)** の最小構成による実装デモです。

本リポジトリでは、**GRPO (Group Relative Policy Optimization)** と **Plackett-Luceサンプリング**  を用いて、Dense Retriever（密ベクトル検索器）を強化学習（RL）で最適化する手法を実証します。これにより、Retrieverが下流のLLMからのフィードバック（F1スコアなど）を直接学習できるようになります。

## 📖 概要

RAG (Retrieval-Augmented Generation) において、Retrieverの事前学習目的（類似度予測）とGeneratorの目的（回答の正確性）の間には不一致が生じがちです 。HARRはこの問題を解決するため、強化学習を用いてRetrieverをファインチューニングします 。

**本コードの特徴:**

* **Plackett-Luceサンプリング**: 決定的なTop-k検索の代わりに、微分可能な確率的サンプリングメカニズムを実装 。


* **GRPO (Group Relative Policy Optimization)**: Criticモデル（価値関数）を必要としない、効率的なRL損失関数 。


* **Mock Environment**: LLMや大規模データセットを用意せずともアルゴリズムの動作原理を理解できるよう、クエリエンコーダや環境（報酬計算）を模倣（Mock）したコンポーネントを使用。

> **Note**: 本コードはHARRのアルゴリズム的な核心を理解するための教育用実装です。すべての環境で即座に動作確認ができるよう、エンコーダや環境は `Mock` クラスで代用しています。

## 🚀 クイックスタート

### 前提条件

* Python 3.x
* PyTorch
* NumPy

```bash
pip install torch numpy

```

### デモの実行

スクリプトを実行するだけで、Retrieverが特定のクエリに対して「正解ドキュメント（Doc #0）」を見つけるように学習していく過程をシミュレーションできます。

```bash
python main.py

```

### 実行結果の例

学習が進むにつれて、**Success Rate**（正解ドキュメントを検索できた確率）が上昇していく様子が確認できます。

```text
🚀 HARR Training Demo Started...
🎯 Objective: Queryに対し、正解の 'Document #0' をRetrievalさせる
----------------------------------------
Step 10 | Loss: -0.0000 | Avg Reward: 0.49 | Success Rate: 50%
Step 20 | Loss: 0.0000 | Avg Reward: 0.50 | Success Rate: 50%
Step 30 | Loss: -0.0000 | Avg Reward: 0.83 | Success Rate: 88%
Step 40 | Loss: 0.0000 | Avg Reward: 0.83 | Success Rate: 88%
Step 50 | Loss: -0.0000 | Avg Reward: 0.63 | Success Rate: 62%
----------------------------------------
🎉 Training Finished!

```

## 📂 コード構成

* **`MockQueryEncoder`**: Transformerベースのクエリエンコーダ（BERTやQwen-Embeddingなど）を模倣したクラス。
* **`MockEnvironment`**: RAGプロセス全体をシミュレートします。正解ドキュメントが取得できた場合に高い報酬（約1.0）、それ以外の場合に低い報酬を返す「LLMの審査員」として機能します 。


* **`HARRRetriever`**:
* `forward()`: クエリとドキュメント間の類似度スコアを計算します。
* `sample_documents()`: **Plackett-Luceサンプリング** を実行し、確率に基づいてドキュメントを選択（探索）します 。




* **`compute_grpo_loss()`**: GRPO損失関数の実装です。選択されたアクションのアドバンテージ（グループ平均との差分）を計算して最適化します 。



## 🛠️ 実運用への適用 (Adaptation)

このロジックを実際のRAGパイプラインに適用する場合は、Mockコンポーネントを以下のように置き換えてください。

1. **Encoder**: `MockQueryEncoder` を Hugging Face の実モデル（例: `transformers.AutoModel`）に置き換える。
2. **Environment**: `MockEnvironment.get_reward()` を以下の処理を行う関数に置き換える：
   * 検索されたドキュメントを受け取る。
   * 実際のLLM（GPT-4, Claude, ローカルLLM等）を使用して回答を生成する 。


   * 生成された回答と正解データ（Ground Truth）を比較し、F1スコアなどで評価する 。




3. **Data**: 実際の「クエリ - 回答」ペアのデータセットを使用する。

## 📄 参考文献

本コードは以下の論文に基づいています。

* **Title**: Reinforcement Fine-Tuning for History-Aware Dense Retriever in RAG 


* **Authors**: Yicheng Zhang, Zhen Qin, et al. 


* **ArXiv**: 2602.03645v1 

URL: https://arxiv.org/abs/2602.03645



## 👤 Author

* **Shogo Miyawaki** 