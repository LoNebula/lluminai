# 🕸️ Mini-GraphRAG Implementation

Microsoft Researchが提唱した **GraphRAG (Retrieval-Augmented Generation with Graphs)** のコアロジック（抽出・グラフ構築・コミュニティ検出・要約）を、公式ライブラリを使わずに `LangChain` + `NetworkX` でミニマムに実装・解説しています。


## 🚀 特徴 (What is this?)

このノートブックでは、以下の GraphRAG パイプラインをPythonだけで再現します：

1.  **Extract**: LLMを使ってテキストから「エンティティ」と「関係性」を構造化データとして抽出
2.  **Graph Construction**: `NetworkX` を用いてナレッジグラフを構築
3.  **Community Detection**: グラフから密接なグループ（コミュニティ）を検出
4.  **Global Summary**: コミュニティごとに要約を生成（ここがGraphRAGの肝！）
5.  **Graph-based QA**: 生成された要約のみを用いて、全体俯瞰的な質問に回答

## 🛠️ 必要要件 (Requirements)

* Python 3.9+
* OpenAI API Key (GPT-4o 推奨)

### 使用ライブラリ
* `langchain` / `langchain-openai`
* `networkx`
* `python-dotenv`
* `cdlib` (Optional: 本格的なコミュニティ検出をする場合)

## 🏃 使い方 (Getting Started)

### 1. ライブラリのインストール

```bash
pip install -r requirements.txt

```

### 2. 環境変数の設定

プロジェクト直下に `.env` ファイルを作成し、OpenAIのAPIキーを設定してください。

**`.env` ファイルの内容:**

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

```

### 3. ノートブックの実行

Jupyter Lab または VS Code で `main.ipynb` を開き、セルを上から順に実行してください。

## ⚠️ 注意事項 (Disclaimer)

* **教育目的の実装です**: 本コードはGraphRAGのアルゴリズムを理解するための簡易実装（Minimal Implementation）です。Microsoft公式の `graphrag` ライブラリのような大規模データ対応や並列処理機能はありません。
* **コストにご注意ください**: 実行時にGPT-4oを使用するため、OpenAI APIの利用料金が発生します。
