# 🦋 Simple-AIANO-Core

**AIANO (AI Augmented annotation)** のコアロジックである「Human-AI 協働アノテーション」と「ブロックシステム」を再現した最小実装です。

論文「[AIANO: Enhancing Information Retrieval with AI-Augmented Annotation](https://www.arxiv.org/abs/2602.04579)」 で提唱された、RAG（Retrieval-Augmented Generation）評価データセット作成の効率化手法を Python で体験できます。

> **Note:** このリポジトリは論文の概念実証（PoC）用のバックエンドロジックであり、論文中のフルスタックアプリケーション（React/FastAPI）そのものではありません。

## 📖 概要

RAGの評価データセット（クエリ・根拠・回答のトリプレット）を作成する際、人間がゼロから回答を書くのではなく、**「人間が根拠をハイライト」→「AIが回答を生成」** というフローを採用することで、作業時間を短縮し精度を向上させるアプローチの実装例です 。

### 特徴

* **Block Architecture**: タスクをモジュール化する `BaseBlock`, `AISoloBlock`, `CollaborativeBlock` の設計 。


* **Context Injection**: 人間が選択した根拠（Span）のみをコンテキストとしてLLMに渡す仕組み 。


* **Pydantic Models**: 堅牢なデータ構造定義。

## 🚀 セットアップ

### 必要要件

* Python 3.10+
* OpenAI API Key (実際に生成を行う場合)

### インストール


1. 依存ライブラリをインストールします:
```bash
pip install openai pydantic

```


2. 環境変数を設定します（推奨）:
```bash
# Mac/Linux
export OPENAI_API_KEY="sk-proj-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-proj-..."

```


> **Note**: APIキーがない場合でも、コード内の `mock-key` により動作確認は可能ですが、実際の生成は行われません（認証エラーになります）。



## 💻 使い方

`main.py` を実行すると、デモシナリオ（AIANO自体のメリットを問うQ&A作成）が走ります。

```bash
python main.py

```

### 実行結果例

以下のように、人間がハイライトした根拠（Evidence）に基づいて、AIが回答を生成します。

```text
🤝 [Collaborative] Generating answer from 1 spans...
--------------------
Q: AIANOを使うメリットは何ですか？
Evidence: ['RAGのデータ作成速度が約2倍になります。']
A (AI Generated): AIANOを使うメリットは、RAGのデータ作成速度が約2倍になることです。
--------------------

```

## 📂 ファイル構成

```text
.
├── main.py          # コアロジックと実行デモ
└── README.md        # ドキュメント（本ファイル）

```

## 🛠️ 技術解説

このコードの核心は `CollaborativeBlock` クラスにあります。

1. **Selection**: ユーザーがドキュメント内の特定箇所を `Span` オブジェクトとして選択。
2. **Prompting**: 選択された `Span` のテキストのみを抽出し、プロンプトの `{evidence}` フィールドに注入。
3. **Generation**: LLMが根拠に基づいた回答（Grounded Answer）を生成。

これにより、LLMの幻覚（Hallucination）を抑制しつつ、アノテーション作業の認知負荷を下げています 。

## 📚 参考文献

* **Paper**: [AIANO: Enhancing Information Retrieval with AI-Augmented Annotation (arXiv:2602.04579)](https://www.arxiv.org/abs/2602.04579) 



## 👤 Author

**Shogo Miyawaki**