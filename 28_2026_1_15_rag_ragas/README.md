# RAG Evaluation Demo with Ragas

RAG (Retrieval-Augmented Generation) パイプラインの精度を評価フレームワーク **[Ragas](https://docs.ragas.io/)** を用いて定量化するためのサンプルコードです。

LLM（GPT-4o-mini）を審査員（Judge）として使用し、検索精度や回答の忠実性を自動スコアリングする「LLM-as-a-Judge」の実装例です。

## 🚀 特徴

* **Ragasによる多角的評価**: Faithfulness, Answer Relevancy, Context Precision, Context Recallの4指標を計測。
* **OpenAIモデルの明示的指定**: `langchain-openai` を使用し、評価用モデル（GPT-4o-mini）とEmbeddingモデルを明示的に制御。
* **意図的な「失敗ケース」の検証**:
* **ハルシネーション（外部知識の混入）**: 文脈に含まれていない正しい知識を答えた場合のスコア低下を確認可能。
* **検索失敗**: 必要な情報が検索できていないケースを確認可能。



## 📦 必要要件

* Python 3.10+
* OpenAI API Key

## 🛠️ インストール

必要なライブラリをインストールしてください。

```bash
pip install ragas langchain-openai openai datasets python-dotenv pandas

```

## ⚙️ セットアップ

プロジェクトルートに `.env` ファイルを作成し、OpenAIのAPIキーを設定してください。
※コード内で `OPENAI_API_KEY_LLUMINAI` という環境変数名を参照しています。

```env
OPENAI_API_KEY_LLUMINAI=sk-proj-your-api-key-here...

```

## 📊 評価指標とサンプルデータの意図

このコードに含まれるデータセットは、RAGのよくある失敗パターンを検出できるように設計されています。

### 計測するメトリクス

| 指標                           | 説明                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| **Faithfulness** (忠実性)      | 回答が「検索された文脈」のみに基づいているか。外部知識（ハルシネーション）がないか。 |
| **Answer Relevancy** (関連性)  | ユーザーの質問に対して的確に答えているか。                                           |
| **Context Precision** (適合率) | 検索結果の上位に関連度の高いドキュメントが含まれているか。                           |
| **Context Recall** (再現率)    | 正解（Ground Truth）を導くのに必要な情報が検索できているか。                         |

### データセットの解説（実行結果の読み方）

実行結果では、以下のような挙動が確認できます。

1. **大谷翔平の質問**
* **挙動**: 回答は事実として正しい（ドジャース移籍など）が、検索された文脈（Contexts）にはその記述がない。
* **結果**: **Faithfulness が 0.0 になる**（文脈にない外部知識で答えているため、RAGとしては不正解判定）。


2. **Pythonの質問**
* **挙動**: 検索結果に回答に必要な情報（動的型付けなど）が含まれていない。
* **結果**: **Context Recall が 0.0 になる**（検索失敗）。



## 📝 実行結果サンプル

```text
詳細スコア:
             question  faithfulness  context_recall  answer_relevancy
0      フランスの首都はどこですか？           1.0             1.0          0.9823
1      Pythonの特徴について教えて。           0.0             0.0          0.8312
2  大谷翔平の2024年の所属チームは？           0.0             1.0          0.8337

```