# 📏 RAG Chunk Size Optimizer with Ragas

RAG (Retrieval-Augmented Generation) システムにおける「最適なチャンクサイズ」を科学的に導き出すための実験スクリプトです。

文書の長さ（Short 〜 Very Long）とチャンクサイズ（128 〜 4096）の組み合わせを総当たり（Grid Search）で評価し、評価フレームワーク **[Ragas](https://docs.ragas.io/)** を用いてスコアをヒートマップ化します。

## 🚀 プロジェクトの目的

RAGの精度は「チャンクサイズ」に大きく依存しますが、最適なサイズはドキュメントの性質によって異なります。
このプロジェクトでは、以下の仮説を検証・可視化することを目的としています。

* 短い文書には小さいチャンクサイズが適しているか？
* 長文（小説や規約など）において、中途半端なチャンクサイズが検索精度（Recall）を低下させる「埋没問題」は発生するか？
* LLM-as-a-Judge（Ragas）を用いた定量評価の有用性。

## 📊 実験の概要

以下の4種類のドキュメントタイプに対し、6段階のチャンクサイズを適用して評価を行います。

### 1. ドキュメントタイプ

* **Short**: 約250文字（日記・短文）
* **Medium**: 約500文字（手紙・メール）
* **Long**: 約2,500文字（回想録・記事）
* **Very Long**: 約10,000文字（小説・長文規約）

### 2. 検証するチャンクサイズ

`128`, `256`, `512`, `1024`, `2048`, `4096`

### 3. 評価指標 (Ragas Metrics)

* **Context Recall**: 正解に必要な情報が検索できているか（再現率）。
* **Context Precision**: 検索結果にノイズが含まれていないか（適合率）。
* **Faithfulness**: 回答が文脈に忠実か（ハルシネーションのなさ）。
* **Answer Relevancy**: 質問に対して的確に答えているか。

## 📦 必要要件 (Prerequisites)

* Python 3.10+
* OpenAI API Key (GPT-4oなどのモデルを利用)

## 🛠️ インストール (Installation)

リポジトリをクローンし、必要なライブラリをインストールします。

```bash
git clone https://github.com/lonebula/29_1_15_rag_ragas2.git
cd 29_1_15_rag_ragas2
pip install -r requirements.txt

```

※ `requirements.txt` の内容は以下を想定しています：

```txt
ragas
langchain
langchain-community
langchain-openai
faiss-cpu
pandas
numpy
datasets
python-dotenv

```

## ⚙️ セットアップ (Setup)

プロジェクトルートに `.env` ファイルを作成し、OpenAIのAPIキーを設定してください。
（コード内では `OPENAI_API_KEY_LLUMINAI` を読み込んでいますが、ご自身の環境に合わせて変更可能です）

```env
OPENAI_API_KEY_LLUMINAI=sk-proj-your-api-key-here...

```

## 📝 出力サンプル

実行が完了すると、以下のような分析結果が表示されます。

```text
c:\Users\LoNebula\miniconda3\envs\bs\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
  from .autonotebook import tqdm as notebook_tqdm
WARNING:tensorflow:From c:\Users\LoNebula\miniconda3\envs\bs\Lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

C:\Users\LoNebula\AppData\Local\Temp\ipykernel_21556\1350883754.py:12: DeprecationWarning: Importing faithfulness from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import faithfulness
  from ragas.metrics import (
C:\Users\LoNebula\AppData\Local\Temp\ipykernel_21556\1350883754.py:12: DeprecationWarning: Importing answer_relevancy from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import answer_relevancy
  from ragas.metrics import (
C:\Users\LoNebula\AppData\Local\Temp\ipykernel_21556\1350883754.py:12: DeprecationWarning: Importing context_precision from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import context_precision
  from ragas.metrics import (
C:\Users\LoNebula\AppData\Local\Temp\ipykernel_21556\1350883754.py:12: DeprecationWarning: Importing context_recall from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import context_recall
  from ragas.metrics import (
🧪 Ragas深掘り実験を開始します（4指標計測）...

📂 Document Type: 1. Short
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:14<00:00,  3.72s/it]
   - Chunk  128: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:11<00:00,  2.88s/it]
   - Chunk  256: Recall=1.00, Faith=0.50
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.10s/it]
   - Chunk  512: Recall=1.00, Faith=0.50
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.24s/it]
   - Chunk 1024: Recall=1.00, Faith=0.50
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:15<00:00,  3.95s/it]
   - Chunk 2048: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:21<00:00,  5.32s/it]
   - Chunk 4096: Recall=1.00, Faith=0.50

📂 Document Type: 2. Medium
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:13<00:00,  3.31s/it]
   - Chunk  128: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:09<00:00,  2.49s/it]
   - Chunk  256: Recall=1.00, Faith=1.00
Evaluating:  25%|██▌       | 1/4 [00:02<00:06,  2.22s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:10<00:00,  2.53s/it]
   - Chunk  512: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:11<00:00,  2.87s/it]
   - Chunk 1024: Recall=1.00, Faith=1.00
Evaluating:  25%|██▌       | 1/4 [00:02<00:06,  2.27s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:10<00:00,  2.60s/it]
   - Chunk 2048: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:07<00:00,  1.97s/it]
   - Chunk 4096: Recall=1.00, Faith=1.00

📂 Document Type: 3. Long
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:14<00:00,  3.70s/it]
   - Chunk  128: Recall=0.00, Faith=0.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.15s/it]
   - Chunk  256: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.05s/it]
   - Chunk  512: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.16s/it]
   - Chunk 1024: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.03s/it]
   - Chunk 2048: Recall=1.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:14<00:00,  3.72s/it]
   - Chunk 4096: Recall=1.00, Faith=1.00

📂 Document Type: 4. Very Long
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:12<00:00,  3.03s/it]
   - Chunk  128: Recall=0.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:13<00:00,  3.39s/it]
   - Chunk  256: Recall=1.00, Faith=0.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:13<00:00,  3.46s/it]
   - Chunk  512: Recall=0.00, Faith=0.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:13<00:00,  3.40s/it]
   - Chunk 1024: Recall=0.00, Faith=0.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:13<00:00,  3.46s/it]
   - Chunk 2048: Recall=0.00, Faith=1.00
Evaluating:   0%|          | 0/4 [00:00<?, ?it/s]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████| 4/4 [00:14<00:00,  3.53s/it]
   - Chunk 4096: Recall=1.00, Faith=1.00

==================================================
🏆 実験結果：指標別ヒートマップ
==================================================

📊 Context Recall (高いほど良い)
------------------------------
Chunk         128   256   512   1024  2048  4096
Type                                            
1. Short       1.0   1.0   1.0   1.0   1.0   1.0
2. Medium      1.0   1.0   1.0   1.0   1.0   1.0
3. Long        0.0   1.0   1.0   1.0   1.0   1.0
4. Very Long   0.0   1.0   0.0   0.0   0.0   1.0

📊 Context Precision (高いほど良い)
------------------------------
Chunk         128   256   512   1024  2048  4096
Type                                            
1. Short      1.00  1.00   1.0   1.0   1.0   1.0
2. Medium     0.33  1.00   1.0   1.0   1.0   1.0
3. Long       0.00  1.00   0.5   1.0   1.0   1.0
4. Very Long  1.00  0.58   0.0   0.0   1.0   1.0

📊 Faithfulness (高いほど良い)
------------------------------
Chunk         128   256   512   1024  2048  4096
Type                                            
1. Short       1.0   0.5   0.5   0.5   1.0   0.5
2. Medium      1.0   1.0   1.0   1.0   1.0   1.0
3. Long        0.0   1.0   1.0   1.0   1.0   1.0
4. Very Long   1.0   0.0   0.0   0.0   1.0   1.0

📊 Answer Relevancy (高いほど良い)
------------------------------
Chunk         128   256   512   1024  2048  4096
Type                                            
1. Short      0.88  0.85  0.85  0.85  0.83  0.88
2. Medium     0.96  0.96  0.96  0.96  0.96  0.96
3. Long       0.00  0.84  0.84  1.00  0.84  0.88
4. Very Long  0.82  0.90  0.90  0.89  0.83  0.89

✅ 実験完了

```