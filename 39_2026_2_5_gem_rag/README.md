# GEM-RAG Inspired Graph RAG Engine

GEM-RAG の「グラフ＋固有ベクトルで記憶を構造化する」という発想にインスパイアされた、**ミニマルな Graph-based RAG エンジン**です。
`sentence-transformers` でテキストを埋め込み、`NetworkX` の **eigenvector centrality**（固有ベクトル中心性）を使って「情報のハブ度」を計算し、**類似度＋重要度のハイブリッド検索**を行います。

> 注意: 本リポジトリは論文 GEM-RAG を簡略化した **実験用・教育用実装** であり、元論文のアルゴリズム（ユーティリティ質問・ラプラシアン固有分解・テーマノード生成など）をそのまま再現したものではありません。

---

## 特徴

- ✅ 任意のテキストリスト（技術メモ / スライド断片 / 物語の段落など）を入力にできる
- ✅ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` による多言語埋め込み対応
- ✅ 類似度グラフ上の **固有ベクトル中心性** を計算し、「グラフの中で重要なチャンク」をスコアリング
- ✅ クエリに対して「類似度 × 重要度」を混ぜた **ハイブリッドスコア** で検索
- ✅ ノイズチャンク（孤立文）を固有値スコアで自然に抑制
- ✅ 技術記事だけでなく、長文の物語テキストにもそのまま使える

---

## インストール

Python 3.10+ を想定しています。

```bash
pip install -U numpy networkx scikit-learn sentence-transformers
```

---

## 使い方（クイックスタート）

### 1. エンジンの準備

```python
from sentence_transformers import SentenceTransformer
from gemrag_engine import GemRagEngine  # ファイル名に応じてパスを調整

# 日本語対応の軽量モデル（384次元）
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
engine = GemRagEngine(model)
```

### 2. サンプルドキュメントをインデックス

```python
docs = [
    "RAG (Retrieval-Augmented Generation) はLLMに外部知識を与える技術だ。",  # 0: 概要
    "RAGの実装には、一般的にベクトルデータベースが使用される。",            # 1: 技術詳細
    "ベクトル検索では、コサイン類似度を用いて関連文書を探す。",              # 2: 技術詳細
    "GEM-RAGは、グラフ理論を用いてRAGの精度を向上させる手法である。",        # 3: 発展手法
    "GEM-RAGでは、文書間の関係性から固有値を計算し、重要度とする。",          # 4: 3の詳細
    "今日の夕飯はカレーライスにしようと思う。",                              # 5: 完全なノイズ
]

engine.ingest(docs, similarity_threshold=0.3)
```

### 3. 固有値スコアの確認

```python
print("\n--- 各チャンクの固有値スコア (情報の重要度) ---")
for i, score in enumerate(engine.eigen_scores):
    print(f"ID {i}: {score:.4f} | {docs[i][:20]}...")
```

**例:**

```text
--- 各チャンクの固有値スコア (情報の重要度) ---
ID 0: 0.9359 | RAG (Retrieval-Augme...
ID 1: 0.9765 | RAGの実装には、一般的にベクトルデータ...
ID 2: 0.6383 | ベクトル検索では、コサイン類似度を用いて...
ID 3: 0.8743 | GEM-RAGは、グラフ理論を用いてRA...
ID 4: 1.0000 | GEM-RAGでは、文書間の関係性から固...
ID 5: 0.0000 | 今日の夕飯はカレーライスにしようと思う。...
```

- GEM-RAGの中核説明（ID 4）が 1.0
- ノイズ文（ID 5）が 0.0 になり、**自動的に「重要でない」と扱われている**ことが分かります。

### 4. 検索（類似度 × 重要度）

```python
print("\n--- 検索テスト: 'GEM-RAGの仕組みは？' ---")
results = engine.search("GEM-RAGの仕組みは？", top_k=3, alpha=0.4)

for res in results:
    print(f"Score: {res['score']:.4f} (Sim: {res['similarity']:.2f}, Eigen: {res['eigen_score']:.2f})")
    print(f"Content: {res['chunk']}")
    print("-" * 20)
```

**例:**

```text
--- 検索テスト: 'GEM-RAGの仕組みは？' ---
Score: 0.8887 (Sim: 0.72, Eigen: 1.00)
Content: GEM-RAGでは、文書間の関係性から固有値を計算し、重要度とする。
--------------------
Score: 0.8310 (Sim: 0.77, Eigen: 0.87)
Content: GEM-RAGは、グラフ理論を用いてRAGの精度を向上させる手法である。
--------------------
Score: 0.7155 (Sim: 0.32, Eigen: 0.98)
Content: RAGの実装には、一般的にベクトルデータベースが使用される。
--------------------
```

- クエリに直接近い2文（GEM-RAGの説明）が上位に来る
- 類似度はそこまで高くないが「文脈上のハブ」である RAG実装文も、固有値スコアにより 3 位に入ってきます

`alpha` パラメータで

- `alpha = 1.0`: 純粋ベクトル検索
- `alpha = 0.0`: 純粋に固有値スコア順
- `0 < alpha < 1.0`: 類似度と重要度のハイブリッド

を切り替えられます。

---

## story.txt のような長文への適用例

`story.txt` に 1 万字程度の物語が入っている場合、以下のように使えます。

```python
# story.txt を読み込み
with open("story.txt", encoding="utf-8") as f:
    text = f.read()

# 段落ごとに分割（空行区切り）
raw = text.split("\n\n")
docs = [p.strip() for p in raw if p.strip()]

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
engine = GemRagEngine(model)

engine.ingest(docs, similarity_threshold=0.3)

print("\n--- 各チャンクの固有値スコア (情報の重要度) ---")
for i, score in enumerate(engine.eigen_scores):
    print(f"ID {i}: {score:.4f} | {docs[i][:20]}...")
```

物語の場合:

- 手紙の告白文
- 再会シーン
- 物語の転機となる「実家住所の手がかり」など

が高い固有値スコアを持つ「ハブ」として浮かび上がることが確認できます。

---

## 実装上の設計と制約

- グラフ構築:
  - ノード: チャンク（文字列）
  - エッジ重み: 埋め込み間コサイン類似度（閾値 `similarity_threshold` より大きいもののみ）
- スペクトル解析:
  - NetworkX の `eigenvector_centrality` を利用し、**固有ベクトル中心性**を計算
  - 収束しない場合は次数に基づくスコアにフォールバック
- 検索スコア:
  - `final = alpha * similarity + (1 - alpha) * eigen_score`

### 規模・性能

- 現状は小〜中規模（数千チャンク程度）を想定したシンプル実装です。
- 大規模グラフ（数万〜数百万ノード）では `eigenvector_centrality` が重くなるため、`scipy.sparse.linalg.eigs` や独自の power iteration 実装などへの差し替えが必要です。

---

## ファイル構成

```text
.
├── story.txt # サンプル物語テキスト
├── main.ipynb # ノートブック
└── README.md
```
