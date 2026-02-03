# MiniRAG: トポロジー強化型グラフ検索の簡易シミュレーション

このリポジトリは、論文 **"MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation"** (Fan et al., 2025)  で提案された主要アルゴリズムの最小実装（Proof of Concept）です。

**異種グラフインデックス（Heterogeneous Graph Indexing）** と **トポロジー強化グラフ検索（Topology-Enhanced Graph Retrieval）**  のメカニズムをシミュレートし、構造情報がいかにしてSLM（小規模言語モデル）の検索精度を向上させるかをデモします。

> 
> **注記:** 本コードはアルゴリズムの挙動確認を目的としています。重いLLMへの依存を排除し、スコアリングロジック（論文中の式2および式3）  の検証に集中するため、埋め込みベクトルにはランダム値、グラフ操作には `NetworkX` を使用しています。
> 
> 

## 🧩 主な機能 (Key Features)

* **異種グラフ構築**: テキストチャンク（Chunk）とエンティティ（Entity）を、意味的なエッジで結合したグラフ構造を扱います 。


* **エッジ関連性スコアリング ()**: グラフのトポロジー（形状）に基づき、クエリや回答候補に近い「重要な関係性」を特定する式(2)を実装しています 。


* **推論パス探索 ()**: クエリから潜在的な回答へと至る論理的なパスを発見し、ランク付けする式(3)を実装しています 。


* **軽量シミュレーション**: `NetworkX` ベースで動作し、GPU不要でアルゴリズムの挙動を追跡可能です。

## 🚀 インストール (Installation)


*必要なパッケージ:*

* `networkx`
* `numpy`
* `scikit-learn`

## 🏃 使い方 (Usage)

シミュレーションスクリプトを実行します。

```bash
python main.py

```

### 出力例と解説

スクリプトは、「ハウスルール（House Rules）」に関するクエリを投げた際のパス探索をシミュレートします。アルゴリズムはトポロジー的な重要度に基づいて推論パスをランク付けします 。

```text
🔍 Searching MiniRAG Graph...

🏆 Rank 1 (Score: 5.9239)
   Path: ['HouseRules', 'Chunk1', 'Adam']
   Relevant Chunks: ['Chunk1']

🏆 Rank 2 (Score: 3.9492)
   Path: ['HouseRules', 'Chunk1']
   Relevant Chunks: ['Chunk1']

```

この例では、**Rank 1** のスコアが高くなっています。
直接的なリンクである `HouseRules -> Chunk1` (Rank 2) よりも、作成者（Adam）への接続を含むパス `HouseRules -> Chunk1 -> Adam` (Rank 1) の方が、より多くの「重要なエッジ（Key Relationships）」を含んでいるためです 。これにより、MiniRAGは文脈の濃い情報をSLMに提供できます。


## 🧠 理論背景 (Theory)

本実装は、論文中の以下の2つの主要な数式に基づいています。

1. エッジ関連性スコア ($\omega_e$) 

開始ノード（クエリエンティティ）や回答候補ノードへの近接度に基づいて、エッジの重要度を計算します。




$$\omega_{e}(e)=\sum_{\hat{v}_{s}\in\hat{\mathcal{V}}_{s}}count(\hat{v}_{s},\hat{\mathcal{G}}_{e,k})+\sum_{\hat{v}_{a}\in\hat{\mathcal{V}}_{a}}count(\hat{v}_{a},\hat{\mathcal{G}}_{e,k})$$

ここで、$\mathcal{G}_{e,k}$ はエッジ $e$ を中心とした $k$-hop サブグラフを表します 。

2. パススコア ($\omega_p$) 

意味的類似度（$\omega_v$）と構造的接続性（パス上のエッジ重要度の総和など）を組み合わせて、推論パスを評価します。

$$\omega_{p}(p|v_{q})=\omega_{v}(\hat{v}_{s}|v_{q})\cdot(1+\sum_{v\in(p\wedge\hat{v}_{a})}count(v,p)+\sum_{e\in(p\wedge\hat{\mathcal{E}}_{\alpha})}\omega_{e}(e))$$

## 📚 引用 (Citation)

[MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation](https://arxiv.org/abs/2501.06713)