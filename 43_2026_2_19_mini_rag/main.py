import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimpleMiniRAG:
    def __init__(self):
        # 異種グラフの初期化
        self.G = nx.Graph()
        # エンティティとチャンクの埋め込み（シミュレーション用）
        self.embeddings = {}

    def add_chunk(self, chunk_id, content, embedding):
        """チャンクノードの追加"""
        self.G.add_node(chunk_id, type='chunk', content=content)
        self.embeddings[chunk_id] = embedding

    def add_entity(self, entity_id, entity_type, embedding):
        """エンティティノードの追加"""
        self.G.add_node(entity_id, type='entity', entity_type=entity_type)
        self.embeddings[entity_id] = embedding

    def add_relation(self, source, target, relation_type, description=""):
        """エッジの追加（Entity-Entity または Entity-Chunk）"""
        self.G.add_edge(source, target, relation=relation_type, desc=description)

    def _get_k_hop_subgraph(self, edge, k=2):
        """エッジ周辺のk-hopサブグラフを取得"""
        u, v = edge
        nodes_u = nx.single_source_shortest_path_length(self.G, u, cutoff=k).keys()
        nodes_v = nx.single_source_shortest_path_length(self.G, v, cutoff=k).keys()
        return set(nodes_u) | set(nodes_v)

    def calculate_edge_score(self, start_nodes, answer_candidates, k=1):
        """Eq(2): エッジの重要度スコア ω_e を計算"""
        edge_scores = {}
        
        for u, v in self.G.edges():
            subgraph_nodes = self._get_k_hop_subgraph((u, v), k)
            
            # クエリエンティティ(start_nodes)が近傍にいくつあるか
            score_s = sum(1 for n in start_nodes if n in subgraph_nodes)
            # 答え候補(answer_candidates)が近傍にいくつあるか
            score_a = sum(1 for n in answer_candidates if n in subgraph_nodes)
            
            edge_scores[(u, v)] = score_s + score_a
            
        return edge_scores

    def search(self, query_embedding, start_nodes, answer_candidates, top_k=3):
        """Eq(3): パス探索とスコアリング"""
        # 1. エッジスコア計算
        edge_scores = self.calculate_edge_score(start_nodes, answer_candidates)
        
        paths = []
        
        # start_nodesから始まるパスを探索（簡易的に長さ2までとする）
        for start_node in start_nodes:
            # クエリとの類似度 ω_v (Cosine Similarity)
            sim = cosine_similarity(
                [query_embedding], 
                [self.embeddings[start_node]]
            )[0][0]
            
            # 2-hop先のノードまで探索
            for target in nx.single_source_shortest_path_length(self.G, start_node, cutoff=2).keys():
                if target == start_node: continue
                
                # 単純パスを取得
                for path in nx.all_simple_paths(self.G, start_node, target, cutoff=2):
                    # Eq(3)の実装
                    # パスに含まれるエッジのスコア合計
                    path_edge_score_sum = 0
                    path_edges = list(zip(path, path[1:]))
                    for edge in path_edges:
                        # 無向グラフなので順序ケア
                        e_score = edge_scores.get(edge, edge_scores.get((edge[1], edge[0]), 0))
                        path_edge_score_sum += e_score
                    
                    # 答え候補が含まれているか
                    contains_answer = sum(1 for n in path if n in answer_candidates)

                    # 最終スコア ω_p
                    path_score = sim * (1 + contains_answer + path_edge_score_sum)
                    
                    paths.append({
                        "path": path,
                        "score": path_score,
                        "chunks": [n for n in path if self.G.nodes[n].get('type') == 'chunk']
                    })

        # スコア順にソート
        sorted_paths = sorted(paths, key=lambda x: x['score'], reverse=True)
        return sorted_paths[:top_k]

# --- 実行パート ---

# 1. データのセットアップ（論文の「ハウスルール」の例を模倣）
rag = SimpleMiniRAG()

# 埋め込みはランダムベクトルで代用
v_dim = 64
emb_lihua = np.random.rand(v_dim)
emb_adam = np.random.rand(v_dim)
emb_rule = np.random.rand(v_dim)
emb_wifi = np.random.rand(v_dim)
emb_query = emb_rule + np.random.normal(0, 0.1, v_dim) # Queryは"House Rules"に近いと仮定

# ノード追加
rag.add_entity("LiHua", "Person", emb_lihua)
rag.add_entity("Adam", "Person", emb_adam)
rag.add_entity("HouseRules", "Concept", emb_rule)
rag.add_chunk("Chunk1", "Adam: Keep noise down at night.", np.random.rand(v_dim))
rag.add_chunk("Chunk2", "Wifi password is Family123.", np.random.rand(v_dim))

# エッジ追加（関係性を定義）
rag.add_relation("LiHua", "Adam", "friend")
rag.add_relation("Adam", "Chunk1", "author_of")
rag.add_relation("HouseRules", "Chunk1", "mentioned_in") # HouseRulesはChunk1に関連
rag.add_relation("Adam", "Chunk2", "author_of")

# 2. 検索シミュレーション
# クエリ: "What are the House Rules?" -> Entity extraction: "HouseRules"
# 想定: HouseRulesからChunk1へのパスが高く評価されるはず

print("🔍 Searching MiniRAG Graph...")
results = rag.search(
    query_embedding=emb_query,
    start_nodes=["HouseRules"],     # クエリから抽出されたエンティティ
    answer_candidates=["Chunk1"]    # 本来は推論で候補を出すが、ここではChunk1をターゲットと仮定
)

for i, res in enumerate(results):
    print(f"\n🏆 Rank {i+1} (Score: {res['score']:.4f})")
    print(f"   Path: {res['path']}")
    print(f"   Relevant Chunks: {res['chunks']}")