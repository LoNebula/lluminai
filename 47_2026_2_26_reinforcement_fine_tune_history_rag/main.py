import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# ==========================================
# 1. Mock Components (環境のシミュレーション)
# ==========================================

class MockQueryEncoder(nn.Module):
    """
    Qwen-EmbeddingなどのQuery Encoderを模倣したニューラルネット。
    入力IDを埋め込みベクトルに変換します。
    """
    def __init__(self, vocab_size=1000, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, input_ids):
        # 簡易的なMean Pooling
        embeds = self.embedding(input_ids) 
        return self.proj(embeds.mean(dim=1)) 

class MockEnvironment:
    """
    RAGの環境（LLM + Ground Truth）をシミュレート。
    実環境ではここが「LLMへの問い合わせ」と「F1スコア計算」になります。
    """
    def __init__(self, doc_embeddings, ground_truth_indices_map):
        self.doc_embeddings = doc_embeddings
        self.gt_map = ground_truth_indices_map 

    def get_reward(self, query_id, selected_doc_indices):
        """
        報酬関数。
        正解ドキュメントが含まれていれば高い報酬(0.9~1.0)を返します。
        """
        target_doc_idx = self.gt_map[query_id]
        if target_doc_idx in selected_doc_indices:
             return np.random.uniform(0.9, 1.0) 
        else:
             return np.random.uniform(0.0, 0.1) 

# ==========================================
# 2. HARR Core Implementation (論文の実装)
# ==========================================

class HARRRetriever(nn.Module):
    def __init__(self, query_encoder, temperature=1.0):
        super().__init__()
        self.query_encoder = query_encoder
        self.temperature = temperature

    def forward(self, state_input_ids, candidate_doc_embs):
        """
        クエリ(State)と候補ドキュメントの類似度スコアを計算
        """
        state_emb = self.query_encoder(state_input_ids)
        
        # コサイン類似度のための正規化
        state_emb = F.normalize(state_emb, p=2, dim=1)
        candidate_doc_embs = F.normalize(candidate_doc_embs, p=2, dim=2)

        # 内積計算 [Batch, 1, Dim] x [Batch, Pool, Dim]^T -> [Batch, 1, Pool]
        scores = torch.bmm(
            candidate_doc_embs,
            state_emb.unsqueeze(2)
        ).squeeze(2) 
        return scores

    def sample_documents(self, scores, k_retrieve):
        """
        Plackett-Luce Sampling: 確率に基づいてk個のドキュメントを非復元抽出
        """
        batch_size, pool_size = scores.shape
        selected_indices = []
        action_log_probs = []
        
        # 選択済みマスク
        mask = torch.zeros_like(scores, dtype=torch.bool)
        logits = scores / self.temperature
        
        for _ in range(k_retrieve):
            # 選択済みのスコアを -inf にして選ばれないようにする
            masked_logits = logits.clone()
            masked_logits[mask] = float('-inf')
            
            # 確率分布作成
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # サンプリング実行
            action = dist.sample() # [Batch]
            log_prob = dist.log_prob(action)
            
            selected_indices.append(action)
            action_log_probs.append(log_prob)
            
            # マスク更新 (In-place操作を避けるため論理和を使用)
            step_mask = torch.zeros_like(mask).scatter(1, action.unsqueeze(1), True)
            mask = mask | step_mask
            
        selected_indices = torch.stack(selected_indices, dim=1)
        # 軌跡全体の対数確率は各ステップの和
        total_log_probs = torch.stack(action_log_probs, dim=1).sum(dim=1)
        
        return selected_indices, total_log_probs

def compute_grpo_loss(current_log_probs, old_log_probs, rewards, clip_epsilon=0.2):
    """
    GRPO Loss関数の計算
    """
    # 1. Advantageの計算 (グループ内での正規化)
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    advantages = advantages.detach() # 勾配を切る
    
    # 2. Importance Sampling Ratio
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    # 3. Clipped Surrogate Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # 最大化したいのでマイナスをかけて最小化問題にする
    loss = -torch.min(surr1, surr2).mean()
    return loss

# ==========================================
# 3. Main Training Loop (実行デモ)
# ==========================================

def run_harr_demo():
    print("🚀 HARR Training Demo Started...")
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Hyperparameters ---
    vocab_size = 100
    embed_dim = 32
    pool_size = 10     # 候補ドキュメント数
    k_retrieve = 3     # 取得するドキュメント数
    group_size = 8     # GRPOのグループサイズ (1クエリあたりの試行回数)
    steps = 50         # 学習ステップ数
    learning_rate = 0.05
    
    # --- Setup ---
    encoder = MockQueryEncoder(vocab_size, embed_dim)
    retriever = HARRRetriever(encoder, temperature=1.0)
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    
    # ドキュメント埋め込み (固定)
    doc_embs = torch.randn(pool_size, embed_dim)
    doc_embs = F.normalize(doc_embs, p=2, dim=1)
    
    # Query 0 の正解は Doc 0 とする
    gt_map = {0: 0} 
    env = MockEnvironment(doc_embs, gt_map)
    
    # テスト用クエリデータ
    query_input_ids = torch.randint(0, vocab_size, (1, 10)) 
    
    print(f"🎯 Objective: Queryに対し、正解の 'Document #0' をRetrievalさせる")
    print("-" * 40)

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        
        # バッチ作成: 同じクエリをGroup Size分複製して並列試行させる
        batch_input_ids = query_input_ids.repeat(group_size, 1)
        batch_candidate_embs = doc_embs.unsqueeze(0).repeat(group_size, 1, 1)
        
        # --- 1. Experience Collection (Old Policy) ---
        # 現在のポリシーでサンプリングを行い、軌跡データを集める
        with torch.no_grad():
            scores = retriever(batch_input_ids, batch_candidate_embs)
            selected_indices, old_log_probs = retriever.sample_documents(scores, k_retrieve)
        
        # --- 2. Reward Calculation ---
        # 選んだドキュメントに対して報酬をもらう
        rewards = []
        for i in range(group_size):
            indices = selected_indices[i].tolist()
            r = env.get_reward(0, indices) # Query ID 0
            rewards.append(r)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # --- 3. Policy Update (New Policy) ---
        # 勾配計算のために再度Forward計算
        new_scores = retriever(batch_input_ids, batch_candidate_embs)
        new_logits = new_scores / retriever.temperature
        
        # サンプリング時と同じアクションの確率を、勾配付きで再計算する
        new_log_probs_list = []
        mask = torch.zeros_like(new_scores, dtype=torch.bool)
        
        for k in range(k_retrieve):
            actions_at_step = selected_indices[:, k] # Rolloutで選んだアクション
            
            # Masking
            masked_logits = new_logits.clone()
            masked_logits[mask] = float('-inf')
            
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            # 選んだアクションの対数確率
            log_prob = dist.log_prob(actions_at_step)
            new_log_probs_list.append(log_prob)
            
            # マスク更新 (論理和)
            step_mask = torch.zeros_like(mask).scatter(1, actions_at_step.unsqueeze(1), True)
            mask = mask | step_mask
            
        new_log_probs = torch.stack(new_log_probs_list, dim=1).sum(dim=1)
        
        # Loss計算 & バックプロパゲーション
        loss = compute_grpo_loss(new_log_probs, old_log_probs, rewards_tensor)
        loss.backward()
        optimizer.step()
        
        # ログ出力
        if step % 10 == 0:
            avg_reward = rewards_tensor.mean().item()
            # Doc #0 (正解) が含まれていた割合 (Recall@K)
            success_rate = (selected_indices == 0).any(dim=1).float().mean().item()
            print(f"Step {step:02d} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.0%}")

    print("-" * 40)
    print("🎉 Training Finished!")

if __name__ == "__main__":
    run_harr_demo()