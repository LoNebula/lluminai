import time
import heapq


class Node:
    def __init__(self, doc_id, parent=None, size=100, cost=10):
        self.doc_id = doc_id
        self.parent = parent
        self.children = {}  # {doc_id: Node}
        self.size = size    # Token size
        self.cost = cost    # Recomputation cost (ms, simplified)
        self.frequency = 0
        self.priority = 0.0
        self.location = "DISK"  # DISK, HOST, GPU
        
        # PGDSF metrics
        self.last_access = 0
    
    def get_path(self):
        path = []
        curr = self
        while curr:
            path.append(curr.doc_id)
            curr = curr.parent
        return path[::-1]


class RAGCacheSimulator:
    def __init__(self, gpu_capacity, host_capacity):
        self.root = Node("ROOT", size=10, cost=1)
        self.root.location = "GPU"
        self.gpu_capacity = gpu_capacity
        self.host_capacity = host_capacity
        self.gpu_usage = 10  # Root size
        self.host_usage = 0
        self.clock = 0.0
        
        # Logical clock L for GDSF-like behavior
        self.L = 0.0 

    def access_sequence(self, doc_sequence):
        """
        Retrieves a sequence of documents (e.g., ["D1", "D2"]).
        Updates the tree and cache status.
        """
        current_node = self.root
        
        for doc_id in doc_sequence:
            self.clock += 1
            
            # 1. Tree Traversal / Creation
            if doc_id not in current_node.children:
                # New node (simulate fetching logic)
                new_node = Node(doc_id, parent=current_node)
                current_node.children[doc_id] = new_node
            
            current_node = current_node.children[doc_id]
            
            # 2. Update Frequency & Priority (PGDSF-like Logic)
            current_node.frequency += 1
            
            # Simple Cost/Size estimation for simulation
            # （論文の式 (1) Priority = Clock + Frequency * (Cost/Size) の簡略版）
            cost_factor = current_node.cost / current_node.size 
            
            # Update Priority
            current_node.priority = self.L + current_node.frequency * cost_factor
            
            # 3. Cache Management (Move to GPU if needed)
            if current_node.location != "GPU":
                print(f"🔄 Miss! Loading {doc_id} to GPU...")
                self._promote_to_gpu(current_node)
            else:
                print(f"✅ Hit! {doc_id} is in GPU.")
                # Update priority for hit case logic (simplified)
                current_node.priority = self.L + current_node.frequency * cost_factor

    def _promote_to_gpu(self, node):
        """Moves a node to GPU, evicting others if necessary."""
        required_size = node.size
        
        # Evict until space is available
        while (self.gpu_capacity - self.gpu_usage) < required_size:
            evicted = self._evict_from_gpu()
            if not evicted:
                raise Exception("OOM: Cannot fit even after eviction!")
        
        # Promote
        if node.location == "HOST":
            self.host_usage -= node.size
        node.location = "GPU"
        self.gpu_usage += node.size
        print(f"   -> Promoted {node.doc_id} to GPU. Usage: {self.gpu_usage}/{self.gpu_capacity}")

    def _evict_from_gpu(self):
        """Finds the lowest priority leaf node in GPU to evict."""
        # Traverse tree to find GPU nodes (simplified DFS)
        stack = [self.root]
        gpu_nodes = []
        while stack:
            n = stack.pop()
            if n.location == "GPU" and n != self.root:  # Don't evict root
                gpu_nodes.append(n)
            for child in n.children.values():
                stack.append(child)
        
        if not gpu_nodes:
            return False

        # In実システム: まず葉ノードから優先的に追い出す（PGDSFの設計方針に対応）
        leaf_candidates = [n for n in gpu_nodes if all(c.location != "GPU" for c in n.children.values())]
        
        if not leaf_candidates:
            # Fallback if no clean leaves (shouldn't happen in strict tree)
            leaf_candidates = gpu_nodes

        victim = min(leaf_candidates, key=lambda x: x.priority)
        
        # Eviction時に L を victim の priority に更新（GDSFのClock更新ルール）
        self.L = max(self.L, victim.priority)
        
        # Demote to Host
        victim.location = "HOST"
        self.gpu_usage -= victim.size
        self.host_usage += victim.size
        print(f"   👋 Evicted {victim.doc_id} (P={victim.priority:.2f}) to HOST.")
        return True


# --- 実行 ---
# GPU容量 350 (小さい設定), Hostは十分大きいと仮定
sim = RAGCacheSimulator(gpu_capacity=350, host_capacity=1000)


print("--- Step 1: Request [D1, D2] ---")
sim.access_sequence(["D1", "D2"]) 


print("\n--- Step 2: Request [D1, D3] (D1 should hit) ---")
sim.access_sequence(["D1", "D3"])


print("\n--- Step 3: Request [D4, D5] (Forces eviction) ---")
sim.access_sequence(["D4", "D5"])


print("\n--- Step 4: Request [D1, D2] again (D1 should stay/return) ---")
sim.access_sequence(["D1", "D2"])