import ollama
import numpy as np
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Embedding ---
def embed(text: str):
    res = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return np.array(res["embedding"]).tolist()


# --- Insert memory ---
def add_memory(user_id: str, content: str):
    emb = embed(content)

    res = supabase.table("memory_records").insert({
        "user_id": user_id,
        "content": content,
        "embedding": emb
    }).execute()

    return res


# --- Search memory ---
def search_memory(user_id: str, query: str, top_k=3):
    q_emb = embed(query)

    result = supabase.rpc(
        "match_memory",
        {
            "query_embedding": q_emb,
            "match_threshold": 0.75,
            "match_count": top_k
        }
    ).execute()

    return result.data
