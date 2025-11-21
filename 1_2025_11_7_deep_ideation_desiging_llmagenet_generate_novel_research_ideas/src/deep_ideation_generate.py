import subprocess
import networkx as nx
from pathlib import Path
from datetime import datetime
import json

# === 設定 ===
MODEL = "gpt-oss:20b"
keywords = ["retrieval", "contrastive learning", "multimodal", "graph reasoning", "alignment"]

# === キーワードネットワーク構築 ===
G = nx.Graph()
for i in range(len(keywords) - 1):
    G.add_edge(keywords[i], keywords[i + 1])

# === Ollama用プロンプト ===
prompt = f"""
You are a research ideation agent.
Given these keywords: {keywords},
find two concepts that are not directly connected in the network,
but could form a novel and meaningful research idea.
Describe the idea briefly in one paragraph, focusing on novelty and feasibility.
"""

def run_ollama(model: str, prompt: str) -> str:
    """
    Ollama を Windows でも安全に呼ぶ。
    - text=True は使わず bytes で受け取る
    - 入力も UTF-8 で渡す
    - 失敗時は stderr を UTF-8 で表示
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),   # ← プロンプトは stdin で UTF-8
            capture_output=True,            # ← bytes を受け取る
            check=False
        )
    except FileNotFoundError:
        print("❌ ollama が見つかりません。PATH を確認してください。")
        return ""

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        print("❌ Ollama 実行エラー:\n" + err)
        return ""

    # 標準出力を UTF-8 で安全にデコード
    out = result.stdout.decode("utf-8", errors="replace").strip()
    return out

# === 実行 ===
print(f"🚀 Generating research idea via Ollama model: {MODEL} ...")
content = run_ollama(MODEL, prompt)

if not content:
    print("⚠️ No content generated.")
    exit()

print("\n💡 Generated Idea:\n")
print(content)

# === 出力保存 ===
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

# テキスト保存
txt_path = Path(__file__).with_name("generated_ideas.txt")
text_block = f"[{timestamp}]\n{content}\n{'-'*60}\n"
with txt_path.open("a", encoding="utf-8") as txt_file:
    txt_file.write(text_block)

# JSON保存
json_path = Path(__file__).with_name("generated_ideas.json")
idea_entry = {"timestamp": timestamp, "content": content}
if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []
data.append(idea_entry)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\n✅ Saved to generated_ideas.txt, and generated_ideas.json")
