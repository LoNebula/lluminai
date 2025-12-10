import os
from collections import Counter
from typing import List
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

# OpenAI API Client (または Ollama/Local LLM)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_solution(problem: str, model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """
    単一の解を生成する関数。
    Temperatureを上げて多様性を持たせるのがポイント。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "あなたは数学の専門家です。最終的な答えだけをシンプルに出力してください。"},
            {"role": "user", "content": problem}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def best_of_n_strategy(problem: str, n: int = 5) -> str:
    """
    Best-of-N 戦略（Majority Voting）の実装。
    N回生成を行い、最も頻出する答えを採用する（Self-Consistency）。
    数学や論理パズルなど、正解が一つに定まるタスクで有効。
    """
    solutions = []
    print(f"--- Running Best-of-{n} Strategy ---")
    
    # 1. Generate N solutions
    for i in range(n):
        sol = generate_solution(problem)
        solutions.append(sol)
        print(f"Attempt {i+1}: {sol}")
    
    # 2. Majority Vote
    # 簡易的に文字列完全一致でカウントするが、実務では正規化が必要
    counts = Counter(solutions)
    most_common_solution, count = counts.most_common(1)[0]
    
    confidence = count / n
    print(f"Selected Answer: {most_common_solution} (Confidence: {confidence:.2f})")
    
    return most_common_solution

# --- 実験 ---
# 典型的な、LLMが間違えやすい論理パズル
problem = "私が6歳のとき、妹は私の半分の年齢でした。そして私が10歳のとき、弟は私の半分の年齢でした。私が70歳になったとき、妹と弟の年齢の合計は何歳ですか？"

print("【Single Shot】")
print(generate_solution(problem, temperature=0.7))

print("\n【Best-of-5】")
best_of_n_strategy(problem, n=5)