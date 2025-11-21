import requests
import json
import re
from dataclasses import dataclass
from typing import Literal, List, Union, Optional
from time import perf_counter

# ==========================================
# 設定
# ==========================================
MODEL = "llama3.1:8b"
API_URL = "http://localhost:11434/api/generate"

# モード定義
Mode = Literal["no_cot", "long_cot", "short_cot"]

# プロンプトテンプレート（全タスク共通で使えるように調整）
PROMPTS = {
    "no_cot": """
あなたは優秀なアシスタントです。
次の質問に対し、思考プロセスや説明は一切省き、**答えだけ**を出力してください。
最後に必ず「答え: [回答]」の形式で締めくくってください。

Q: {question}
A:
""",
    "long_cot": """
あなたは優秀なアシスタントです。
次の質問に対し、**ステップバイステップで順を追って**論理的に考え、日本語で詳しく説明してください。
考え終わったら、最後に必ず「答え: [回答]」の形式で締めくくってください。

Q: {question}
A:
""",
    "short_cot": """
あなたは優秀なアシスタントです。
次の質問に対し、**重要なポイントだけを1〜2文で**簡潔に説明してください。
そのあと、最後に必ず「答え: [回答]」の形式で締めくくってください。

Q: {question}
A:
""",
}

# ==========================================
# データ定義
# ==========================================
@dataclass
class Sample:
    category: Literal["arithmetic", "logic", "date"] # タスク種別
    question: str
    answer: Union[int, str] # 正解は数値または文字列

# ベンチマーク問題集（3つのカテゴリ）
SAMPLES: List[Sample] = [
    # 1. 合成算術 (Arithmetic)
    Sample("arithmetic", "37 + 48 はいくつですか？", 85),
    Sample("arithmetic", "144 ÷ 12 はいくつですか？", 12),
    
    # 2. 論理パズル (Logic) -> True/False や はい/いいえ で判定
    Sample("logic", "リンゴはミカンより高く、ミカンはバナナより高いです。リンゴはバナナより高いですか？「はい」か「いいえ」で答えてください。", "はい"),
    Sample("logic", "スイッチを押すと電気がつきます。今、電気は消えています。スイッチは押されていますか？「はい」か「いいえ」で答えてください。", "いいえ"),

    # 3. 日付計算 (Date) -> 曜日や日付特定
    Sample("date", "今日は月曜日です。ここから3日後は何曜日ですか？曜日だけ答えてください（例：金曜日）。", "木曜日"),
    Sample("date", "今日は10月1日です。昨日は何月何日でしたか？（例：1月1日）。", "9月30日"),
]

# ==========================================
# 実行・抽出・判定ロジック
# ==========================================
def run_ollama_api(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_tokens": 300
        }
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error: {e}"

def extract_and_judge(text: str, sample: Sample) -> bool:
    """
    モデルの出力テキストから答えを抽出し、正解と比較する
    カテゴリごとに判定ロジックを少し変える
    """
    # 1. 「答え: X」の箇所を探す
    match = re.search(r"答え[:：]\s*(.+)", text)
    
    # 見つからなかった場合、文末付近を見るなどのフォールバックも考えられるが、
    # 今回はプロンプトでフォーマットを強制しているので厳密に判定する。
    if not match:
        return False
    
    extracted_str = match.group(1).strip()

    # --- カテゴリ別判定 ---
    if sample.category == "arithmetic":
        # 数値として一致するか
        # 余計な文字（句読点など）を除去して数値化トライ
        nums = re.findall(r"(\-?\d+)", extracted_str)
        if not nums: return False
        return int(nums[0]) == sample.answer

    elif sample.category == "logic":
        # 部分一致で判定（"はい、そうです" などが含まれる場合のため）
        # 正解が "はい" なら、抽出文字列に "はい" が含まれていればOKとする
        return str(sample.answer) in extracted_str

    elif sample.category == "date":
        # 日付・曜日も部分一致で判定
        return str(sample.answer) in extracted_str

    return False

def eval_mode(mode: Mode):
    prompt_tmpl = PROMPTS[mode]
    results = []

    for s in SAMPLES:
        prompt = prompt_tmpl.format(question=s.question)
        
        start = perf_counter()
        output = run_ollama_api(prompt)
        elapsed = perf_counter() - start

        is_correct = extract_and_judge(output, s)

        results.append({
            "mode": mode,
            "category": s.category,
            "question": s.question,
            "correct": is_correct,
            "latency": elapsed,
            "output_len": len(output),
            # "output": output # ログが見づらくなるので必要ならコメントアウト解除
        })

    return results

# ==========================================
# メイン処理
# ==========================================
def main():
    print(f"Starting benchmark with {MODEL}...")
    all_results = []
    
    for mode in ["no_cot", "short_cot", "long_cot"]:
        print(f"Testing mode: {mode} ...")
        res = eval_mode(mode) # type: ignore
        all_results.extend(res)

    # 集計表示
    from collections import defaultdict
    # (mode, category) ごとの集計
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "latency": 0.0})

    for r in all_results:
        key = (r["mode"], r["category"])
        stats[key]["total"] += 1
        stats[key]["latency"] += r["latency"]
        if r["correct"]:
            stats[key]["correct"] += 1

    print("\n=== Result Summary ===")
    # カテゴリごとに並べて表示
    categories = ["arithmetic", "logic", "date"]
    modes = ["no_cot", "short_cot", "long_cot"]
    
    for cat in categories:
        print(f"\n--- Category: {cat.upper()} ---")
        for mode in modes:
            key = (mode, cat)
            st = stats[key]
            if st["total"] == 0: continue
            acc = st["correct"] / st["total"] * 100
            avg_lat = st["latency"] / st["total"]
            print(f"{mode.ljust(10)} | Acc: {acc:5.1f}% | Avg Latency: {avg_lat:.2f}s")

    # JSON保存
    with open("cot_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nDetails saved to cot_benchmark_results.json")

if __name__ == "__main__":
    main()