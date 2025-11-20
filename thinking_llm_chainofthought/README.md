# 🧠 Chain-of-Thought Benchmark for Thinking LLMs

**Evaluate “no CoT / short CoT / long CoT” reasoning modes on local LLMs (Ollama).**

このリポジトリは、Chain-of-Thought（CoT）推論の **長さによる性能差** を
ローカルLLM（Ollama）で簡易ベンチマークするためのツールです。

---

## 📌 目的

* **CoTなし（no CoT）**
* **短いCoT（short CoT）**
* **長いCoT（long CoT）**

の3種類について、
算術・論理・日付推論のミニベンチを実行し、モデルの挙動を比較します。

CoTを盲信せず、
「どれくらいの思考をさせるとモデルが最も安定するのか？」
を理解することが目的です。

---

## 🚀 実行環境

* **Python 3.9+**
* **Ollama**（ローカル推論環境）
* **llama3.2**（推奨・任意で差し替え可）

インストール：
[https://ollama.ai](https://ollama.ai)

---

## ▶️ 実行方法

```bash
python main.py
```

実行すると、次の3モードのベンチマークが走ります：

| mode        | 説明              |
| ----------- | --------------- |
| `no_cot`    | 思考プロセスなしで答えだけ出す |
| `short_cot` | 1〜2文の短い説明＋答え    |
| `long_cot`  | 詳しいステップバイステップ推論 |

---

## 📊 出力例

```
=== Result Summary ===

--- Category: ARITHMETIC ---
no_cot     | Acc: 50.0% | Avg Latency: 0.12s
short_cot  | Acc: 100%  | Avg Latency: 0.26s
long_cot   | Acc: 100%  | Avg Latency: 0.48s

--- Category: LOGIC ---
no_cot     | Acc: 50.0% | Avg Latency: 0.11s
short_cot  | Acc: 50.0% | Avg Latency: 0.24s
long_cot   | Acc: 50.0% | Avg Latency: 0.46s
```

※ モデルによって結果は大きく変わります。

また、詳細ログは JSON に保存されます：

```
cot_benchmark_results.json
```

---

## 🧩 タスク内容

ベンチマークは3タイプの推論タスクから構成されています：

### 1. 合成算術（Arithmetic）

* 加算・減算・乗算・除算
* 最もCoTの効果が分かりやすいタスク

### 2. 論理パズル（Logic）

* 簡単な三段論法・因果関係推論
* CoTが必ずしも有効とは限らない領域

### 3. 日付推論（Date Reasoning）

* 「◯曜日から3日後」などの計算
* 小型LLMが苦手とするジャンル

---

## 🏗 コード構成

```
main.py                 # ベンチマーク本体
cot_benchmark_results.json  # 実行結果（自動生成）
```

### 主な関数

| 関数名                   | 役割                          |
| --------------------- | --------------------------- |
| `run_ollama_api()`    | Ollama API を叩いて推論結果を取得      |
| `extract_and_judge()` | モデル出力から答えを抽出し正誤判定           |
| `eval_mode()`         | no/short/long CoT で全サンプルを評価 |
| `main()`              | 全モード × 全カテゴリのベンチ実行と集計       |

---

## 🔧 MODELS の差し替え

`main.py` 冒頭の `MODEL` を変更すれば自由に使えます。

```python
MODEL = "llama3.1:8b"
# MODEL = "mistral:latest"
# MODEL = "qwen2.5:7b"
# MODEL = "phi3:mini"
```

特に **Small LLM × 長いCoTの相性問題** を体感するのに最適です。

---