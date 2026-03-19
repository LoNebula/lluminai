---
title: "エラー放置でバグは直る？LangGraph×MiniMax M2.7の自己修正ループ限界検証と導入判断"
emoji: 🎛️
type: "tech"
topics: ["langgraph", "minimax", "python", "aiagent", "ollama"]
published: false
---

## はじめに

ルミナイR&Dチームの宮脇彰梧です。
現在はマルチモーダルAIの研究を行う大学院生として、生成AIやAIエージェントの技術を実践的に探求しています。

皆さんは休日の夜、APIサーバーのエラーログを眺めながら「誰か勝手に直しておいてくれないかな…」と天を仰いだことはありませんか？
今回は、リリースされたばかりの自己進化型モデル「MiniMax M2.7」と「LangGraph」を組み合わせ、**エラーを出して自己修正するコーディング・ループ**を構築。どこまでAIに丸投げできるのか限界テストを行いました。

**この記事で学べること**
* LangGraphによる「自己修正（Self-Correction）ループ」の基礎設計
* `minimax-m2.7:cloud`（Ollama経由）のコーディング能力とコスパ評価
* 無限ループの恐怖と、エージェント開発におけるリアルな落とし穴

**記事の構成**
1. なぜこのテーマを選んだのか？
2. この技術は採用すべき？
3. 技術の仕組みとプロトタイプ構想
4. 実装・検証
5. 筆者の考察
6. まとめ

**結論を先に言います**
個人開発やPoC環境において、**「エラー原因の切り分けから初期パッチ作成まで」を任せるなら、この構成は即採用すべき**です。圧倒的な低コスト（入力 $0.30 / 1M tokens）でClaude 3.5 Sonnetに迫る自律性を見せます。ただし、安全装置（最大試行回数の制限など）を入れないと平気で無意味な修正ループを回し続けるので、設計には少しコツが要ります。



## 概要

LangGraphのCyclic Graph（巡回グラフ）を利用し、MiniMax M2.7にコード生成・実行・エラー分析・修正を繰り返させる自律エージェントを試作。バグありのFastAPIサーバーを自動修復できるかを検証し、コストと精度のトレードオフから導入の是非を判断します。

### 1. なぜこのテーマを選んだのか？

Agentic Workflow（エージェント的ワークフロー）の要は、「失敗から学ぶ」ことです。コードを書いて、実行して、エラーを読んで、また直す。
しかし、これをClaude 3.5 SonnetやGPT-4oでやると、**API代がとんでもない速度で溶けます。** ループが長引くほどコンテキスト（履歴）が肥大化し、気づけばコーヒー1杯分のコストが消し飛ぶことも。

そこにOllama経由でシームレスに叩ける `minimax-m2.7:cloud` が登場しました。コーディング能力が高い上に、入力トークン単価が激安（$0.30 / 1M）。
「これなら、多少アホな修正ループに入ってもお財布が痛まないのでは？」という、開発者特有のケチな発想から検証をスタートしました。

### 2. この技術は採用すべき？

LangGraphを使った自律修正ループとM2.7の組み合わせ。現場に導入すべきか、ぶっちゃけベースで整理します。

#### 💡 Why / When：なぜ必要か？いつ使うべきか？
* **APIコストを気にせず、エージェントの限界まで試行錯誤させたい時**に最適です。
* **複雑な環境依存エラー**や、人間が読むのも嫌になるような**長大なスタックトレースの解析**を、夜間にバッチ処理的にぶん投げておく用途に輝きます。

#### ⚠️ 向き・不向き
* **クリティカルに効くケース**：型の不一致、DBのマイグレーション忘れ、ライブラリのバージョン競合など、エラーログに答えがモロに書いてあるバグの自動修正。
* **ぶっちゃけ微妙なケース**：仕様そのものが間違っているロジックバグ。AIは「エラーが出ないこと」をゴールにするため、エラーをもみ消すような（`try-except pass` みたいな）悪魔の修正を提案しがちです。

#### 📊 導入判断のための比較表

「バグ自動修正」のアプローチとして、どれを選ぶべきかの判断基準です。

| アプローチ | コスト感 | 成功率（体感） | セットアップ | 採用すべきケース |
| :--- | :--- | :--- | :--- | :--- |
| **LangGraph + M2.7 (本記事)** | 🟢 激安 ($0.30/1M) | 🟡 中〜高 (無限ループ注意) | 🔺 面倒 | **PoC・個人開発・API代を極限まで削りたい時** |
| LangGraph + Claude 3.5 Sonnet | 🔺 高額 | 👑 最高レベル | 🔺 面倒 | 予算があり、絶対に直したい本番バグの調査 |
| Claude Code (Ollama連携) | 🟡 中〜安 | 🟢 高い | 🟢 楽 | ターミナルから離れず、対話的に直したい時 |
| 自分が休日に泣きながら直す | 👑 無料 | 🔺 疲労度による | 👑 不要 | 納期が今日で、誰も助けてくれない時 |

**【結論】**
手軽さなら「Claude Code」などの完成されたツールに軍配が上がります。しかし、**「自社の業務フローに合わせたカスタム自動デバッグ基盤」をスクラッチで組むなら、M2.7のコスパは圧倒的**です。



### 3. 技術の仕組みと関連調査

今回のプロトタイプのアーキテクチャを解説します。

#### 構想：宮脇製「自動デバッグ・ループ」
LangGraphは、エージェントの「状態（State）」を定義し、ノード（処理）間をループさせることができるフレームワークです。今回は以下のようなシンプルな巡回グラフ（Cyclic Graph）を設計しました。

1. **Coder Node**：バグの内容と現在のコードを受け取り、修正案を生成して上書きする。
2. **Executor Node**：修正されたコードをローカルで実行し、テストする。
3. **Evaluator Node**：実行結果を判定。成功すれば終了（END）。エラーが出ればログを取得し、再び Coder Node へ戻す。

#### なぜM2.7なのか？
M2.7の最大の特徴は「自己進化（Self-Evolution）」による強化学習を経たことで、**「一度間違えたアプローチを繰り返さない」能力が高い**とされています。これをLangGraphのループ内でどう発揮するかが見ものです。



### 4. 実装・検証（判断の裏付けとして）

手元の環境で「エラーを出して自己修正するループ」を実際に動かしてみましょう。Ollamaで `minimax-m2.7:cloud` がプルされている前提です。

#### 準備：必要なパッケージ
```bash
pip install langgraph langchain-openai pydantic
```

#### 【コピペで動く】完全版・自動デバッグループ
以下のコードを `main.py` として保存し、実行してみてください。
このスクリプトは、実行されると自動的にバグを含んだ `app.py`（テーブル作成を忘れたSQLiteアプリ）を生成し、M2.7がそれを読み取って修正ループを開始します。

※もしプロンプトの指示が甘いと、AIがパニックを起こして無限に `pip install sqlite3` を叩き始める狂気のBotが爆誕するため、安全装置（`MAX_RETRIES`）とプロンプトのガードレールをしっかり組み込んであります。

```python
# main.py
import os
import subprocess
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ==========================================
# 1. 準備：検証用の「バグありアプリ」を生成する
# ==========================================
TARGET_FILE = "app.py"

# わざと「DBの初期化（CREATE TABLE）を忘れた」コードを用意
INITIAL_BUGGY_CODE = """
import sqlite3
import json

def init_db():
    # 本来はここで CREATE TABLE が必要だが、忘れている！
    pass

def get_todos():
    conn = sqlite3.connect("test.db")
    cursor = conn.cursor()
    # テーブルが存在しないため、ここでエラー（sqlite3.OperationalError）が起きる
    cursor.execute("SELECT * FROM todos")
    todos = cursor.fetchall()
    conn.close()
    return json.dumps(todos)

if __name__ == "__main__":
    init_db()
    print(get_todos())
"""

def setup_target_file():
    with open(TARGET_FILE, "w") as f:
        f.write(INITIAL_BUGGY_CODE.strip())
    print(f"📁 検証用ファイル '{TARGET_FILE}' を作成しました（バグ混入済み）")


# ==========================================
# 2. LangGraphエージェントの構築
# ==========================================
# Ollamaのローカルエンドポイント経由でM2.7:cloudを呼び出す
llm = ChatOpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # Ollamaの場合は適当な文字列でOK
    model="minimax-m2.7:cloud",
    temperature=0.2,
)

# 状態（State）の定義
class GraphState(TypedDict):
    code: str
    error: str
    iteration: int

MAX_RETRIES = 3 # 【重要】無限ループ防止の安全装置

# 2-A. Coderノード：エラーを読んでコードを書き直す
def coder_node(state: GraphState):
    print(f"\n🔄 [Coder] M2.7が修正を試みます... (試行回数: {state['iteration']})")
    
    # プロンプト：pip installの幻覚を防ぐための指示を含める
    prompt = ChatPromptTemplate.from_messages([
        ("system", "あなたは優秀なPythonエンジニアです。以下のコードを実行した際に出たエラーを解決し、修正済みの完全なPythonコードのみを出力してください。Markdownの```pythonタグは不要です。また、標準ライブラリ（sqlite3など）をpip installしようとしないでください。"),
        ("user", "【現在のコード】\n{code}\n\n【エラー内容】\n{error}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"code": state["code"], "error": state["error"]})
    
    # AIが余計なMarkdownタグを出力したときの泥臭い除去処理
    cleaned_code = response.content.replace("```python", "").replace("```", "").strip()
    
    return {"code": cleaned_code, "iteration": state["iteration"] + 1}

# 2-B. Executorノード：コードを保存して実行する
def executor_node(state: GraphState):
    print(f"🚀 [Executor] '{TARGET_FILE}' を上書きして実行中...")
    
    # 対象ファイルを上書き保存
    with open(TARGET_FILE, "w") as f:
        f.write(state["code"])
    
    # サブプロセスで実行してエラーをキャッチ
    result = subprocess.run(
        ["python", TARGET_FILE], 
        capture_output=True, 
        text=True
    )
    
    error_msg = result.stderr.strip()
    if error_msg:
        # ログが長すぎる場合は切り詰める（コンテキスト肥大化防止）
        print(f"❌ [Executor] エラー発生:\n{error_msg[:300]}...\n")
    else:
        print(f"✅ [Executor] 実行成功！出力:\n{result.stdout.strip()}\n")
        
    return {"error": error_msg}

# 2-C. 条件分岐：エラーがあればCoderへ戻る、成功なら終了
def route_check(state: GraphState):
    if state["error"] == "":
        return END
    if state["iteration"] >= MAX_RETRIES:
        print(f"⚠️ [警告] 試行回数が {MAX_RETRIES} 回に達したため、ループを強制終了します。")
        return END
    return "coder"

# グラフの組み立て
workflow = StateGraph(GraphState)
workflow.add_node("coder", coder_node)
workflow.add_node("executor", executor_node)

workflow.add_edge(START, "executor") # 最初は「実行」からスタートしてエラーを拾う
workflow.add_conditional_edges("executor", route_check)
workflow.add_edge("coder", "executor")

app = workflow.compile()


# ==========================================
# 3. 実行メイン処理
# ==========================================
if __name__ == "__main__":
    setup_target_file()
    
    # 初期コードの読み込み
    with open(TARGET_FILE, "r") as f:
        current_code = f.read()
        
    print("🔥 LangGraph × MiniMax M2.7 自動デバッグループを開始します...\n")
    
    # 初期状態を渡してエージェント起動
    app.invoke({
        "code": current_code, 
        "error": "", 
        "iteration": 0
    })
```

#### 実際の実行ログ（見事な一発クリア！）
これを手元で実行した実際のログがこちらです。

```text
python main.py
📁 検証用ファイル 'app.py' を作成しました（バグ混入済み）
🔥 LangGraph × MiniMax M2.7 自動デバッグループを開始します...

🚀 [Executor] 'app.py' を上書きして実行中...
❌ [Executor] エラー発生:
Traceback (most recent call last):
  File "C:\env\note\lluminai\59_2026_3_25_minimax\app.py", line 19, in <module>
    print(get_todos())
          ^^^^^^^^^^^
  File "C:\env\note\lluminai\59_2026_3_25_minimax\app.py", line 12, in get_todos
    cursor.execute("SELECT * FROM todos")
sqlite3.Operation...

🔄 [Coder] M2.7が修正を試みます... (試行回数: 0)
🚀 [Executor] 'app.py' を上書きして実行中...
✅ [Executor] 実行成功！出力:
[]
```

**お見事です！** 初回の実行で正しく `sqlite3.OperationalError` を検知した後、M2.7が呼び出され、見事に1発（試行回数: 0）で修正を完了させました。
出力結果が `[]`（空のJSON配列）になっているのは、M2.7がコード内に `CREATE TABLE IF NOT EXISTS todos` の処理を正しく挿入し、正常に空のテーブルからデータを引っぱってこれた証拠です。


### 5. 筆者の考察

この「一発クリア」のログを見て、私はAIエージェントの進化の早さに改めて恐怖すら覚えました。

これまでの軽量ローカルモデルであれば、「エラーログとコードを渡す」→「変な位置にパッチを当てる」→「Syntax Errorを起こす」という不毛なやり取りが数回続くのがデフォでした。
しかし、今回のM2.7は「エラーの根本原因（テーブル未作成）を特定し、元のロジックを壊さない位置（`init_db`関数内）に適切なDDL文を挿入する」という、シニアエンジニアと全く同じ思考プロセスを1回の推論で完了させています。

しかも、これをOllama経由でローカルのPythonスクリプトから叩いており、裏側では $0.30/1M という破格のコストで処理されています。

ただ、忘れてはいけないのが、プロンプトによる手綱引きの重要性です。
コードにも書きましたが、システムプロンプトに「標準ライブラリをpip installしようとしないで」というガードレールを入れておかなければ、AIは「モジュールがないなら外から持ってこよう」という誤った局所最適化に走りがちです。

完全に自動化された魔法なんてありません。
エージェント開発の真髄は、「AIの不器用な思考プロセス」を予測し、LangGraphのノード（状態管理）とプロンプトで**いかにAIが転ばないようなレールを敷いてあげるか**に尽きます。そのレールさえ敷ければ、M2.7は現場の泥臭いバグ取りにおいて、最強の相棒になります。

### 6. まとめ

1. **安さと賢さは正義**：M2.7は激安かつ高精度なので、LangGraphで大量の試行錯誤をさせる自律ループのバックエンドとして即戦力レベルです。
2. **無限ループの罠**：エージェントを組む際は、必ず `max_retries` などのリミッターと、ハルシネーションを予測したガードレール（プロンプト）を設計してください。
3. **導入の判断**：個人開発やPoCなら今すぐ採用すべきです。API代を気にせず、自分の代わりに休日のバグ取りを丸投げできます。


---
*執筆：宮脇 彰梧（ルミナイ株式会社 / Lluminai）*
---
【現在採用強化中です！】
- AIエンジニア
- PM/PdM
- 戦略投資コンサルタント

▼代表とのカジュアル面談URL

https://pitta.me/matches/VCmKMuMvfBEk
