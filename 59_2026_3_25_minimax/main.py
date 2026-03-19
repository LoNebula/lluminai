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