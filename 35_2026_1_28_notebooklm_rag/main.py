from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import researcher_node, writer_node, reviewer_node

def router(state: AgentState):
    """査読結果を見て、終了するか書き直すかを決める"""
    comment = state["review_comment"]
    count = state["revision_count"]
    max_revisions = 3 # 最大修正回数
    
    if "ACCEPT" in comment:
        print("\n✅ Reviewer: 承認しました！")
        return "end"
    elif count >= max_revisions:
        print("\n⚠️ Reviewer: 修正回数が上限に達しました。強制終了します。")
        return "end"
    else:
        print(f"\n🔄 Reviewer: 修正指示があります（{count}/{max_revisions}回目）。Writerに戻します。")
        print(f"   指示内容: {comment[:100]}...") # 長いので先頭だけ表示
        return "writer"

def main():
    # 1. グラフの定義
    workflow = StateGraph(AgentState)

    # 2. ノードの追加
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # 3. エッジ（流れ）の定義
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")

    # 4. 条件付きエッジ（分岐）
    workflow.add_conditional_edges(
        "reviewer",
        router,
        {
            "writer": "writer",
            "end": END
        }
    )

    # 5. コンパイル
    app = workflow.compile()

    # 6. 実行
    print("🚀 AI編集部を起動します...")
    input_topic = input("記事のテーマを入力してください（例：LangGraphの最新機能）: ")
    
    initial_state = {
        "topic": input_topic,
        "revision_count": 0,
        "review_comment": "" # 初期化
    }

    # グラフを実行し、最終状態を取得
    final_state = app.invoke(initial_state)

    # 7. 結果の出力
    print("\n" + "="*50)
    print(" 🎉 完成した記事")
    print("="*50 + "\n")
    print(final_state["draft"])
    
    # ローカルファイルにも保存
    filename = "output_article.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_state["draft"])
    print(f"\n📁 記事を {filename} に保存しました。")

if __name__ == "__main__":
    main()
