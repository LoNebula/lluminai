from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import researcher_node, writer_node, reviewer_node, reflector_node

def router(state: AgentState):
    comment = state["review_comment"]
    count = state["revision_count"]
    
    if "ACCEPT" in comment:
        print("\n✅ Reviewer: 承認！記事が完成しました。")
        return "end"
    elif count >= 3:
        print("\n⚠️ Loop Limit: 修正上限です。")
        return "end"
    else:
        print(f"\n🔄 Reviewer: 修正が必要です。Reflectorに回します。")
        print(f"   指示内容: {comment[:100]}...")
        return "reflector" # ここが変更点！WriterではなくReflectorへ

def main():
    workflow = StateGraph(AgentState)

    # ノード追加
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("reflector", reflector_node) # 新規追加

    # エッジ定義
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "reviewer")
    
    # Reviewerからの分岐 (NGなら Reflector へ)
    workflow.add_conditional_edges(
        "reviewer",
        router,
        {
            "end": END,
            "reflector": "reflector"
        }
    )
    
    # Reflector -> Writer (計画を持って執筆へ戻る)
    workflow.add_edge("reflector", "writer")

    app = workflow.compile()

    print("🚀 Self-Correction AI Writer 起動...")
    topic = input("テーマを入力: ")
    
    final = app.invoke({"topic": topic, "revision_count": 0})
    
    with open("final_article.md", "w", encoding="utf-8") as f:
        f.write(final["draft"])
    print("\n📁 保存完了: final_article.md")

if __name__ == "__main__":
    main()