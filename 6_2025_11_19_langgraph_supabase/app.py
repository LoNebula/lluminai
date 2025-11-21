from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent import AgentState, retrieve_memory, think


graph = StateGraph(AgentState)
memory = MemorySaver()

graph.add_node("retrieve", retrieve_memory)
graph.add_node("think", think)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "think")
graph.add_edge("think", END)

app = graph.compile(checkpointer=memory)

message = "明日は何する日？？データを参照して答えて．"


if __name__ == "__main__":
    state = {
        "user_id": "user_123",
        "messages": [{
            "role": "user", 
            "content": f"{message}"}],
            "memory": []
    }

    res = app.invoke(state, config={"configurable": {"thread_id": "1"}})
    print(res["messages"][-1]["content"])
