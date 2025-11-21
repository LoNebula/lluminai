from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from memory import search_memory, add_memory
import ollama


class AgentState(TypedDict):
    user_id: str
    messages: List[Dict]
    memory: List[str]


# --- Memory retrieve node ---
def retrieve_memory(state: AgentState):
    last = state["messages"][-1]["content"]
    memories = search_memory(state["user_id"], last)
    state["memory"] = [m["content"] for m in memories]
    return state


# --- LLM think node ---
def think(state: AgentState):
    last = state["messages"][-1]["content"]
    memories = state["memory"]

    prompt = f"""
You are a personal AI secretary with persistent memory.

User message:
{last}

Relevant memories:
{memories}

Respond concisely.
If there's anything you should remember for the future, output:
MEM_TO_SAVE: <text>.
"""

    res = ollama.chat(
        model="llama3.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    output = res["message"]["content"]

    # --- Extract memory to save ---
    if "MEM_TO_SAVE:" in output:
        mem = output.split("MEM_TO_SAVE:", 1)[1].strip()
        add_memory(state["user_id"], mem)

    state["messages"].append({"role": "assistant", "content": output})
    return state
