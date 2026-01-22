import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from src.state import AgentState

load_dotenv()

key = os.getenv("OPENAI_API_KEY_LLUMINAI")

llm = ChatOpenAI(model="gpt-5.2", temperature=0.7, api_key=key)
search_tool = TavilySearchResults(max_results=3)

def researcher_node(state: AgentState):
    topic = state["topic"]
    print(f"\n🕵️  Researcher: 「{topic}」について調査中...")
    try:
        results = search_tool.invoke(topic)
        context = "\n".join([f"- {r['content']} (url: {r['url']})" for r in results])
    except:
        context = "調査失敗"
    return {"research_data": context}

def writer_node(state: AgentState):
    topic = state["topic"]
    data = state["research_data"]
    reflection = state.get("reflection", "") # 反省文を取得
    count = state.get("revision_count", 0)

    # 初稿か修正稿かで表示を変える
    if count == 0:
        print("\n🖊️  Writer: 初稿を執筆中...")
        instruction = "調査データを元に、構成案を作成し記事を執筆してください。"
    else:
        print(f"\n🖊️  Writer: 修正執筆中（{count}回目）...")
        # 反省文をプロンプトに注入！
        instruction = f"""
        あなたは前回の執筆で指摘を受けました。
        以下の【修正計画】を厳密に守り、記事を全面的に書き直してください。
        
        【修正計画（Reflectorからの指示）】:
        {reflection}
        """

    prompt = f"""
    あなたはテックブログのライターです。
    テーマ: {topic}
    調査データ: {data}
    
    指示: {instruction}
    
    出力はMarkdown形式のみにしてください。
    """
    
    res = llm.invoke(prompt)
    return {"draft": res.content, "revision_count": count + 1}

def reviewer_node(state: AgentState):
    print("\n🧐 Reviewer: 査読中...")
    draft = state["draft"]
    
    prompt = f"""
    あなたは厳格な編集長です。記事を評価してください。
    
    合格基準:
    1. 技術的に正確か
    2. 具体的なコードや事例があるか
    
    合格なら「ACCEPT」とだけ出力。
    不合格なら、**具体的な指摘事項（Critique）のみ**を出力してください。
    
    記事:
    {draft}
    """
    res = llm.invoke(prompt)
    return {"review_comment": res.content}

def reflector_node(state: AgentState):
    """🧠 新追加: なぜダメだったかを分析し、修正プランを立てる"""
    print("\n🧠 Reflector: 反省会中（修正計画の策定）...")
    
    draft = state["draft"]
    critique = state["review_comment"]
    
    prompt = f"""
    あなたはライターのメンターです。
    以下の記事ドラフトに対し、査読者から指摘が入りました。
    
    【指摘】: {critique}
    【ドラフト】: {draft}
    
    Writerが次に何をすべきか、具体的な**「修正計画（Step-by-Step Action Plan）」**を作成してください。
    感情的な言葉は不要です。やるべきタスクを3点以内でリストアップしてください。
    """
    
    res = llm.invoke(prompt)
    return {"reflection": res.content}