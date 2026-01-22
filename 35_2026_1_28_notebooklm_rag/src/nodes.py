import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.state import AgentState

# 環境変数の読み込み
load_dotenv()

key = os.getenv("OPENAI_API_KEY_LLUMINAI")

# モデルの初期化
llm = ChatOpenAI(model="gpt-5.2", temperature=0.7, api_key=key)
search_tool = TavilySearchResults(max_results=3)

def researcher_node(state: AgentState):
    """Web検索を行い、情報を収集する"""
    topic = state["topic"]
    print(f"\n🕵️  Researcher: 「{topic}」について調査しています...")
    
    # Tavilyで検索
    try:
        search_results = search_tool.invoke(topic)
        # 結果を見やすく整形
        context_text = "\n".join(
            [f"- {res['content']} (Source: {res['url']})" for res in search_results]
        )
    except Exception as e:
        context_text = f"検索エラーが発生しました: {e}"
        
    return {"research_data": context_text}

def writer_node(state: AgentState):
    """調査結果と指摘をもとに記事を書く"""
    print("\n🖊️  Writer: 執筆中...")
    
    topic = state["topic"]
    research_data = state.get("research_data", "データなし")
    feedback = state.get("review_comment", "なし")
    current_count = state.get("revision_count", 0)

    prompt_text = f"""
    あなたはZennなどのテックブログで活躍する技術ライターです。
    以下の情報を元に、エンジニア向けの技術記事をMarkdown形式で執筆してください。

    【テーマ】: {topic}
    【調査データ】: {research_data}
    【査読者からの指摘】: {feedback} (※「なし」の場合は初回執筆です)

    要件:
    - 調査データを根拠にする（ハルシネーションを避ける）
    - コードスニペットが必要な場合は、Python等の適切なコードを含める
    - 査読者からの指摘がある場合は、必ず修正して反映する
    """
    
    messages = [("system", prompt_text), ("human", "記事を作成してください。")]
    response = llm.invoke(messages)
    
    return {
        "draft": response.content,
        "revision_count": current_count + 1
    }

def reviewer_node(state: AgentState):
    """記事を査読する"""
    print("\n🧐 Reviewer: 査読中...")
    
    draft = state["draft"]
    
    prompt_text = f"""
    あなたは厳格な技術メディアの編集長です。
    以下の記事ドラフトを批判的にレビューしてください。

    【チェック項目】
    1. 技術的な誤りはないか？
    2. 読者にとって有益な具体性があるか？
    3. 構成は論理的か？

    【出力フォーマット】
    - 問題がない場合: 文頭に「ACCEPT」と記述し、評価コメントを続ける。
    - 修正が必要な場合: 具体的な修正指示（Critique）のみを記述する。

    【対象ドラフト】:
    {draft}
    """
    
    messages = [("system", prompt_text), ("human", "レビューをお願いします。")]
    response = llm.invoke(messages)
    
    return {"review_comment": response.content}
