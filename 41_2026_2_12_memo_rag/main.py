import os
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import dotenv

# ==========================================
# 0. 設定とデータの準備
# ==========================================

dotenv.load_dotenv()
# 環境変数名が異なる場合は適宜修正してください
key = os.getenv("OPENAI_API_KEY_LLUMINAI") 
os.environ["OPENAI_API_KEY"] = key

# 物語データの読み込み
with open("./story.txt", "r", encoding="utf-8") as f:
    STORY_TEXT = f.read()

# ==========================================
# 1. MemoRAG クラスの実装（LangChain対応版）
# ==========================================

class ConceptualMemoRAG:
    def __init__(self, llm_fast, llm_smart, retriever):
        self.memory_model = llm_fast
        self.generator_model = llm_smart
        self.retriever = retriever
        self.global_memory = ""

    def memorize(self, long_context: str):
        """Step 1: Memory Formation"""
        print("\n🧠 [Step 1] Forming Global Memory...")
        
        # 物語用にプロンプトを少し調整（登場人物やプロット重視）
        prompt = (
            f"以下のテキストは長い物語です。\n"
            f"後の質問に答えるために、この物語の「あらすじ」「主要な登場人物の関係性」「結末に至る重要な伏線」を\n"
            f"詳細かつ簡潔に要約して記憶メモリを作成してください。\n\n"
            f"--- 物語 ---\n{long_context[:15000]}...\n-------------------" # トークン溢れ防止のためtruncate
        )
        
        response = self.memory_model.invoke(prompt)
        self.global_memory = response.content
        print(f"✅ Memory Created (Length: {len(self.global_memory)} chars)")
        print(f"   (Content Preview: {self.global_memory[:50]}...)")

    def recall_clues(self, query: str) -> List[str]:
        """Step 2: Clue Generation"""
        print(f"\n🤔 [Step 2] Thinking about clues for: '{query}'")
        
        prompt = (
            f"あなたは物語全体の「グローバルメモリ」を持っています。\n"
            f"ユーザーの質問に答えるために、物語の「どの場面」を探せばよいか、具体的な「手がかり（Clues）」を出力してください。\n"
            f"手がかりは検索クエリとして使われます。\n\n"
            f"【メモリ】: {self.global_memory}\n"
            f"【質問】: {query}\n\n"
            f"検索クエリとして使える短いフレーズを箇条書きで3つ出力してください。"
        )
        
        response = self.memory_model.invoke(prompt)
        clues = [line.strip().replace("- ", "").replace("・", "") for line in response.content.split('\n') if line.strip()]
        
        print(f"💡 Generated Clues: {clues}")
        return clues

    def retrieve_evidence(self, clues: List[str]) -> str:
        """Step 3: Retrieval"""
        print("\n🔍 [Step 3] Retrieving evidence based on clues...")
        
        aggregated_evidence = set()
        for clue in clues:
            # ★ここを修正しました (get_relevant_documents -> invoke)
            docs = self.retriever.invoke(clue)
            for d in docs:
                aggregated_evidence.add(d.page_content)
        
        final_evidence = "\n\n".join(list(aggregated_evidence))
        print(f"📚 Retrieved {len(aggregated_evidence)} chunks of evidence.")
        return final_evidence

    def generate_response(self, query: str, evidence: str) -> str:
        """Step 4: Final Generation"""
        print("\n📝 [Step 4] Generating final answer...")
        
        prompt = (
            f"以下の【検索された証拠】に基づいて、質問に感情豊かに答えてください。\n"
            f"物語の文脈を反映させてください。\n\n"
            f"【質問】: {query}\n\n"
            f"【検索された証拠】:\n{evidence}"
        )
        
        response = self.generator_model.invoke(prompt)
        return response.content

# ==========================================
# 2. 実行パイプライン
# ==========================================

def main():
    print("⚙️ Initializing Vector Database...")
    
    # テキストをチャンク分割
    chunks = [p.strip() for p in STORY_TEXT.split("\n\n") if p.strip()]
    documents = [Document(page_content=c) for c in chunks]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # モデルの準備
    llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    llm_smart = ChatOpenAI(model="gpt-4o", temperature=0)

    rag = ConceptualMemoRAG(llm_fast=llm_fast, llm_smart=llm_smart, retriever=retriever)

    # 1. 記憶形成
    rag.memorize(STORY_TEXT)

    # 2. 質問
    query = "藤原健一が15年前に白石幸子の前から姿を消した本当の理由は何ですか？また、二人の結末はどうなりましたか？"
    
    # 3. 実行
    clues = rag.recall_clues(query)
    evidence = rag.retrieve_evidence(clues)
    final_answer = rag.generate_response(query, evidence)

    print("\n" + "="*50)
    print("🤖 Final Answer:")
    print("="*50)
    print(final_answer)

if __name__ == "__main__":
    main()