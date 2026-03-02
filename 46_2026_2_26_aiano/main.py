import os
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from openai import OpenAI
import dotenv
dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_LLUMINAI", "mock-key"))

# --- データモデル定義 ---

class Document(BaseModel):
    id: str
    content: str
    metadata: dict = {}

class Span(BaseModel):
    """ドキュメント内のハイライト箇所"""
    doc_id: str
    start: int
    end: int
    text: str

class AnnotationContext(BaseModel):
    """アノテーション作業の現在の状態"""
    query: str
    selected_spans: List[Span] = []
    documents: List[Document] = []

# --- AIANO Blockの基底クラス ---

class BaseBlock:
    def __init__(self, name: str, prompt_template: str = ""):
        self.name = name
        self.prompt_template = prompt_template

    def process(self, context: AnnotationContext) -> str:
        raise NotImplementedError

# --- 各モードの実装 ---

class AISoloBlock(BaseBlock):
    """
    Mode (ii) AI Solo: システムプロンプトのみに基づいて自動生成
    例：ドキュメントから質問(Query)を生成する
    """
    def process(self, context: AnnotationContext) -> str:
        # 文脈として全ドキュメントの内容を結合（簡易実装）
        full_text = "\n".join([d.content for d in context.documents])
        
        prompt = self.prompt_template.format(context=full_text)
        
        print(f"🤖 [AI Solo] Generating for {self.name}...")
        # 実際はここでAPIコール
        # response = client.chat.completions.create(...)
        return f"Generated Content based on {len(full_text)} chars doc."

class CollaborativeBlock(BaseBlock):
    """
    Mode (iii) Human-AI Collaborative: 
    ユーザーの入力(Query) + ハイライト(Spans) + プロンプトで生成
    """
    def process(self, context: AnnotationContext) -> str:
        # ハイライトされた根拠のみを抽出
        evidence_text = "\n".join([f"- {span.text}" for span in context.selected_spans])
        
        if not evidence_text:
            return "⚠️ 根拠箇所がハイライトされていません。"

        # プロンプトの構築
        system_msg = "あなたはRAGデータセット作成のアシスタントです。提供された根拠に基づいて、質問に対する正確な回答を作成してください。"
        user_msg = self.prompt_template.format(
            query=context.query,
            evidence=evidence_text
        )
        
        print(f"🤝 [Collaborative] Generating answer from {len(context.selected_spans)} spans...")
        
        # モックではなく実際に動くコード例（APIキーがあれば動作）
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

# --- 実行デモ ---

def run_aiano_demo():
    # 1. ドキュメントの準備
    docs = [
        Document(id="doc1", content="AIANOは2026年に発表されたアノテーションツールです。"),
        Document(id="doc2", content="このツールを使うと、RAGのデータ作成速度が約2倍になります。"),
        Document(id="doc3", content="ReactとFastAPIで構築されています。")
    ]

    # 2. ユーザーのアクション（質問入力 + ハイライト）
    context = AnnotationContext(
        query="AIANOを使うメリットは何ですか？",
        documents=docs,
        selected_spans=[
            Span(doc_id="doc2", start=0, end=10, text="RAGのデータ作成速度が約2倍になります。")
        ]
    )

    # 3. Collaborative Blockの設定
    answer_block = CollaborativeBlock(
        name="Answer Generation",
        prompt_template="""
        質問: {query}
        
        以下の根拠となるテキストを使用して、質問に答えてください:
        {evidence}
        
        回答:
        """
    )

    # 4. 生成実行
    result = answer_block.process(context)
    
    print("-" * 20)
    print(f"Q: {context.query}")
    print(f"Evidence: {[s.text for s in context.selected_spans]}")
    print(f"A (AI Generated): {result}")
    print("-" * 20)

if __name__ == "__main__":
    run_aiano_demo()