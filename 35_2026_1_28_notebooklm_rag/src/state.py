from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    topic: str                # 記事のテーマ
    research_data: str        # 検索結果のまとめ
    draft: str                # 現在の原稿
    review_comment: str       # 査読コメント
    revision_count: int       # 修正回数（ループ防止用）