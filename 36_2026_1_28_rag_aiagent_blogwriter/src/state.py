from typing import TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    topic: str                # テーマ
    research_data: str        # 調査データ
    draft: str                # 原稿
    review_comment: str       # 査読コメント
    reflection: str           # 🧠 修正計画（反省文）を追加
    revision_count: int       # 修正回数