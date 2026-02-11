from datetime import datetime
from typing import Any, List, Optional

from beanie import Document
from pydantic import Field

from ai.graphs.rpa_graph.state import ActorResult, UseCase


class Diagram(Document):
    """
    Lưu kết quả sau khi user hoàn thành step-3 (use case diagram).
    Source of truth: requirement_text, use_cases, actors.
    """

    # Ai tạo (nếu có auth)
    user_id: Optional[str] = Field(default=None, description="ID user tạo diagram")

    # Dữ liệu gốc từ RPA
    requirement_text: List[str] = Field(
        default_factory=list,
        description="Danh sách câu requirement gốc",
    )
    use_cases: List[UseCase] = Field(
        default_factory=list,
        description="Danh sách use case đã qua review",
    )
    actors: List[ActorResult] = Field(
        default_factory=list,
        description="Danh sách actor (canonical + aliases + sentence_idx)",
    )

    # Tham chiếu session LangGraph (optional)
    thread_id: Optional[str] = Field(default=None, description="thread_id từ graph run")

    # Snapshot diagram cho frontend (optional, để hiển thị/history nhanh)
    diagram_snapshot: Optional[dict[str, Any]] = Field(
        default=None,
        description="nodes + links từ _to_diagram_data",
    )

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "diagrams"
        # indexes = [IndexModel("user_id"), IndexModel("created_at")]  # nếu cần query nhanh
