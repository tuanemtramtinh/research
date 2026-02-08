from typing import List

from pydantic import BaseModel
from ai.graphs.rpa_graph.state import UseCase


class UsecaseReqDTO(BaseModel):
    thread_id: str
    usecases: List[UseCase]
