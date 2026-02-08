from typing import List

from pydantic import BaseModel
from ai.graphs.rpa_graph.state import ActorResult


class ActorReqDTO(BaseModel):
    thread_id: str
    actors: List[ActorResult]
