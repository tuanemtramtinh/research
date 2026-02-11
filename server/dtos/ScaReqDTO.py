from pydantic import BaseModel


class ScaReqDTO(BaseModel):
    thread_id: str
