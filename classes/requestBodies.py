from pydantic import BaseModel


class RequestBody(BaseModel):
    key: str


class ChatRequest(RequestBody):
    chatInput: str
