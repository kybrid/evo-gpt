from pydantic import BaseModel


class ResponseBody(BaseModel):
    message: str
    success: bool
