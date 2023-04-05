from pydantic import BaseModel

class responseBody(BaseModel):
    message: str
    success: bool