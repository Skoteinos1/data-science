from pydantic import BaseModel

class User(BaseModel):
    id: str
    role: int
    first_name: str = Field(default=None,)