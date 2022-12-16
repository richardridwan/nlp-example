from pydantic import BaseModel
from typing import List

class InputQuery(BaseModel):
    sentence: str

class Extraction(BaseModel): List[str]

class BaseResponse(BaseModel):
    message: str