from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    top_k: int = 2
    database: str = "faiss"

class TrainingExample(BaseModel):
    input: str
    output: str

class FineTuneRequest(BaseModel):
    training_data: List[TrainingExample]
    epochs: int = 3
