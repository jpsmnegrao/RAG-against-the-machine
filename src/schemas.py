"""mambo jambo"""


import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """mambo"""
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """mambo"""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """mambo"""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """mambo"""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """mambo"""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """mambo"""
    answer: str


class StudentSearchResults(BaseModel):
    """mambo"""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """mambo"""
    search_results: list[MinimalAnswer]
    k: int
