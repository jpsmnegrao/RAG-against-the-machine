"""Schemas for RAG search and evaluation pipeline."""


import uuid
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a minimal source reference."""
    file_path: str
    first_character_index: int
    last_character_index: int


class IndexedChunk(BaseModel):
    """Represents a text chunk with source metadata."""
    minimalsource: MinimalSource
    text: str


class UnansweredQuestion(BaseModel):
    """Represents an unanswered RAG question."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents an answered RAG question with sources and answer."""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represents minimal search results for a question."""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents a generated answer and sources."""
    answer: str


class StudentSearchResults(BaseModel):
    """Container schema for student search results."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Container schema for search or answer results."""
    search_results: list[MinimalAnswer]
    k: int
