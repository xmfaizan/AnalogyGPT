from pydantic import BaseModel
from typing import Optional

class AnalogyRequest(BaseModel):
    question: str
    context: Optional[str] = None
    difficulty_level: Optional[str] = "medium"  # easy, medium, hard

class AnalogyResponse(BaseModel):
    analogy: str
    explanation: str
    original_question: str
    success: bool
    error_message: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    message: str