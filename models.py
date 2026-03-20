from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class WhisperSegment(BaseModel):
    """Represents a segment in the transcription"""
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.1
    speaker: Optional[str] = None  # For speaker diarization

class ActionItem(BaseModel):
    """An action item assigned during the meeting"""
    assignee: str
    task: str
    deadline: Optional[str] = None

class Decision(BaseModel):
    """A decision made during the meeting"""
    decision: str
    context: str

class UnresolvedQuestion(BaseModel):
    """A question raised but not answered"""
    question: str
    raised_by: Optional[str] = None

class Participant(BaseModel):
    """A meeting participant"""
    speaker_label: str
    role: Optional[str] = None

class FollowUp(BaseModel):
    """An item needing follow-up"""
    item: str
    owner: Optional[str] = None

class RiskOrBlocker(BaseModel):
    """A risk or blocker mentioned in the meeting"""
    description: str
    raised_by: Optional[str] = None

class MeetingIntelligence(BaseModel):
    """Structured meeting intelligence extracted by LLM"""
    summary: str
    action_items: List[ActionItem] = []
    decisions: List[Decision] = []
    unresolved_questions: List[UnresolvedQuestion] = []
    key_topics: List[str] = []
    participants: List[Participant] = []
    follow_ups: List[FollowUp] = []
    risks_and_blockers: List[RiskOrBlocker] = []
    model: Optional[str] = None
    generation_time_seconds: Optional[float] = None

class AnalyzeRequest(BaseModel):
    """Request body for the /v1/meeting/analyze endpoint"""
    transcript: str
    model: Optional[str] = None

class TranscriptionResponse(BaseModel):
    """Represents the response format for transcription"""
    text: str
    segments: Optional[List[WhisperSegment]] = None
    language: Optional[str] = None
    task: str = "transcribe"
    duration: Optional[float] = None
    model: Optional[str] = None
    meeting_intelligence: Optional[MeetingIntelligence] = None

    class Config:
        json_schema_extra = {"example": {"text": "Hello world", "segments": []}}

    def dict(self, **kwargs):
        """Custom dict method to handle response format"""
        result = super().dict(**kwargs)
        if not self.segments:
            result.pop("segments", None)
        if not self.meeting_intelligence:
            result.pop("meeting_intelligence", None)
        return result

class ModelInfo(BaseModel):
    """Information about a model available in the API"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: str
    parent: Optional[str] = None

class ModelList(BaseModel):
    """List of available models"""
    object: str = "list"
    data: List[ModelInfo]
