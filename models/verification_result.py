"""Data structures shared by all services and the UI."""
from dataclasses import asdict, dataclass, field
from typing import List, Optional

VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_DISPUTED = "DISPUTED"
VERDICT_UNCLEAR = "UNCLEAR"
ALLOWED_VERDICTS = (VERDICT_SUPPORTED, VERDICT_DISPUTED, VERDICT_UNCLEAR)

NO_EVIDENCE_MESSAGE = (
    "No sufficient evidence was found to confidently evaluate this claim."
)


@dataclass
class Source:
    """One real search result. Only the search service creates these."""
    title: str
    url: str
    source_name: str
    published_date: Optional[str] = None
    snippet: str = ""
    source_type: str = "Other"


@dataclass
class EvidenceItem:
    """A finding taken from one retrieved source (source_index is 1-based)."""
    source_index: int
    text: str


@dataclass
class AIAnalysis:
    """Validated output of the AI service."""
    verdict: str
    confidence: float
    summary: str
    reasoning: str
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Everything the UI needs to display one verification."""
    claim: str
    verdict: Optional[str] = None
    confidence: Optional[float] = None
    summary: str = ""
    reasoning: str = ""
    evidence: List[EvidenceItem] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    error: Optional[str] = None  # user-friendly message when something failed

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)
