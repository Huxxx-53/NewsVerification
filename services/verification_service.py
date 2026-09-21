"""Coordinates the full pipeline:

Claim -> validate -> search queries -> Search -> Sources -> filter/prioritise
      -> AI analysis -> VerificationResult
"""
import logging
from typing import Callable, List, Optional

from models.verification_result import (
    NO_EVIDENCE_MESSAGE,
    VERDICT_UNCLEAR,
    Source,
    VerificationResult,
)
from services.ai_service import AIService, AIServiceError
from services.search_service import SearchError, SearchProvider
from utils.helpers import SOURCE_PRIORITY, truncate, validate_claim

logger = logging.getLogger(__name__)

GENERIC_ERROR = "Something went wrong while verifying this claim. Please try again later."


class VerificationService:
    def __init__(
        self,
        search_provider: SearchProvider,
        ai_service: AIService,
        max_sources: int = 8,
        results_per_query: int = 5,
    ):
        self.search_provider = search_provider
        self.ai_service = ai_service
        self.max_sources = max_sources
        self.results_per_query = results_per_query

    # ------------------------------------------------------------------ steps
    @staticmethod
    def generate_queries(claim: str) -> List[str]:
        """Two simple queries: the claim itself, and the claim phrased as a fact-check."""
        return [truncate(claim, 250), "fact check: " + truncate(claim, 200)]

    def _select_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicates and empty results, then list better-known source types first."""
        seen, unique = set(), []
        for s in sources:
            key = s.url.split("#")[0].rstrip("/").lower()
            if key in seen or not s.snippet.strip():
                continue
            seen.add(key)
            unique.append(s)
        unique.sort(key=lambda s: SOURCE_PRIORITY.get(s.source_type, 3))  # stable sort
        return unique[: self.max_sources]

    # ------------------------------------------------------------------- main
    def verify(self, claim: str, on_status: Optional[Callable[[str], None]] = None) -> VerificationResult:
        def status(message: str) -> None:
            if on_status:
                on_status(message)

        clean_claim, error = validate_claim(claim)
        if error:
            return VerificationResult(claim=claim or "", error=error)

        queries = self.generate_queries(clean_claim)

        status("Searching for sources...")
        collected: List[Source] = []
        search_errors: List[str] = []
        for query in queries:
            try:
                collected.extend(self.search_provider.search(query, self.results_per_query))
            except SearchError as exc:
                search_errors.append(str(exc))
            except Exception:  # never let an unexpected provider bug crash the app
                logger.exception("Unexpected search failure")
                search_errors.append("Unable to retrieve sources at the moment. Please try again later.")

        if not collected and search_errors:
            return VerificationResult(claim=clean_claim, search_queries=queries, error=search_errors[0])

        sources = self._select_sources(collected)
        if not sources:
            return VerificationResult(
                claim=clean_claim,
                verdict=VERDICT_UNCLEAR,
                summary=NO_EVIDENCE_MESSAGE,
                reasoning="The search did not return any usable sources for this claim.",
                search_queries=queries,
            )

        status("Analyzing available evidence...")
        try:
            analysis = self.ai_service.analyze(clean_claim, sources)
        except AIServiceError as exc:
            # Keep the sources so the user can still read them.
            return VerificationResult(claim=clean_claim, sources=sources, search_queries=queries, error=str(exc))
        except Exception:
            logger.exception("Unexpected AI failure")
            return VerificationResult(claim=clean_claim, sources=sources, search_queries=queries, error=GENERIC_ERROR)

        return VerificationResult(
            claim=clean_claim,
            verdict=analysis.verdict,
            confidence=analysis.confidence,
            summary=analysis.summary,
            reasoning=analysis.reasoning,
            evidence=analysis.evidence,
            sources=sources,
            search_queries=queries,
        )
