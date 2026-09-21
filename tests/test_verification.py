"""Automated tests. Run from the project root with:  python -m unittest discover -v

External services are replaced with fakes, so no API keys or internet are needed.
"""
import unittest
from unittest.mock import MagicMock, patch

import requests

from models.verification_result import (
    NO_EVIDENCE_MESSAGE,
    AIAnalysis,
    EvidenceItem,
    Source,
)
from services.ai_service import AIService, AIServiceError, MalformedAIResponse
from services.search_service import SearchError, SearchProvider, TavilySearchProvider
from services.verification_service import VerificationService
from utils.helpers import classify_source, extract_json, is_safe_url, validate_claim


def make_source(name="who.int", stype="Scientific / health organization", snippet="Some text."):
    return Source(title=f"Article on {name}", url=f"https://{name}/a", source_name=name,
                  snippet=snippet, source_type=stype)


class FakeSearch(SearchProvider):
    def __init__(self, results=None, error=None):
        self.results, self.error = results or [], error

    def search(self, query, max_results=5):
        if self.error:
            raise self.error
        return list(self.results)


class FakeAI:
    def __init__(self, analysis=None, error=None):
        self.analysis, self.error, self.calls = analysis, error, 0

    def analyze(self, claim, sources):
        self.calls += 1
        if self.error:
            raise self.error
        return self.analysis


def analysis(verdict):
    return AIAnalysis(verdict=verdict, confidence=0.8, summary="Summary.", reasoning="Reasoning.",
                      evidence=[EvidenceItem(1, "Source 1 says something relevant.")])


class TestCourseScenarios(unittest.TestCase):
    def test_1_supported(self):
        svc = VerificationService(FakeSearch([make_source()]), FakeAI(analysis("SUPPORTED")))
        result = svc.verify("The Earth orbits the Sun once every year.")
        self.assertTrue(result.ok)
        self.assertEqual(result.verdict, "SUPPORTED")
        self.assertEqual(result.sources[0].url, "https://who.int/a")

    def test_2_disputed(self):
        svc = VerificationService(FakeSearch([make_source()]), FakeAI(analysis("DISPUTED")))
        self.assertEqual(svc.verify("Vaccines contain tracking microchips.").verdict, "DISPUTED")

    def test_3_unknown_claim_no_results(self):
        ai = FakeAI(analysis("SUPPORTED"))
        result = VerificationService(FakeSearch([]), ai).verify("My neighbour's cat can predict lotto numbers.")
        self.assertEqual(result.verdict, "UNCLEAR")
        self.assertEqual(result.summary, NO_EVIDENCE_MESSAGE)
        self.assertEqual(ai.calls, 0)  # AI is not asked to guess without evidence

    def test_4_empty_input(self):
        result = VerificationService(FakeSearch(), FakeAI()).verify("   ")
        self.assertEqual(result.error, "Please enter a claim.")

    def test_5_search_failure_is_friendly(self):
        err = SearchError("Unable to retrieve sources at the moment. Please try again later.")
        result = VerificationService(FakeSearch(error=err), FakeAI()).verify("A perfectly normal claim here.")
        self.assertIn("Unable to retrieve sources", result.error)
        self.assertIsNone(result.verdict)


class TestVerificationService(unittest.TestCase):
    def test_ai_failure_keeps_sources(self):
        svc = VerificationService(FakeSearch([make_source()]), FakeAI(error=AIServiceError("AI down.")))
        result = svc.verify("Some longer claim to check.")
        self.assertEqual(result.error, "AI down.")
        self.assertEqual(len(result.sources), 1)

    def test_unexpected_exception_does_not_crash(self):
        svc = VerificationService(FakeSearch([make_source()]), FakeAI(error=RuntimeError("boom")))
        self.assertIsNotNone(svc.verify("Some longer claim to check.").error)

    def test_sources_deduplicated_and_prioritised(self):
        other = make_source("blog.example.com", "Other (not in a recognised category)")
        gov = make_source("cdc.gov", "Government")
        dup = make_source("cdc.gov", "Government")
        chosen = VerificationService(FakeSearch(), FakeAI())._select_sources([other, gov, dup])
        self.assertEqual([s.source_name for s in chosen], ["cdc.gov", "blog.example.com"])

    def test_empty_snippets_dropped(self):
        chosen = VerificationService(FakeSearch(), FakeAI())._select_sources([make_source(snippet="  ")])
        self.assertEqual(chosen, [])

    def test_status_callback(self):
        messages = []
        svc = VerificationService(FakeSearch([make_source()]), FakeAI(analysis("UNCLEAR")))
        svc.verify("Some longer claim to check.", on_status=messages.append)
        self.assertEqual(messages, ["Searching for sources...", "Analyzing available evidence..."])


class TestSearchProvider(unittest.TestCase):
    def test_missing_key(self):
        with self.assertRaises(SearchError):
            TavilySearchProvider(api_key="").search("q")

    def test_timeout(self):
        with patch("services.search_service.requests.post", side_effect=requests.exceptions.Timeout):
            with self.assertRaises(SearchError) as ctx:
                TavilySearchProvider(api_key="k").search("q")
        self.assertIn("too long", str(ctx.exception))

    def test_status_codes(self):
        for code, text in [(401, "API key"), (429, "rate limit"), (500, "Unable to retrieve")]:
            with patch("services.search_service.requests.post", return_value=MagicMock(status_code=code)):
                with self.assertRaises(SearchError) as ctx:
                    TavilySearchProvider(api_key="k").search("q")
            self.assertIn(text, str(ctx.exception))

    def test_parses_results_and_skips_unsafe_urls(self):
        payload = {"results": [
            {"title": "Good", "url": "https://www.cdc.gov/x", "content": "text", "published_date": "2024-01-02"},
            {"title": "Bad", "url": "javascript:alert(1)", "content": "text"},
        ]}
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch("services.search_service.requests.post", return_value=resp):
            sources = TavilySearchProvider(api_key="k").search("q")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_name, "cdc.gov")
        self.assertEqual(sources[0].source_type, "Government")


def ai_response(content, status=200):
    resp = MagicMock(status_code=status)
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


class TestAIService(unittest.TestCase):
    def setUp(self):
        self.ai = AIService(api_key="k", model="m", base_url="http://x/v1")
        self.sources = [make_source(), make_source("cdc.gov", "Government")]

    def run_ai(self, content):
        with patch("services.ai_service.requests.post", return_value=ai_response(content)):
            return self.ai.analyze("claim text here", self.sources)

    def test_valid_response(self):
        out = self.run_ai('```json\n{"verdict":"supported","confidence":0.9,"summary":"S","reasoning":"R",'
                          '"evidence":[{"source":2,"finding":"F"}]}\n```')
        self.assertEqual(out.verdict, "SUPPORTED")
        self.assertEqual(out.evidence[0].source_index, 2)

    def test_invented_source_numbers_dropped_and_verdict_downgraded(self):
        out = self.run_ai('{"verdict":"DISPUTED","confidence":0.9,"summary":"S","reasoning":"R",'
                          '"evidence":[{"source":99,"finding":"made up"}]}')
        self.assertEqual(out.evidence, [])
        self.assertEqual(out.verdict, "UNCLEAR")

    def test_confidence_clamped(self):
        out = self.run_ai('{"verdict":"UNCLEAR","confidence":7,"summary":"S","reasoning":"R","evidence":[]}')
        self.assertEqual(out.confidence, 1.0)

    def test_malformed_response_retries_then_fails(self):
        with patch("services.ai_service.requests.post", return_value=ai_response("not json at all")) as post:
            with self.assertRaises(MalformedAIResponse):
                self.ai.analyze("claim text here", self.sources)
        self.assertEqual(post.call_count, 2)

    def test_invalid_verdict_rejected(self):
        with self.assertRaises(MalformedAIResponse):
            self.run_ai('{"verdict":"TRUE","summary":"S"}')

    def test_error_statuses(self):
        for code, text in [(401, "API key"), (429, "rate limit"), (503, "unavailable")]:
            with patch("services.ai_service.requests.post", return_value=MagicMock(status_code=code)):
                with self.assertRaises(AIServiceError) as ctx:
                    self.ai.analyze("claim text here", self.sources)
            self.assertIn(text, str(ctx.exception))

    def test_network_timeout(self):
        with patch("services.ai_service.requests.post", side_effect=requests.exceptions.Timeout):
            with self.assertRaises(AIServiceError):
                self.ai.analyze("claim text here", self.sources)

    def test_prompt_contains_no_urls(self):
        prompt = AIService._build_user_prompt("claim", self.sources)
        self.assertNotIn("http", prompt)  # the model never sees URLs, so it cannot invent or alter them


class TestHelpers(unittest.TestCase):
    def test_validate_claim(self):
        self.assertEqual(validate_claim("")[1], "Please enter a claim.")
        self.assertIn("too short", validate_claim("hi")[1])
        self.assertIn("too long", validate_claim("x" * 1001)[1])
        self.assertEqual(validate_claim("  Water   boils at 100C  ")[0], "Water boils at 100C")

    def test_classify_source(self):
        self.assertEqual(classify_source("nasa.gov"), "Government")
        self.assertEqual(classify_source("gov.ph"), "Government")
        self.assertEqual(classify_source("mit.edu"), "University / academic")
        self.assertEqual(classify_source("ox.ac.uk"), "University / academic")
        self.assertEqual(classify_source("news.bbc.co.uk"), "Established news organization")
        self.assertTrue(classify_source("random-blog.com").startswith("Other"))

    def test_safe_url_and_json(self):
        self.assertTrue(is_safe_url("https://example.com/a"))
        self.assertFalse(is_safe_url("javascript:alert(1)"))
        self.assertEqual(extract_json('Sure! {"a": 1} done'), {"a": 1})


if __name__ == "__main__":
    unittest.main()
