"""Small helper functions: input validation, URL handling, source labels, JSON parsing."""
import json
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

MIN_CLAIM_LENGTH = 8
MAX_CLAIM_LENGTH = 1000

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_claim(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (clean_claim, error_message). Exactly one of them is None."""
    if raw is None or not raw.strip():
        return None, "Please enter a claim."
    cleaned = re.sub(r"\s+", " ", raw).strip()
    cleaned = "".join(ch for ch in cleaned if ch.isprintable())  # drop control chars
    if len(cleaned) < MIN_CLAIM_LENGTH:
        return None, "That is too short to verify. Please enter a full statement or headline."
    if len(cleaned) > MAX_CLAIM_LENGTH:
        return None, (
            f"That claim is too long ({len(cleaned)} characters). "
            f"Please shorten it to {MAX_CLAIM_LENGTH} characters or fewer."
        )
    return cleaned, None


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def is_safe_url(url: str) -> bool:
    """Only plain http(s) links with a host are ever shown or made clickable."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def get_domain(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def markdown_safe_url(url: str) -> str:
    return url.replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def escape_md(text: str) -> str:
    """Stop '$' from being rendered as LaTeX by Streamlit's markdown."""
    return (text or "").replace("$", "\\$")


# ---------------------------------------------------------------------------
# Source categories (a transparent label, NOT a numeric credibility score)
# ---------------------------------------------------------------------------

GOVERNMENT = "Government"
ACADEMIC = "University / academic"
SCIENTIFIC = "Scientific / health organization"
FACT_CHECK = "Fact-checking organization"
NEWS = "Established news organization"
OTHER = "Other (not in a recognised category)"

# Lower number = listed first. Being listed first does NOT mean "true".
SOURCE_PRIORITY = {GOVERNMENT: 0, ACADEMIC: 1, SCIENTIFIC: 1, FACT_CHECK: 2, NEWS: 2, OTHER: 3}

_SCIENTIFIC = {
    "who.int", "un.org", "nature.com", "science.org", "thelancet.com", "nejm.org",
    "bmj.com", "cochrane.org", "mayoclinic.org", "royalsociety.org", "nationalacademies.org",
    "esa.int", "cern.ch",
}
_FACT_CHECK = {
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org", "leadstories.com",
    "factcheck.afp.com",
}
_NEWS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org", "pbs.org", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "wsj.com", "bloomberg.com", "aljazeera.com",
    "dw.com", "france24.com", "cbsnews.com", "nbcnews.com", "abcnews.go.com", "cnn.com",
    "rappler.com", "inquirer.net", "philstar.com", "straitstimes.com", "abc.net.au",
    "cbc.ca", "economist.com", "ft.com",
}


def _matches(domain: str, known: set) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in known)


def classify_source(domain: str) -> str:
    """Label a domain using simple, explainable rules."""
    parts = domain.split(".")
    if parts[-1] in ("gov", "mil") or (len(parts) >= 2 and parts[-2] == "gov"):
        return GOVERNMENT
    if parts[-1] == "edu" or (len(parts) >= 2 and parts[-2] in ("edu", "ac")):
        return ACADEMIC
    if _matches(domain, _SCIENTIFIC):
        return SCIENTIFIC
    if _matches(domain, _FACT_CHECK):
        return FACT_CHECK
    if _matches(domain, _NEWS):
        return NEWS
    return OTHER


# ---------------------------------------------------------------------------
# JSON parsing for AI output
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and chatter."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data
