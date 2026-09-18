"""
services/ai_service.py

Sends the claim + retrieved evidence to Claude (Anthropic API) and asks
for a structured verdict. Kept separate from the UI and from search_service.

The AI is explicitly instructed to:
- Only reason from the evidence it was given (no inventing sources/quotes)
- Return UNCLEAR when evidence is insufficient or conflicting
- Never assume "no evidence found" means the claim is false
"""

import json
import os
import re

from anthropic import Anthropic, APIError, APIConnectionError, AuthenticationError

ALLOWED_STATUSES = {"SUPPORTED", "DISPUTED", "UNCLEAR"}


class AIServiceError(Exception):
    """Raised when the AI call fails or returns something unusable."""
    pass


def _get_client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AIServiceError(
            "No ANTHROPIC_API_KEY found. Copy .env.example to .env and add your key."
        )
    return Anthropic(api_key=api_key)


def _build_system_prompt() -> str:
    return (
        "You are a careful claim-verification assistant used in a news "
        "verification app. You will be given a user's claim and a list of "
        "web sources (title, url, snippet) retrieved by a search engine.\n\n"
        "Rules you MUST follow:\n"
        "1. Base your analysis ONLY on the provided sources. Do not invent "
        "sources, URLs, or quotations that are not in the input.\n"
        "2. Do not claim a source says something it does not say.\n"
        "3. Treat your own reasoning as interpretation, not proof - keep "
        "the 'evidence' field limited to things actually present in the "
        "sources.\n"
        "4. If the sources are empty, insufficient, outdated, or conflicting, "
        "return status UNCLEAR. Never assume a claim is false just because "
        "no supporting evidence was found.\n"
        "5. Respond with ONLY a single JSON object, no markdown fences, no "
        "preamble, no explanation outside the JSON. The JSON must have "
        "exactly these keys:\n"
        '   "status": one of "SUPPORTED", "DISPUTED", "UNCLEAR"\n'
        '   "summary": a short (2-4 sentence) plain-language explanation\n'
        '   "evidence": a list of short strings, each a specific evidence point\n'
        '   "confidence": one of "low", "moderate", "high"\n'
    )


def _build_user_prompt(claim: str, sources: list) -> str:
    if not sources:
        sources_block = "(No sources were found by the search service.)"
    else:
        lines = []
        for i, s in enumerate(sources, start=1):
            lines.append(
                f"Source {i}:\n"
                f"Title: {s.get('title', '')}\n"
                f"URL: {s.get('url', '')}\n"
                f"Snippet: {s.get('snippet', '') or '(no snippet available)'}\n"
            )
        sources_block = "\n".join(lines)

    return (
        f"Claim to verify:\n\"{claim}\"\n\n"
        f"Retrieved sources:\n{sources_block}\n\n"
        "Analyze the claim against these sources and respond with the JSON "
        "object described in your instructions."
    )


def _extract_json(text: str) -> dict:
    """
    Best-effort extraction of a JSON object from the model's reply, in case
    it wraps the JSON in markdown fences or adds stray whitespace.
    """
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block as a last resort.
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
        raise AIServiceError("The AI response was not valid JSON.")


def analyze_claim(claim: str, sources: list) -> dict:
    """
    Send the claim and sources to Claude and return a validated dict:
        {"status": str, "summary": str, "evidence": [str, ...], "confidence": str}

    Raises AIServiceError on any failure (missing key, network error,
    invalid/unparseable response) so the controller can show a clear
    message instead of a crash.
    """
    if not claim or not claim.strip():
        raise AIServiceError("Cannot analyze an empty claim.")

    client = _get_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

    try:
        response = client.messages.create(
            model=model,
            max_tokens=800,
            system=_build_system_prompt(),
            messages=[
                {"role": "user", "content": _build_user_prompt(claim, sources)}
            ],
        )
    except AuthenticationError as exc:
        raise AIServiceError("Invalid Anthropic API key. Check your .env file.") from exc
    except APIConnectionError as exc:
        raise AIServiceError("Could not reach the AI service. Check your internet connection.") from exc
    except APIError as exc:
        raise AIServiceError(f"AI service error: {exc}") from exc

    raw_text = "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )

    if not raw_text.strip():
        raise AIServiceError("The AI returned an empty response.")

    result = _extract_json(raw_text)

    # Validate and normalize the structure so the UI can trust it.
    status = str(result.get("status", "")).strip().upper()
    if status not in ALLOWED_STATUSES:
        status = "UNCLEAR"

    summary = str(result.get("summary", "")).strip() or "No explanation was provided."

    evidence = result.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(item).strip() for item in evidence if str(item).strip()]

    confidence = str(result.get("confidence", "moderate")).strip().lower()
    if confidence not in {"low", "moderate", "high"}:
        confidence = "moderate"

    # If there were no sources at all, force UNCLEAR regardless of what the
    # model returned - this enforces rule 4 even if the model slips up.
    if not sources and status != "UNCLEAR":
        status = "UNCLEAR"
        summary = ("No reliable sources were found for this claim, so it "
                    "cannot be confirmed or disputed. " + summary)

    return {
        "status": status,
        "summary": summary,
        "evidence": evidence,
        "confidence": confidence,
    }
