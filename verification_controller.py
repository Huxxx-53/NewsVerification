"""
controllers/verification_controller.py

Coordinates the verification flow so the Flet views never call
services or the model directly:

    claim -> search_service -> ai_service -> history_model -> result dict

Each step can fail independently; this module turns those failures into
a consistent {"ok": bool, ...} shape the UI can branch on without
needing to know which layer raised the error.
"""

from models import history_model
from services import search_service
from services import ai_service
from utils.helpers import clean_text


def run_verification(raw_claim: str) -> dict:
    """
    Run the full verification pipeline for a single claim.

    Returns one of:
        {"ok": True, "record": <history record dict>}
        {"ok": False, "stage": "input" | "search" | "ai" | "storage",
         "message": <user-facing error string>}

    Design choices that follow the spec:
    - A search failure does NOT get silently turned into DISPUTED/UNCLEAR;
      it's reported as an error so the UI can let the user retry.
    - Finding zero sources is NOT an error - it's a valid outcome that
      ai_service.analyze_claim() turns into an UNCLEAR status.
    """
    claim = clean_text(raw_claim)

    if not claim:
        return {"ok": False, "stage": "input", "message": "Please enter a claim to verify."}

    if len(claim) < 5:
        return {"ok": False, "stage": "input",
                 "message": "That claim looks too short to search for. Please add more detail."}

    # Step 1: find sources. A hard failure here (network/API down) is
    # reported to the user rather than treated as "no evidence found".
    try:
        sources = search_service.find_sources(claim)
    except search_service.SearchServiceError as exc:
        return {"ok": False, "stage": "search",
                 "message": f"Source search failed: {exc}"}

    # Step 2: AI analysis over whatever sources we found (possibly none).
    try:
        ai_result = ai_service.analyze_claim(claim, sources)
    except ai_service.AIServiceError as exc:
        return {"ok": False, "stage": "ai", "message": str(exc)}

    # Step 3: persist to local history.
    try:
        record = history_model.save_result(claim, ai_result, sources)
    except Exception as exc:  # JSON file I/O issues, disk full, etc.
        return {"ok": False, "stage": "storage",
                 "message": f"Could not save this result to history: {exc}"}

    return {"ok": True, "record": record}


def get_claims_checked_count() -> int:
    """Used by the home dashboard to show 'Claims Checked: N'."""
    return history_model.get_history_count()


def get_history() -> list:
    """Used by the history screen."""
    return history_model.load_history()


def get_result(record_id: int):
    """Used when opening a single history item."""
    return history_model.get_result_by_id(record_id)


def delete_history_item(record_id: int) -> bool:
    return history_model.delete_result(record_id)
