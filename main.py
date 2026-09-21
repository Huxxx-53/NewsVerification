"""News Verification App - Streamlit user interface and entry point.

Run with:  streamlit run main.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env locally; on Streamlit Cloud, secrets are provided as env vars

from models.verification_result import (  # noqa: E402
    VERDICT_DISPUTED,
    VERDICT_SUPPORTED,
    VerificationResult,
)
from services.ai_service import AIService  # noqa: E402
from services.search_service import SearchError, get_search_provider  # noqa: E402
from services.verification_service import VerificationService  # noqa: E402
from utils.helpers import escape_md, markdown_safe_url, truncate, validate_claim  # noqa: E402

st.set_page_config(page_title="News Verification App", page_icon="🔎", layout="centered")

EXAMPLES = {
    "Example: coffee": "Scientists have discovered that drinking coffee completely prevents heart disease.",
    "Example: 5G": "5G mobile networks spread the coronavirus.",
    "Example: moon": "NASA landed astronauts on the Moon in 1969.",
}


# ----------------------------------------------------------------- callbacks
def set_example(text: str) -> None:
    st.session_state["claim_input"] = text
    st.session_state["result"] = None


def reset() -> None:
    st.session_state["claim_input"] = ""
    st.session_state["result"] = None


# ----------------------------------------------------------------- rendering
def render_sources(result: VerificationResult) -> None:
    st.subheader("Sources")
    st.caption(
        "These are the pages the search returned. They are listed with better-known source "
        "types first, but that does not guarantee they are correct - please read them yourself."
    )
    for i, s in enumerate(result.sources, start=1):
        with st.container(border=True):
            st.markdown(f"**Source {i}: {escape_md(s.source_name)}**  \n*{escape_md(s.source_type)}*")
            st.markdown(f"**{escape_md(s.title)}**")
            st.caption(f"Date: {s.published_date or 'not available'}")
            st.write(escape_md(truncate(s.snippet, 300)))
            url = markdown_safe_url(s.url)
            st.markdown(f"[{escape_md(s.url)}]({url})")


def render_result(result: VerificationResult) -> None:
    st.divider()

    if result.error:
        st.error(result.error)
        if result.sources:
            st.info("Sources were found before the problem occurred, so you can still read them.")
            render_sources(result)
        return

    st.markdown("#### VERDICT")
    label = f"**{result.verdict}**"
    if result.verdict == VERDICT_SUPPORTED:
        st.success(label, icon="✅")
    elif result.verdict == VERDICT_DISPUTED:
        st.error(label, icon="⚠️")
    else:
        st.warning(label, icon="❓")
    if result.confidence is not None and result.sources:
        st.caption(
            f"AI self-reported confidence: {result.confidence:.0%}. "
            "This is the model's own estimate, not a measured probability."
        )

    st.subheader("Explanation")
    st.write(escape_md(result.summary))
    if result.reasoning:
        st.write(escape_md(result.reasoning))

    if result.evidence:
        st.subheader("Evidence")
        st.caption("What the retrieved sources say (the interpretation is in the explanation above).")
        for item in result.evidence:
            source = result.sources[item.source_index - 1]
            st.markdown(f"- {escape_md(item.text)}  \n  *Source {item.source_index}: {escape_md(source.source_name)}*")

    if result.sources:
        render_sources(result)

    st.info(
        "This result was produced by AI from the sources above. It is **not absolute truth** - "
        "open the original sources and judge for yourself."
    )
    st.button("Verify another claim", on_click=reset)


# ---------------------------------------------------------------------- main
def build_service() -> VerificationService:
    return VerificationService(search_provider=get_search_provider(), ai_service=AIService())


def main() -> None:
    st.session_state.setdefault("claim_input", "")
    st.session_state.setdefault("result", None)

    st.title("News Verification App")
    st.write("Check online claims using AI-assisted evidence retrieval.")

    if not os.getenv("SEARCH_API_KEY") or not os.getenv("AI_API_KEY"):
        st.warning(
            "API keys are not fully configured. Copy `.env.example` to `.env` and add "
            "`SEARCH_API_KEY` and `AI_API_KEY` (see README)."
        )

    st.text_area(
        "Enter a claim or headline:",
        key="claim_input",
        height=150,
        placeholder="Paste a headline, social-media claim, or short statement here...",
    )

    cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

    if st.button("Verify Claim", type="primary"):
        _, error = validate_claim(st.session_state["claim_input"])
        if error:
            st.session_state["result"] = None
            st.error(error)
        else:
            with st.status("Verifying...", expanded=True) as status_box:
                try:
                    service = build_service()
                    result = service.verify(
                        st.session_state["claim_input"], on_status=lambda msg: st.write(msg)
                    )
                except SearchError as exc:  # e.g. unknown SEARCH_PROVIDER
                    result = VerificationResult(claim=st.session_state["claim_input"], error=str(exc))
                if result.error:
                    status_box.update(label="Could not finish verification", state="error", expanded=False)
                else:
                    status_box.update(label="Verification complete", state="complete", expanded=False)
            st.session_state["result"] = result

    if st.session_state["result"] is not None:
        render_result(st.session_state["result"])


main()
