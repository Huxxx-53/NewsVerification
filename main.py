"""
main.py

Flet application entrypoint. Responsible for:
- Setting up routing between Home / Result / History views
- Running the (potentially slow) verification pipeline on a background
  thread so the UI doesn't freeze
- Wiring view callbacks to controllers/verification_controller.py

No search or AI logic lives here - see controllers/verification_controller.py.
"""

import threading

import flet as ft
from dotenv import load_dotenv

from controllers import verification_controller as controller
from views.home_view import build_home_view
from views.result_view import build_result_view
from views.history_view import build_history_view

load_dotenv()  # reads ANTHROPIC_API_KEY / ANTHROPIC_MODEL from .env


def main(page: ft.Page):
    page.title = "News Verification App"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.bgcolor = ft.colors.GREY_100
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO

    # Holds the record most recently viewed, so /result can be rebuilt
    # on refresh/back-navigation without re-running verification.
    state = {"current_record": None}

    def go_home(error_message: str = "", prefill_claim: str = ""):
        page.views.clear()
        page.views.append(
            build_home_view(
                page,
                claims_checked=controller.get_claims_checked_count(),
                on_verify=start_verification,
                on_view_history=go_history,
                error_message=error_message,
                prefill_claim=prefill_claim,
            )
        )
        page.update()

    def go_result(record: dict):
        state["current_record"] = record
        page.views.clear()
        page.views.append(build_result_view(page, record, on_back=lambda: go_home()))
        page.update()

    def go_history():
        page.views.clear()
        page.views.append(
            build_history_view(
                page,
                history=controller.get_history(),
                on_open_item=open_history_item,
                on_back=lambda: go_home(),
                on_clear_all=clear_all_history,
            )
        )
        page.update()

    def open_history_item(record_id: int):
        record = controller.get_result(record_id)
        if record:
            go_result(record)
        else:
            go_history()

    def clear_all_history():
        from models import history_model
        history_model.clear_history()
        go_history()

    def start_verification(claim_text: str):
        """
        Called from the Home view's Verify button. Runs the pipeline on a
        background thread so typing/scrolling elsewhere isn't blocked,
        then routes to /result on success or back to home with an error.
        """

        def worker():
            result = controller.run_verification(claim_text)
            if result["ok"]:
                go_result(result["record"])
            else:
                # Keep the user's claim text so they don't have to retype it.
                go_home(error_message=result["message"], prefill_claim=claim_text)

        threading.Thread(target=worker, daemon=True).start()

    # Initial route.
    go_home()


if __name__ == "__main__":
    ft.app(target=main, view=ft.AppView.WEB_BROWSER)
