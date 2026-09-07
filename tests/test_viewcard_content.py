"""Typed viewcards and canonical generated-content policy."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tools.implementations.viewcard_content import normalize_content
from tools.implementations.viewcard_tool import ViewcardRequest, ViewcardTool
from utils.user_context import set_current_segment_id


def test_markdown_is_rendered_server_side_to_canonical_html() -> None:
    canonical, mode = normalize_content(
        "# Visit\n\n| Day | Job |\n| --- | --- |\n| Mon | Roof |\n\n```text\nsafe\n```",
        "markdown",
    )

    assert mode == "html"
    assert "<h1>Visit</h1>" in canonical
    assert "<table>" in canonical
    assert "<pre><code" in canonical


def test_valid_html_css_svg_links_and_raster_data_images_are_preserved() -> None:
    image = base64.b64encode(b"image bytes").decode("ascii")
    canonical, mode = normalize_content(
        "<style>.meter{fill:url(#paint)} @media (min-width:20px){.meter{opacity:.8}}</style>"
        "<a href='https://example.com/report'>Report</a>"
        f"<img alt='chart' src='data:image/png;base64,{image}'>"
        "<svg viewBox='0 0 20 20'><defs><linearGradient id='paint'>"
        "<stop offset='0' stop-color='#fff'></stop></linearGradient></defs>"
        "<rect class='meter' width='20' height='20' fill='url(#paint)'></rect></svg>",
        "html",
    )

    assert mode == "html"
    assert "https://example.com/report" in canonical
    assert "data:image/png;base64" in canonical
    assert "linearGradient" in canonical or "lineargradient" in canonical
    assert "url(#paint)" in canonical


@pytest.mark.parametrize(
    ("content", "diagnostic"),
    [
        ("<script>alert(1)</script>", "tag <script>"),
        ("<p onclick='bad()'>No</p>", "event attribute onclick"),
        ("<form><input></form>", "tag <form>"),
        ("<iframe src='https://example.com'></iframe>", "tag <iframe>"),
        ("<img src='https://example.com/a.png'>", "image URL"),
        ("<img src='data:image/svg+xml;base64,PHN2Zz4='>", "data-image MIME type"),
        ("<a href='javascript:alert(1)'>No</a>", "anchor href URL"),
        ("<style>@import 'https://example.com/a.css';</style>", "@import"),
        ("<style>@container card (width > 1px){p{color:red}}</style>", "unknown at-rule"),
        ("<style>:host{color:red}</style>", "selector"),
        ("<style>p{position:fixed}</style>", "position: fixed"),
        ("<style>p{background:url(https://example.com/a.png)}</style>", "external URL"),
        (
            "<style>@supports (background:url(https://example.com/a.png)){p{color:red}}</style>",
            "external URL",
        ),
        ("<svg><path href='https://example.com/icon.svg'></path></svg>", "external URL"),
        ("<marquee>No</marquee>", "unknown tag"),
    ],
)
def test_rejected_generated_content_names_the_exact_contract_violation(
    content: str,
    diagnostic: str,
) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        normalize_content(content, "html")


@pytest.mark.parametrize(
    ("card_type", "data"),
    [
        ("schedule", {"date": "July 14", "appointments": []}),
        ("customer", {"contact": {"name": "Bobby Dispatcher"}}),
        ("ticket", {"customer_name": "Bobby", "status": "scheduled"}),
        (
            "invoice",
            {
                "invoice_number": "INV-1",
                "customer_name": "Bobby",
                "amount": "$10.00",
                "total": "$10.00",
                "status": "open",
            },
        ),
        ("confirmation", {"message": "Appointment confirmed"}),
        ("notification", {"message": "Crew is en route", "level": "info"}),
        (
            "calendar",
            {
                "view": "seven_days",
                "anchor_date": "2026-07-15",
                "source": "manual",
                "activities": [{"date": "2026-07-17", "dots": 2}],
            },
        ),
    ],
)
def test_each_structured_card_has_a_strict_typed_payload(
    card_type: str,
    data: dict[str, object],
) -> None:
    request = ViewcardRequest.model_validate({
        "card_type": card_type,
        "title": "Card",
        "data": data,
    })

    assert request.card_type == card_type
    assert request.data is not None


def test_crm_calendar_cards_aggregate_upcoming_ticket_activity_in_user_timezone(monkeypatch) -> None:
    import tools.implementations.viewcard_tool as viewcard_tool

    class FakeCRMClient:
        def get_data(self, type_: str, **params: object) -> list[dict[str, object]]:
            assert type_ == "tickets"
            assert params == {"filter": "upcoming", "limit": 500}
            return [
                {"ticket": {"scheduled_at": "2026-07-16T01:00:00Z"}},
                {"ticket": {"scheduled_at": "2026-07-18T15:00:00Z"}},
                {"ticket": {"scheduled_at": "2026-07-23T15:00:00Z"}},
            ]

    monkeypatch.setattr(viewcard_tool, "client_for_workspace", lambda: FakeCRMClient())
    monkeypatch.setattr(
        viewcard_tool,
        "get_user_preferences",
        lambda: SimpleNamespace(timezone="America/Detroit"),
    )
    set_current_segment_id("1726fcbb-90f4-4873-a2ef-4291e09a90bb")

    result = ViewcardTool().run(
        card_type="calendar",
        title="Upcoming work",
        data={"view": "seven_days", "anchor_date": "2026-07-15", "source": "crm"},
    )

    assert result["card"]["data"] == {
        "view": "seven_days",
        "anchor_date": "2026-07-15",
        "source": "crm",
        "activities": [
            {"date": "2026-07-15", "dots": 1},
            {"date": "2026-07-18", "dots": 1},
        ],
    }


def test_crm_calendar_cards_reject_caller_supplied_activity_dots() -> None:
    with pytest.raises(ValidationError, match="derive activity dots"):
        ViewcardRequest.model_validate({
            "card_type": "calendar",
            "title": "Upcoming work",
            "data": {
                "view": "seven_days",
                "anchor_date": "2026-07-15",
                "source": "crm",
                "activities": [{"date": "2026-07-17", "dots": 1}],
            },
        })


def test_viewcard_generates_identity_and_canonical_content_for_current_segment() -> None:
    segment_id = "1726fcbb-90f4-4873-a2ef-4291e09a90bb"
    set_current_segment_id(segment_id)

    result = ViewcardTool().run(
        card_type="content",
        title="Status",
        content="**Complete**",
        content_mode="markdown",
        actions=[{"label": "Follow up", "prompt": "Ask for the exact completion time"}],
    )

    assert result["card"]["card_id"].startswith("vcd_")
    assert result["card"]["segment_id"] == segment_id
    assert result["card"]["data"] == {
        "content": "<p><strong>Complete</strong></p>",
        "content_mode": "html",
    }
    assert result["card"]["actions"][0]["prompt"] == "Ask for the exact completion time"
    assert ViewcardTool.is_call_parallel_safe({}) is False


def test_callers_cannot_supply_mutable_card_operations_or_identifiers() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ViewcardRequest.model_validate({
            "operation": "update",
            "card_id": "caller-owned",
            "card_type": "notification",
            "title": "Bad",
            "data": {"message": "No"},
        })
