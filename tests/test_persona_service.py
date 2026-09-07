from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from cns.infrastructure.persona_repository import PersonaRevision
from cns.services import persona_service as persona_module
from cns.services.persona_service import (
    PERSONA_PREVIEW_TTL_SECONDS,
    PersonaParseError,
    PersonaService,
)


USER_ID = "11111111-1111-1111-1111-111111111111"


def revision(number: int = 1, directives: str = "") -> PersonaRevision:
    return PersonaRevision(
        id=uuid4(),
        user_id=UUID(USER_ID),
        revision_number=number,
        directives=directives,
        source="baseline" if number == 1 else "automatic",
        parent_revision_id=None,
        evidence_ids=[],
        created_at=datetime.now(UTC),
        audit_event_id=None,
    )


class FakeRepository:
    def __init__(self):
        self.current = revision(directives="Be concise.")
        self.evidence_id = uuid4()
        self.append_calls = []
        self.attempt_calls = []
        self.saved_signals = []
        self.evaluated = False

    def get_current_revision(self, user_id):
        return self.current

    def list_revisions(self, user_id):
        return [self.current]

    def get_revision(self, user_id, revision_id):
        return revision(2, "Older behavior.")

    def segment_was_evaluated(self, user_id, segment_id):
        return self.evaluated

    def save_signals(self, user_id, signals):
        self.saved_signals.extend(signals)
        self.evaluated = True
        return len(signals)

    def mark_segment_evaluated(self, user_id, segment_id):
        self.evaluated = True

    def refinement_due(self, user_id, activity_day, interval):
        return activity_day == 7 and interval == 7

    def get_unconsumed_signals(self, user_id):
        return [{
            "id": self.evidence_id,
            "behavioral_section": "Directness",
            "outcome": "misalignment",
            "strength": "strong",
            "evidence": "MIRA buried the answer.",
        }]

    def append_revision(self, user_id, directives, source, **kwargs):
        self.append_calls.append((user_id, directives, source, kwargs))
        self.current = PersonaRevision(
            id=uuid4(),
            user_id=UUID(user_id),
            revision_number=self.current.revision_number + 1,
            directives=directives,
            source=source,
            parent_revision_id=self.current.id,
            evidence_ids=kwargs.get("evidence_ids", []),
            created_at=datetime.now(UTC),
            audit_event_id=uuid4(),
        )
        return self.current

    def mark_refinement_attempt(self, *args, **kwargs):
        self.attempt_calls.append((args, kwargs))


class SequenceLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs)

    def extract_text_content(self, response):
        return response


class FakeValkey:
    def __init__(self):
        self.values = {}
        self.setex_calls = []
        self.hdel_calls = []

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.setex_calls.append((key, ttl, value))

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        self.values.pop(key, None)

    def hdel_with_retry(self, key, field):
        self.hdel_calls.append((key, field))


def test_automatic_revision_retries_critic_and_consumes_only_used_evidence(monkeypatch) -> None:
    repo = FakeRepository()
    llm = SequenceLLM([
        "<mira:persona>First candidate.</mira:persona>",
        '<mira:persona_review status="fail"><mira:issue>Too vague.</mira:issue></mira:persona_review>',
        "<mira:persona>Lead with the direct answer.</mira:persona>",
        '<mira:persona_review status="pass"/>',
    ])
    service = PersonaService(repository=repo, llm_provider=llm)
    monkeypatch.setattr(persona_module, "get_user_cumulative_activity_days", lambda user_id: 7)
    invalidations = []
    monkeypatch.setattr(service, "_invalidate_cache", lambda user_id: invalidations.append(user_id))

    published = service.refine_automatically_if_due(USER_ID)

    assert published is not None
    assert published.directives == "Lead with the direct answer."
    assert len(llm.calls) == 4
    assert {call["model_config"] for call in llm.calls} == {"primary"}
    _, _, source, kwargs = repo.append_calls[0]
    assert source == "automatic"
    assert kwargs["evidence_ids"] == [repo.evidence_id]
    assert kwargs["activity_day_checkpoint"] == 7
    assert invalidations == [USER_ID]


def test_manual_preview_accept_decline_and_rollback_are_immutable(monkeypatch) -> None:
    repo = FakeRepository()
    llm = SequenceLLM([
        "malformed",
        "<mira:persona>Use a warmer customer-facing tone.</mira:persona>",
        '<mira:persona_review status="pass"/>',
    ])
    valkey = FakeValkey()
    monkeypatch.setattr("clients.valkey_client.get_valkey_client", lambda: valkey)
    service = PersonaService(repository=repo, llm_provider=llm)

    preview = service.create_preview(USER_ID, "Mira is too blunt with customers")
    assert valkey.setex_calls[0][1] == PERSONA_PREVIEW_TTL_SECONDS
    accepted = service.accept_preview(USER_ID, preview["preview_id"])
    assert accepted.source == "user"
    assert accepted.directives == "Use a warmer customer-facing tone."
    assert valkey.values == {}

    decline_llm = SequenceLLM([
        "<mira:persona>Keep replies short.</mira:persona>",
        '<mira:persona_review status="pass"/>',
    ])
    decline_service = PersonaService(repository=repo, llm_provider=decline_llm)
    declined = decline_service.create_preview(USER_ID, "shorter")
    before = len(repo.append_calls)
    decline_service.decline_preview(USER_ID, declined["preview_id"])
    assert len(repo.append_calls) == before

    rolled_back = service.rollback(USER_ID, uuid4())
    assert rolled_back.source == "rollback"
    assert rolled_back.directives == "Older behavior."


def test_evaluation_parser_is_strict_and_user_knowledge_free() -> None:
    segment_id = uuid4()
    continuum_id = uuid4()
    signals = PersonaService._parse_evaluation(
        """
        <mira:persona_evaluation>
          <mira:signal section="Directness" outcome="alignment" strength="moderate">
            <evidence>MIRA answered the question before explaining.</evidence>
          </mira:signal>
        </mira:persona_evaluation>
        """,
        user_id=USER_ID,
        segment_id=segment_id,
        continuum_id=continuum_id,
    )
    assert len(signals) == 1
    assert signals[0].behavioral_section == "Directness"
    assert signals[0].evidence.startswith("MIRA answered")
    with pytest.raises(PersonaParseError, match="malformed signal XML"):
        PersonaService._parse_evaluation(
            '<mira:persona_evaluation><mira:signal outcome="alignment"/></mira:persona_evaluation>',
            user_id=USER_ID,
            segment_id=segment_id,
            continuum_id=continuum_id,
        )
