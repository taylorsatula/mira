"""
System One decision-model client.

System One models (TypeSafe's Jev, the lunaroute gateway's djev, self-hosted
Kev) take one unstructured `state` plus a set of typed questions and return a
typed, calibrated answer per question in a single pass — no generated text.
This client speaks the shared `POST /v1/systemone` wire format; the endpoint,
model, and Vault key name come from `config.systemone`.

Only the Noul question type (a statement answered with a probability in
[0, 1]) is implemented — it is the only type any consumer asks. Choice and
Score land with their first consumer.

Every question goes out in its own request. The wire format allows several
questions per request, and TypeSafe documents them as answered in isolation,
but djev conditions each answer on the whole request — sibling questions,
their order, and their key names all moved one question's probability by up
to 0.84 (witnessed 2026-09-26). One question per request makes an answer
depend only on its own text and the state, whatever the server does.
"""

import contextvars
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping

from config import config
from clients.vault_client import get_api_key
from utils import http_client

logger = logging.getLogger(__name__)

# Retries per request on 429/5xx/connect errors (utils/http_client policy);
# http_client caps each retry sleep at 30 s.
_MAX_RETRIES = 2
_MAX_RETRY_SLEEP_SECONDS = 30


class SystemOneResponseError(RuntimeError):
    """The endpoint answered, but not with the typed answers that were asked for."""


@dataclass(frozen=True)
class NoulQuestion:
    """A statement the model judges true or false, answered as P(true)."""

    instructions: str

    def __post_init__(self) -> None:
        if not self.instructions.strip():
            raise ValueError("NoulQuestion.instructions must be non-empty")


class SystemOneClient:
    """Typed client for one System One endpoint.

    Holds construction-time wiring only (endpoint, model, bearer token, pooled
    HTTP client, request worker pool sized to the endpoint's in-flight limit),
    so one instance is safely shared across threads and users.
    """

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        api_key: str,
        timeout_seconds: int,
        max_concurrent_requests: int,
    ):
        if not endpoint_url or not model or not api_key:
            raise ValueError(
                "SystemOneClient requires endpoint_url, model, and api_key; got "
                f"endpoint_url={endpoint_url!r}, model={model!r}, api_key={'set' if api_key else 'missing'}"
            )
        if timeout_seconds <= 0 or max_concurrent_requests <= 0:
            raise ValueError(
                f"timeout_seconds and max_concurrent_requests must be positive, got "
                f"{timeout_seconds} and {max_concurrent_requests}"
            )
        self.endpoint_url = endpoint_url
        self.model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._http = http_client.Client(timeout=timeout_seconds, max_retries=_MAX_RETRIES)
        # The pool size is the in-flight cap: every request runs on a pool thread.
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent_requests, thread_name_prefix="systemone")
        # Longest one request can legitimately take: every attempt times out
        # and every retry sleeps the maximum.
        self._answer_wait_seconds = (
            (_MAX_RETRIES + 1) * timeout_seconds + _MAX_RETRIES * _MAX_RETRY_SLEEP_SECONDS
        )

    def ask_nouls(self, state: str, questions: Mapping[str, NoulQuestion]) -> dict[str, float]:
        """Answer every question against `state`, one isolated request each.

        Requests run concurrently. Returns {question name: P(statement true)}
        with exactly the asked names. Raises httpx errors on transport failure
        and on non-2xx status (after retries), SystemOneResponseError when a
        body is not the typed answer that was asked for, TimeoutError when an
        answer outlives the request bound.
        """
        if not state:
            raise ValueError("state must be non-empty")
        if not questions:
            raise ValueError("at least one question is required")
        futures = {
            name: self._pool.submit(contextvars.copy_context().run, self._ask_one, state, name, question)
            for name, question in questions.items()
        }
        return {name: future.result(timeout=self._answer_wait_seconds) for name, future in futures.items()}

    def _ask_one(self, state: str, name: str, question: NoulQuestion) -> float:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {name: {"type": "noul", "instructions": question.instructions}},
        }
        response = self._http.post(self.endpoint_url, json=payload, headers=self._headers)
        if response.status_code >= 400:
            raise http_client.HTTPStatusError(
                f"System One endpoint {self.endpoint_url} returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
                request=response.request,
                response=response,
            )

        try:
            body = response.json()
        except ValueError as e:
            raise SystemOneResponseError(
                f"System One endpoint returned non-JSON body: {response.text[:300]!r}"
            ) from e
        answers = body.get("answers") if isinstance(body, dict) else None
        if not isinstance(answers, dict) or set(answers) != {name}:
            raise SystemOneResponseError(
                f"System One answers do not match the question asked: asked {name!r}, "
                f"got {sorted(answers) if isinstance(answers, dict) else answers!r}"
            )
        answer = answers[name]
        value = answer.get("noul") if isinstance(answer, dict) else None
        if (
            not isinstance(answer, dict)
            or answer.get("type") != "noul"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise SystemOneResponseError(
                f"System One answer {name!r} is not a noul probability in [0, 1]: {answer!r}"
            )
        return float(value)


_client: SystemOneClient | None = None
_client_lock = threading.Lock()


def get_systemone_client() -> SystemOneClient:
    """Process-wide client built from `config.systemone` and its Vault key.

    Construction failures (missing Vault key, invalid config) propagate.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                settings = config.systemone
                _client = SystemOneClient(
                    endpoint_url=settings.endpoint_url,
                    model=settings.model,
                    api_key=get_api_key(settings.api_key_name),
                    timeout_seconds=settings.timeout,
                    max_concurrent_requests=settings.max_concurrent_requests,
                )
    return _client
