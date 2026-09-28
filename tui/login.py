"""Headless token bootstrap for the TUI — ``python3 -m tui --login``.

Mints a long-lived API token against a deployed MIRA instance over the same
auth HTTP surface the web UI uses, and stores it in the 0600 endpoint store —
no browser needed. This module is a pure client: it imports nothing from the
server tree; it mirrors ``auth/api.py`` the way ``tui/protocol.py`` mirrors
frames. Drift rule: any change to those routes changes this module in the same
commit.

Bootstrap chain (server contracts, verified live):

1. ``GET /v0/auth/local/session`` — single-mode instances auto-provision the
   local account and answer 303 + ``Set-Cookie: session=<token>`` (zero
   credentials). Multi-mode answers 404 → fall to the magic-link flow:
   prompt email, ``POST /v0/auth/magic-link``, prompt the emailed token,
   ``POST /v0/auth/verify`` → ``data.session_token``.
2. ``POST /v0/auth/csrf`` (session as cookie AND Bearer header — the header
   takes precedence server-side) → ``data.csrf_token``.
3. ``POST /v0/auth/api-tokens`` (session + ``x-csrf-token`` header) → 201,
   ``data.token`` — the only place the raw token ever appears.
4. Write the token into the store; never print it (last resort: if the store
   cannot be written, print it with a warning rather than lose it silently).
"""

from __future__ import annotations

import httpx

from tui.endpoints import EndpointConfig, EndpointStore

_TIMEOUT = httpx.Timeout(15.0)
_TOKEN_NAME = "mira-tui"
_LOCAL_SESSION_PATH = "/v0/auth/local/session"
_MAGIC_LINK_PATH = "/v0/auth/magic-link"
_VERIFY_PATH = "/v0/auth/verify"
_CSRF_PATH = "/v0/auth/csrf"
_API_TOKENS_PATH = "/v0/auth/api-tokens"


class LoginError(Exception):
    """Bootstrap failed — carries the real server or transport error text."""


# --- pure request builders (server-shape mirrors; verified by dry probe) ---


def magic_link_payload(email: str) -> dict[str, str]:
    """Body for POST /v0/auth/magic-link (MagicLinkRequest)."""
    return {"email": email}


def verify_payload(token: str) -> dict[str, str]:
    """Body for POST /v0/auth/verify (MagicLinkVerifyRequest)."""
    return {"token": token}


def api_token_payload(name: str = _TOKEN_NAME, expires_in_days: int | None = None) -> dict[str, object]:
    """Body for POST /v0/auth/api-tokens (APITokenCreateRequest)."""
    return {"name": name, "expires_in_days": expires_in_days}


def bearer_headers(session_token: str) -> dict[str, str]:
    """Session transport: Bearer header (takes precedence server-side)."""
    return {"Authorization": f"Bearer {session_token}"}


def csrf_headers(csrf_token: str) -> dict[str, str]:
    return {"x-csrf-token": csrf_token}


# --- envelope handling -----------------------------------------------------


def _envelope(response: httpx.Response, step: str) -> dict:
    """Unwrap the standard {success, data|error} envelope; raise on failure
    or non-2xx with the real server message — never a generic one."""
    if response.status_code < 200 or response.status_code >= 300:
        raise LoginError(f"{step}: HTTP {response.status_code}: {response.text.strip()}")
    try:
        payload = response.json()
    except ValueError as error:
        raise LoginError(f"{step}: non-JSON response: {error}") from error
    if payload.get("error"):
        error = payload["error"]
        raise LoginError(f"{step}: [{error.get('code')}] {error.get('message')}")
    data = payload.get("data")
    if data is None:
        raise LoginError(f"{step}: response envelope carried no data: {payload}")
    return data


def _request(client: httpx.Client, method: str, url: str, step: str, **kwargs: object) -> httpx.Response:
    try:
        return client.request(method, url, **kwargs)
    except httpx.HTTPError as error:
        raise LoginError(f"{step}: {error}") from error


# --- session acquisition ----------------------------------------------------


def _local_session_token(client: httpx.Client, base_url: str) -> str | None:
    """Single-mode: 303 + Set-Cookie session=<token>. None on 404 (multi-mode)."""
    response = _request(
        client, "GET", base_url + _LOCAL_SESSION_PATH, "local/session probe",
        follow_redirects=False,
    )
    if response.status_code == 404:
        return None
    if response.status_code != 303:
        raise LoginError(
            f"local/session: expected 303 (single-mode) or 404 (multi-mode), "
            f"got HTTP {response.status_code}: {response.text.strip()}"
        )
    token = response.cookies.get("session")
    if not token:
        raise LoginError("local/session: 303 without a session cookie — server contract drift")
    return token


def _magic_link_session_token(client: httpx.Client, base_url: str) -> str:
    """Multi-mode: email magic-link → emailed token → session token."""
    print("multi-mode: passwordless magic-link flow — check your email", flush=True)
    email = input("email: ").strip()
    if not email:
        raise LoginError("magic-link: an email address is required")
    response = _request(
        client, "POST", base_url + _MAGIC_LINK_PATH, "magic-link",
        json=magic_link_payload(email),
    )
    _envelope(response, "magic-link")  # 200 regardless of registration; validate shape only

    session_token = _verify_emailed_token(client, base_url)
    if not session_token:
        raise LoginError("magic-link: verification failed")
    return session_token


def _verify_emailed_token(client: httpx.Client, base_url: str) -> str | None:
    """Prompt for the emailed token; one re-prompt on expired/invalid, then fail."""
    for attempt in (1, 2):
        token = input("paste the token from your magic-link email: ").strip()
        if not (32 <= len(token) <= 128):
            if attempt == 2:
                raise LoginError("verify: token length outside 32-128 chars")
            print("token must be 32-128 characters; try again", flush=True)
            continue
        response = _request(
            client, "POST", base_url + _VERIFY_PATH, "verify",
            json=verify_payload(token),
        )
        if response.status_code == 200:
            return _envelope(response, "verify").get("session_token")
        payload = response.json() if _is_json(response) else {}
        code = (payload.get("error") or {}).get("code", "")
        if code in ("expired_token", "invalid_token") and attempt == 1:
            print(f"token rejected ({code}); one more try", flush=True)
            continue
        raise LoginError(f"verify: HTTP {response.status_code}: {response.text.strip()}")
    return None


def _is_json(response: httpx.Response) -> bool:
    return "json" in response.headers.get("content-type", "")


# --- the mint chain ----------------------------------------------------------


def mint_api_token(base_url: str, token_name: str = _TOKEN_NAME) -> str:
    """Run the full bootstrap chain and return the raw API token.

    The caller owns secrecy: store it (0600) and never print it — the only
    sanctioned print is the can't-store last resort in ``run_login``.
    """
    base_url = base_url.rstrip("/")
    with httpx.Client(timeout=_TIMEOUT) as client:
        session_token = _local_session_token(client, base_url)
        if session_token is None:
            session_token = _magic_link_session_token(client, base_url)
        else:
            print("single-mode: auto-provisioning local session…", flush=True)

        headers = bearer_headers(session_token)
        cookies = {"session": session_token}
        response = _request(
            client, "POST", base_url + _CSRF_PATH, "csrf",
            headers=headers, cookies=cookies,
        )
        csrf_token = _envelope(response, "csrf")["csrf_token"]

        for attempt in range(1, 5):
            name = token_name if attempt == 1 else f"{token_name}-{attempt}"
            response = _request(
                client, "POST", base_url + _API_TOKENS_PATH, "api-tokens",
                headers={**headers, **csrf_headers(csrf_token)},
                cookies=cookies,
                json=api_token_payload(name),
            )
            if response.status_code == 201:
                data = _envelope(response, "api-tokens")
                token = data.get("token")
                if not token:
                    raise LoginError(f"api-tokens: 201 without data.token: {data}")
                return str(token)
            if response.status_code == 400 and "duplicate_token_name" in response.text and attempt < 4:
                continue  # server refuses duplicate token names: bump the suffix
            raise LoginError(
                f"api-tokens: expected 201, got HTTP {response.status_code}: {response.text.strip()}"
            )
        raise LoginError("api-tokens: unreachable")


# --- CLI entry ----------------------------------------------------------------


def run_login(
    store: EndpointStore,
    endpoint: str | None,
    base_url: str | None,
    save_as: str | None,
) -> int:
    """Pre-run bootstrap command: mint, store, print a summary, return rc.

    - ``--endpoint NAME``: mint against that stored endpoint's base_url and
      write the token into its ``api_key``.
    - ``--base-url URL [--save-as NAME]``: mint and save as a new endpoint
      (default name ``default``), set it active, print the next step.
    - bare: target the store's ACTIVE endpoint (fail loud if none).
    """
    if endpoint is not None and base_url is not None:
        print("fatal: --endpoint and --base-url are mutually exclusive", flush=True)
        return 1

    if base_url is not None:
        target_url = base_url
        name = save_as or "default"
    else:
        if endpoint is not None:
            name = endpoint
            try:
                existing = store.get(endpoint)
            except KeyError as error:
                # store.get names the configured endpoints in its message
                print(f"fatal: {error.args[0]}", flush=True)
                return 1
            target_url = existing.base_url
        else:
            active = store.active_config()
            if active is None:
                print(
                    "fatal: --login needs a target: no active endpoint in "
                    f"{store.path}. Use --endpoint NAME or --base-url URL.",
                    flush=True,
                )
                return 1
            name = store.active_name() or ""
            target_url = active.base_url
        if not target_url:
            print(f"fatal: endpoint {name!r} has no base_url configured", flush=True)
            return 1

    try:
        token = mint_api_token(target_url)
    except LoginError as error:
        print(f"fatal: {error}", flush=True)
        return 1
    except KeyboardInterrupt:
        print("aborted", flush=True)
        return 130
    except EOFError:
        print("fatal: aborted — input closed (EOF)", flush=True)
        return 1

    if endpoint is not None:
        prior = store.get(endpoint)
        new_config = EndpointConfig(
            base_url=target_url.rstrip("/"),
            api_key=token,
            history_fetch=prior.history_fetch,
            include_thinking=prior.include_thinking,
        )
    else:
        new_config = EndpointConfig(
            base_url=target_url.rstrip("/"),
            api_key=token,
            history_fetch="session_plus_summary",
            include_thinking=True,
        )
    try:
        store.upsert(name, new_config)
        if base_url is not None:
            store.set_active(name)
    except OSError as error:
        print(
            f"WARNING: could not write the token to {store.path}: {error}\n"
            "Store it manually, then add an endpoint with this api_key. "
            "The raw token (shown once, last resort):\n"
            f"{token}",
            flush=True,
        )
        return 1

    print(
        f"success: API token minted and stored (0600) as endpoint {name!r}\n"
        f"  base_url: {new_config.base_url}\n"
        f"  store:    {store.path}",
        flush=True,
    )
    if base_url is not None:
        print(f"endpoint {name!r} is now active. Start chatting: python3 -m tui", flush=True)
    return 0
