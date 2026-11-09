# deploy/lib/context_window_config.sh
# Context-window resolution for the install's env file
# Source this file - do not execute directly
#
# The context window is not a literal in this repo: it is whatever the chosen
# chat endpoint reports for the chosen chat model at GET <endpoint>/v1/models.
# Live compaction fires at CONTEXT_WINDOW_COMPACTION_FRACTION of that value, so
# a model swap on the same endpoint moves both numbers together.
#
# The endpoint URL is the /v1/chat/completions URL the install uses, so the
# models list is derived by replacing that suffix (the same shape
# deploy/lib/services.sh:prefill_provider_model uses). An endpoint that does
# not answer, does not list the model, or does not report a window has no
# resolvable value; the caller decides whether that is fatal.

# Live compaction fires at this fraction of the resolved window.
CONTEXT_WINDOW_COMPACTION_FRACTION_NUM=8
CONTEXT_WINDOW_COMPACTION_FRACTION_DEN=10

# Fallback when both sources report nothing (see resolve_context_window's
# return contract). This is a last-resort guess, not a measurement: it
# deliberately does NOT match ApiConfig's field defaults (256000/225000), which
# apply only to a non-deploy run. A deploy that lands here prints a warning and
# writes these, so an endpoint that reports no window at all still gets explicit
# values rather than inherited ones. The provider's own rejection is the backstop
# when the guess is larger than the served window: a llama-server overflow body
# classifies as ProviderContextOverflowError and the turn path remediates.
CONTEXT_WINDOW_FALLBACK_TOKENS=200000

# resolve_context_window PYTHON ENDPOINT MODEL API_KEY
#
# PYTHON must be a MIRA venv interpreter. ENDPOINT is the chat completions URL;
# MODEL the chat model identifier; API_KEY may be empty for an endpoint that
# takes none (passed on stdin, never argv).
# Sets CONTEXT_WINDOW_TOKENS and COMPACTION_TRIGGER_TOKENS and returns 0 when
# the endpoint reports the model's window. Prints the reason on stderr and
# returns 1 otherwise — no fallback is applied here.
#
# Two sources, in order. A hosted gateway reports the window on its models list.
# llama-server reports none there — every entry omits context_window,
# context_length, and max_input_tokens — and publishes the real value as n_ctx
# at GET /props instead, so without the second source every llama.cpp install
# fails this probe, takes the fallback, and can never compact before the
# provider rejects an oversized prompt.
resolve_context_window() {
    local python="$1" endpoint="$2" model="$3" api_key="$4"
    local models_url="${endpoint%/chat/completions}/models"
    # llama-server serves props from the host root, not under the /v1 prefix.
    local props_url="${endpoint%/chat/completions}/props"
    props_url="${props_url%/v1/props}/props"
    local probe='import json, sys, urllib.request
url, props_url, model = sys.argv[1], sys.argv[2], sys.argv[3]
key = sys.stdin.read().strip()

def fetch(target):
    request = urllib.request.Request(target)
    if key:
        request.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)

try:
    body = fetch(url)
except Exception as error:
    print("models list request failed: %s" % error, file=sys.stderr)
    sys.exit(1)
entries = body.get("data") if isinstance(body, dict) else None
if isinstance(entries, list):
    entry = next((m for m in entries if isinstance(m, dict) and m.get("id") == model), None)
    if entry is not None:
        window = entry.get("context_window") or entry.get("context_length") or entry.get("max_input_tokens")
        if window:
            print(int(window))
            sys.exit(0)

try:
    props = fetch(props_url)
except Exception as error:
    print("model %r reports no window and /props failed: %s" % (model, error), file=sys.stderr)
    sys.exit(1)
settings = props.get("default_generation_settings") if isinstance(props, dict) else None
window = settings.get("n_ctx") if isinstance(settings, dict) else None
if not window:
    print("model %r reports no window on the models list or at /props" % model, file=sys.stderr)
    sys.exit(1)
print(int(window))'
    local window

    case "$endpoint" in
        ""|*/chat/completions) ;;
        *)
            echo "context window probe needs a chat completions URL, got '$endpoint'" >&2
            return 1
            ;;
    esac
    [ -n "$endpoint" ] || { echo "context window probe needs a chat completions URL" >&2; return 1; }
    [ -n "$model" ] || { echo "context window probe needs a chat model" >&2; return 1; }

    if ! window=$(printf '%s' "$api_key" | "$python" -c "$probe" "$models_url" "$props_url" "$model"); then
        return 1
    fi
    case "$window" in
        ''|*[!0-9]*)
            echo "context window probe did not report a token count (got: '$window')" >&2
            return 1
            ;;
    esac

    CONTEXT_WINDOW_TOKENS="$window"
    COMPACTION_TRIGGER_TOKENS=$(( window * CONTEXT_WINDOW_COMPACTION_FRACTION_NUM / CONTEXT_WINDOW_COMPACTION_FRACTION_DEN ))
}

# resolve_primary_route_window PYTHON APP_DIR
#
# Resolve the window for the route the install actually serves chat from, at
# update time, where no interview supplied CONFIG_CHAT_*. Reads the primary
# route out of model_configs and its credential out of Vault through the app's
# own clients (the update runner already carries the service's AppRole), then
# probes the endpoint exactly as resolve_context_window does.
#
# An upgraded install keeps whatever systemone.env it already had: update.sh
# preserves that file rather than regenerating it, so a window block added by a
# later release never reaches an install that predates it, and the app falls
# back to ApiConfig's field defaults. Those are not tied to the served model, so
# live compaction can be configured above the provider's real limit and never
# fire. This is the call that closes that gap.
#
# Sets CONTEXT_WINDOW_TOKENS and COMPACTION_TRIGGER_TOKENS; prints the reason on
# stderr and returns 1 when the route or its window cannot be resolved.
resolve_primary_route_window() {
    local python="$1" app_dir="$2"
    local read_route='import sys
from clients.postgres_client import PostgresClient
from clients.vault_client import get_api_key
row = PostgresClient("mira_service").execute_single(
    "SELECT endpoint_url, model, api_key_name FROM model_configs WHERE name = %s",
    ("primary",),
)
if not row:
    print("model_configs has no primary route", file=sys.stderr)
    sys.exit(1)
key = get_api_key(row["api_key_name"]) if row["api_key_name"] else ""
print(row["endpoint_url"])
print(row["model"])
print(key)'
    local route

    if ! route=$(cd "$app_dir" && "$python" -c "$read_route"); then
        return 1
    fi
    local endpoint model api_key
    { IFS= read -r endpoint; IFS= read -r model; IFS= read -r api_key; } <<< "$route"

    resolve_context_window "$python" "$endpoint" "$model" "$api_key"
}
