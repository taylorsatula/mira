# deploy/lib/context_window_config.sh
# Context-window resolution for the install's env file
# Source this file - do not execute directly
#
# The context window is not a literal in this repo: it is whatever the chosen
# chat endpoint reports for the chosen chat model at GET <endpoint>/v1/models.
# Live compaction fires at CONTEXT_WINDOW_COMPACTION_FRACTION of what that
# window leaves for input once the model's own output reserve is subtracted, so
# a model swap on the same endpoint moves both numbers together.
#
# The subtraction is not a safety margin: config/config.py:validate_compaction_budget
# refuses a compaction trigger above `context_window_tokens - primary max_tokens`
# (the model_configs row's output ceiling), and a value above it parks the app at
# its boot gate without ever binding the port. A prompt that fills the trigger
# still needs room for a whole completion, so the trigger is a fraction of the
# input budget rather than of the raw window.
#
# The endpoint URL is the /v1/chat/completions URL the install uses, so the
# models list is derived by replacing that suffix (the same shape
# deploy/lib/services.sh:prefill_provider_model uses). An endpoint that does
# not answer, does not list the model, or does not report a window has no
# resolvable value; the caller decides whether that is fatal.

# Live compaction fires at this fraction of the window's input budget.
CONTEXT_WINDOW_COMPACTION_FRACTION_NUM=8
CONTEXT_WINDOW_COMPACTION_FRACTION_DEN=10

# Fallback when the endpoint reports nothing (see resolve_context_window's
# return contract). Matches the app-side ApiConfig field defaults.
CONTEXT_WINDOW_FALLBACK_TOKENS=200000

# resolve_context_window PYTHON ENDPOINT MODEL API_KEY
#
# PYTHON must be a MIRA venv interpreter. ENDPOINT is the chat completions URL;
# MODEL the chat model identifier; API_KEY may be empty for an endpoint that
# takes none (passed on stdin, never argv).
# Sets CONTEXT_WINDOW_TOKENS, CONTEXT_OUTPUT_RESERVE_TOKENS and
# COMPACTION_TRIGGER_TOKENS and returns 0 when the endpoint reports the model's
# window. Prints the reason on stderr and returns 1 otherwise — no fallback is
# applied here.
#
# CONTEXT_OUTPUT_RESERVE_TOKENS is the model's own output ceiling, 0 when the
# entry reports none. Deriving the trigger from it keeps the pair valid for any
# `model_configs.max_tokens` at or below that ceiling; a row raised past the
# ceiling is caught by the app's validator at boot instead.
# TEMPORARY, retires with PR #72: the draft at taylorsatula/mira#72 rewrites the
# probe below to add llama-server's GET /props as a second window source and
# adds `resolve_primary_route_window` for an update-time backfill. That copy
# still derives the trigger as window * CONTEXT_WINDOW_COMPACTION_FRACTION;
# carry this file's input-budget arithmetic into both of its probe paths at the
# rebase and delete this note.
resolve_context_window() {
    local python="$1" endpoint="$2" model="$3" api_key="$4"
    unset CONTEXT_WINDOW_TOKENS CONTEXT_OUTPUT_RESERVE_TOKENS COMPACTION_TRIGGER_TOKENS
    local models_url="${endpoint%/chat/completions}/models"
    local probe='import json, sys, urllib.request
url, model = sys.argv[1], sys.argv[2]
key = sys.stdin.read().strip()
request = urllib.request.Request(url)
if key:
    request.add_header("Authorization", "Bearer " + key)
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        body = json.load(response)
except Exception as error:
    print("models list request failed: %s" % error, file=sys.stderr)
    sys.exit(1)
entries = body.get("data") if isinstance(body, dict) else None
if not isinstance(entries, list):
    print("models list has no data array", file=sys.stderr)
    sys.exit(1)
entry = next((m for m in entries if isinstance(m, dict) and m.get("id") == model), None)
if entry is None:
    print("model %r is not in the endpoint models list" % model, file=sys.stderr)
    sys.exit(1)
window = entry.get("context_window") or entry.get("context_length") or entry.get("max_input_tokens")
if not window:
    print("model %r reports no context window" % model, file=sys.stderr)
    sys.exit(1)
reserve = entry.get("max_output_tokens") or entry.get("max_completion_tokens") or 0
print(int(window))
print(int(reserve))'
    local window reserve resolved input_budget

    case "$endpoint" in
        ""|*/chat/completions) ;;
        *)
            echo "context window probe needs a chat completions URL, got '$endpoint'" >&2
            return 1
            ;;
    esac
    [ -n "$endpoint" ] || { echo "context window probe needs a chat completions URL" >&2; return 1; }
    [ -n "$model" ] || { echo "context window probe needs a chat model" >&2; return 1; }

    if ! resolved=$(printf '%s' "$api_key" | "$python" -c "$probe" "$models_url" "$model"); then
        return 1
    fi
    window=$(printf '%s\n' "$resolved" | sed -n '1p')
    reserve=$(printf '%s\n' "$resolved" | sed -n '2p')
    case "$window" in
        ''|*[!0-9]*)
            echo "context window probe did not report a token count (got: '$window')" >&2
            return 1
            ;;
    esac
    case "$reserve" in
        ''|*[!0-9]*)
            echo "context window probe did not report a numeric output reserve (got: '$reserve')" >&2
            return 1
            ;;
    esac

    input_budget=$(( window - reserve ))
    if [ "$input_budget" -lt 2 ]; then
        echo "model '$model' reports a $window-token window against a $reserve-token output reserve, leaving $input_budget input tokens; no compaction trigger fits" >&2
        return 1
    fi

    CONTEXT_WINDOW_TOKENS="$window"
    CONTEXT_OUTPUT_RESERVE_TOKENS="$reserve"
    COMPACTION_TRIGGER_TOKENS=$(( input_budget * CONTEXT_WINDOW_COMPACTION_FRACTION_NUM / CONTEXT_WINDOW_COMPACTION_FRACTION_DEN ))
}
