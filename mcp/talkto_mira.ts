/**
 * talkto_mira — pi extension exposing the talkto_mira tool: a model-callable
 * complete chat turn with a deployed MIRA instance over POST /v0/api/chat.
 *
 * This is agent-to-MIRA, not a human frontend: the model calls it to consult
 * or check in with the user's persistent AI companion, and every call leaves a
 * footprint in MIRA's permanent history. The server-side counterpart for
 * non-pi MCP clients is the /v0/mcp endpoint (cns/api/mcp.py, opt-in via
 * MIRA_MCP_ENABLED); pi uses this native tool instead and needs no flag.
 *
 * Credentials come from the MIRA TUI endpoint store
 * (~/.config/mira-tui/config.json, active endpoint's base_url + api_key —
 * minted by `python -m tui --login`). That store's format is owned by
 * tui/endpoints.py in the mira-OSS repo; any format change updates this file
 * in the same commit (twin contract).
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

// Server-side twin of _CHAT_READ_TIMEOUT_S in cns/api/mcp.py: no fixed client
// timeout can cover every legal turn (the server's per-user lock TTL derives
// from (MAX_LOCAL_TOOL_CALLS_PER_TURN + 1) * provider timeouts * 2), so 600 s
// spans every realistic turn and pathological ones get a truthful timeout error.
const READ_TIMEOUT_MS = 600_000;

const DESCRIPTION =
  "Send a message to MIRA, a persistent AI companion that maintains long-term " +
  "memories, a persona, and a conversation history for its user. This performs a " +
  "complete conversational turn: the message is appended to the permanent " +
  "conversation history, may be extracted into long-term memory, and the reply is " +
  "MIRA's own composed response with all of its internal tooling and memory already " +
  "applied. Only one turn runs at a time — a call made while another turn is in " +
  "progress fails with a busy error. A turn can take tens of seconds to minutes.\n\n" +
  "An optional image or document (not both) may accompany the message. image/document " +
  "are base64-encoded bytes with no 'data:' prefix; the matching image_type/document_type " +
  "MIME type is required alongside. Images: image/jpeg, image/png, image/gif, " +
  "image/webp (max 5 MB decoded). Documents: PDF, DOCX, XLSX, TXT, CSV, JSON " +
  "(max 32 MB decoded).";

interface EndpointConfig {
  base_url: string;
  api_key: string;
}

interface EndpointStore {
  active: string | null;
  endpoints: Record<string, EndpointConfig>;
}

function resolveEndpoint(): { baseUrl: string; apiKey: string } {
  const path = join(homedir(), ".config", "mira-tui", "config.json");
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(path, "utf-8"));
  } catch (error) {
    throw new Error(
      `Cannot read the MIRA TUI endpoint store at ${path} ` +
      `(${error instanceof Error ? error.message : String(error)}). ` +
      "Run `python -m tui --login` against the instance to mint a token."
    );
  }
  const store = raw as EndpointStore;
  const name = store.active;
  if (!name || !store.endpoints?.[name]) {
    throw new Error(
      `No active MIRA endpoint in ${path}. Run \`python -m tui --login\` and connect once, ` +
      "or point the store's active endpoint at the instance to use."
    );
  }
  const endpoint = store.endpoints[name];
  if (!endpoint.base_url || !endpoint.api_key) {
    throw new Error(
      `The active MIRA endpoint '${name}' in ${path} is missing base_url or api_key. ` +
      "Re-run `python -m tui --login`."
    );
  }
  return { baseUrl: endpoint.base_url, apiKey: endpoint.api_key };
}

interface ChatEnvelope {
  success: boolean;
  data?: { response?: string; continuum_id?: string; metadata?: unknown };
  error?: { code?: string; message?: string };
}

interface MiraTurnDetails {
  baseUrl: string;
  continuumId: string | null;
  metadata: unknown;
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "talkto_mira",
    label: "Talk to MIRA",
    description: DESCRIPTION,
    promptSnippet:
      "talkto_mira sends a complete conversational turn to MIRA, the user's persistent " +
      "AI companion (memories, persona, history) — consult it or check it in when the " +
      "task calls for it",
    promptGuidelines: [
      "Use talkto_mira when the task calls for consulting MIRA (the user's persistent " +
      "AI companion) or updating it on progress. A call performs a real turn: the " +
      "message lands in MIRA's permanent history and may be extracted into long-term " +
      "memory — say something worth remembering, not a throwaway probe.",
    ],
    parameters: Type.Object({
      message: Type.String({
        description:
          "The message for MIRA, exactly as it should enter its permanent conversation " +
          "history (it is not a throwaway query — compose it deliberately).",
      }),
      image: Type.Optional(
        Type.String({
          description: "Optional image as base64 bytes, no 'data:' prefix; requires image_type.",
        })
      ),
      image_type: Type.Optional(
        Type.String({ description: "MIME type of image, e.g. image/png (required alongside image)." })
      ),
      document: Type.Optional(
        Type.String({
          description:
            "Optional document as base64 bytes, no 'data:' prefix; requires document_type.",
        })
      ),
      document_type: Type.Optional(
        Type.String({
          description:
            "MIME type of document, e.g. application/pdf (required alongside document).",
        })
      ),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, _ctx) {
      const { baseUrl, apiKey } = resolveEndpoint();
      const url = `${baseUrl.replace(/\/$/, "")}/v0/api/chat`;
      const abort = AbortSignal.any([signal, AbortSignal.timeout(READ_TIMEOUT_MS)]);

      let response: Response;
      try {
        response = await fetch(url, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: params.message,
            image: params.image ?? null,
            image_type: params.image_type ?? null,
            document: params.document ?? null,
            document_type: params.document_type ?? null,
          }),
          signal: abort,
        });
      } catch (error) {
        if (signal.aborted) {
          throw new Error("talkto_mira: the turn was aborted; MIRA may still be processing it server-side.");
        }
        if (error instanceof DOMException && error.name === "TimeoutError") {
          throw new Error(
            `talkto_mira: MIRA did not finish the turn within ${READ_TIMEOUT_MS / 1000} seconds; ` +
            "the call was abandoned but the turn may still be running server-side and the message " +
            "was persisted. Retrying immediately will bounce with a busy error until the in-flight " +
            "turn completes."
          );
        }
        throw new Error(`talkto_mira: MIRA is unreachable at ${url} (${String(error)}).`);
      }

      let envelope: ChatEnvelope | null = null;
      try {
        envelope = (await response.json()) as ChatEnvelope;
      } catch {
        envelope = null;
      }

      if (response.ok && envelope?.success) {
        const reply = envelope.data?.response;
        if (typeof reply !== "string") {
          // Twin of cns/api/mcp.py's malformed-success row.
          throw new Error(
            `talkto_mira: MIRA returned a malformed success envelope (no data.response): ` +
            JSON.stringify(envelope).slice(0, 500)
          );
        }
        const details: MiraTurnDetails = {
          baseUrl,
          continuumId: envelope.data.continuum_id ?? null,
          metadata: envelope.data.metadata ?? null,
        };
        return {
          content: [{ type: "text", text: reply }],
          details,
        };
      }

      const errorMessage = envelope?.error?.message ?? "";
      const detail = errorMessage || `HTTP ${response.status}`;

      // Twin of cns/api/mcp.py's busy anchor: chat.py's ValidationError message.
      if (response.status === 400 && errorMessage.includes("already in progress")) {
        throw new Error(
          "talkto_mira: another turn is already in progress for this MIRA user " +
          "(one turn at a time); wait and retry shortly."
        );
      }
      if (response.status === 401 || response.status === 403) {
        throw new Error(
          `talkto_mira: MIRA rejected the token (HTTP ${response.status}); re-run ` +
          `\`python -m tui --login\`. ${detail}`
        );
      }
      throw new Error(`talkto_mira: MIRA chat endpoint returned HTTP ${response.status}: ${detail}`);
    },
  });
}
