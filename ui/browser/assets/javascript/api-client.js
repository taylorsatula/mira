/**
 * API-CLIENT.JS - MIRA network layer (WebSocket turn protocol + HTTP envelope)
 *
 * The one place the browser talks to the server. Every other module goes
 * through a MiraAPIClient instance; raw fetch/WebSocket use elsewhere is a
 * defect.
 *
 * Server authority:
 * - WebSocket frames: cns/api/websocket_chat.py. The strict client twin is
 *   ui/tui/protocol.py + ui/tui/client.py; correlation and lifecycle here
 *   mirror ui/tui/chat.py. A frame-shape change on the server updates this
 *   file and the TUI twin together.
 * - HTTP envelope {success, data|error, meta}: cns/api/base.py, auth/api.py
 *   (main.py flattens auth HTTPException details so the code is at error.code).
 *
 * Auth is the HttpOnly `session` cookie (single-user auto-login). Cookie
 * writes to /v1/api* and /v1/auth* carry X-CSRF-Token.
 *
 * Load order: before core.js, which constructs the chat page's instance.
 */

(() => {
    'use strict';

    const CSRF_PATH = '/v1/auth/csrf';
    const CSRF_RETRY_CODES = new Set(['CSRF_INVALID', 'CSRF_REQUIRED']);

    // Server rejects content outside 1..100_000 characters (MessageFrame) with an
    // unattributed MALFORMED_FRAME, so the bound is enforced before sending.
    const MAX_CONTENT_LENGTH = 100000;
    const IMAGE_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp']);

    // TURN_BUSY backoff — same constants as ui/tui/chat.py (TURN_BUSY_RETRY_DELAYS,
    // TURN_BUSY_POLL_SECONDS, TURN_BUSY_MAX_WAIT_SECONDS). The wait is measured
    // from the first bounce.
    const TURN_BUSY_RETRY_DELAYS_MS = [500, 1000, 2000];
    const TURN_BUSY_POLL_MS = 3000;
    const TURN_BUSY_MAX_WAIT_MS = 600000;

    // Reconnect backoff — same shape as ui/tui/chat.py AUTO_RECONNECT_*: quick
    // attempts, then a steady poll that covers a server restart. An open tab keeps
    // polling until the server is back.
    const RECONNECT_DELAYS_MS = [2000, 4000, 8000];
    const RECONNECT_POLL_MS = 15000;
    // A send that has not reached the server waits this long for a connection.
    const SEND_CONNECT_MAX_WAIT_MS = 600000;
    // The server answers a missing auth frame with AUTH_TIMEOUT at 10 s; cover it.
    const AUTH_WAIT_MS = 12000;
    // A ping left unanswered for a full interval marks the socket dead.
    const KEEPALIVE_INTERVAL_MS = 30000;

    const AUTH_FAILURE_CODES = new Set(['AUTH_FAILED', 'AUTH_REQUIRED', 'AUTH_TIMEOUT']);
    // protocol_error codes the server sends without a message_id while a send is
    // unanswered (process_turn setup failures) — mirrors ui/tui/chat.py.
    const UNATTRIBUTED_SEND_CODES = new Set(['INVALID_MESSAGE', 'TURN_SETUP_FAILED']);
    // Server frames forwarded unchanged to eventHandlers.onMessage. Terminal
    // frames settle the sendMessage promise instead.
    const TURN_STREAM_FRAMES = new Set(['turn_started', 'assistant_delta', 'thinking', 'tool', 'context_reset', 'model_error', 'provider_notice']);

    class MiraHttpError extends Error {
        constructor(status, code, message) {
            super(message);
            this.name = 'MiraHttpError';
            this.status = status;
            this.code = code;
        }
    }

    class MiraTurnError extends Error {
        constructor(code, message) {
            super(message);
            this.name = 'MiraTurnError';
            this.code = code;
        }
    }

    // crypto.randomUUID exists only in secure contexts; a page served over plain
    // http from a non-localhost address is not one. getRandomValues works in every
    // context, so the v4 UUID is assembled from it.
    function newMessageId() {
        const bytes = crypto.getRandomValues(new Uint8Array(16));
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
        return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
    }

    function errorFromBody(status, body, text, endpoint) {
        if (body && body.error && typeof body.error.code === 'string') {
            return new MiraHttpError(status, body.error.code, body.error.message || body.error.code);
        }
        if (body && typeof body.detail === 'string') {
            return new MiraHttpError(status, `HTTP_${status}`, body.detail);
        }
        if (body && Array.isArray(body.detail)) {
            const message = body.detail
                .map((item) => `${(item.loc || []).join('.')}: ${item.msg}`)
                .join('; ');
            return new MiraHttpError(status, 'REQUEST_VALIDATION_ERROR', message);
        }
        if (status >= 200 && status < 300) {
            return new MiraHttpError(status, 'BAD_ENVELOPE', `${endpoint} returned a body without the success envelope`);
        }
        return new MiraHttpError(status, `HTTP_${status}`, text.trim().slice(0, 500) || `${endpoint} failed with HTTP ${status}`);
    }

    class MiraAPIClient {
        constructor() {
            this.baseURL = window.location.origin;
            this.wsURL = this.baseURL.replace(/^http/, 'ws') + '/v1/ws/chat';

            this._authenticated = false;
            this._loggedOut = false;
            this._csrfToken = null;
            this._csrfFetch = null;

            this._ws = null;
            this._connectionState = 'disconnected'; // disconnected | connecting | authenticated
            this._reconnectAttempts = 0;
            this._reconnectTimer = null;
            this._authTimer = null;
            this._keepaliveTimer = null;
            this._awaitingPong = false;

            // The single in-flight sendMessage; null when idle.
            this._send = null;

            this.eventHandlers = {
                onConnect: [],
                onDisconnect: [],
                onMessage: [],
                onError: [],
                onAuthChange: []
            };

            this.auth = {
                isAuthenticated: () => this._authenticated,

                validateToken: async () => {
                    try {
                        await this._httpRequest('/v1/auth/session');
                    } catch (error) {
                        // 401 already ran clearToken (logout event) inside _httpRequest.
                        if (error instanceof MiraHttpError && error.status === 401) {
                            return false;
                        }
                        throw error;
                    }
                    this._markAuthenticated();
                    return true;
                },

                clearToken: () => {
                    if (this._loggedOut) return;
                    this._loggedOut = true;
                    this._authenticated = false;
                    this._csrfToken = null;
                    clearTimeout(this._reconnectTimer);
                    this._reconnectTimer = null;
                    if (this._send) {
                        this._rejectSend('NOT_AUTHENTICATED', 'The session ended before this message was answered');
                    }
                    const ws = this._ws;
                    if (ws) {
                        this._dropSocket();
                        ws.close();
                    }
                    this._emit('onAuthChange', { type: 'logout' });
                },

                onAuthChange: (handler) => {
                    this.eventHandlers.onAuthChange.push(handler);
                }
            };

            this.chat = {
                sendMessage: (text, attachment = null) => this._sendMessage(text, attachment),
                halt: () => this._halt()
            };

            this.history = {
                getHistory: async ({ before, limit } = {}) => {
                    const data = await this.data.getData('history', { before, limit });
                    return {
                        messages: data.messages,
                        hasMore: data.meta.has_more,
                        nextBefore: data.meta.next_before ?? null
                    };
                }
            };

            this.actions = {
                executeAction: (domain, action, data = {}) => this._httpRequest('/v1/api/actions', {
                    method: 'POST',
                    body: JSON.stringify({ domain, action, data })
                })
            };

            this.data = {
                getData: (dataType, params = {}) => {
                    const query = new URLSearchParams({ type: dataType });
                    for (const [key, value] of Object.entries(params)) {
                        if (value !== undefined && value !== null) {
                            query.append(key, String(value));
                        }
                    }
                    return this._httpRequest(`/v1/api/data?${query.toString()}`);
                }
            };

            this.connect();
        }

        // ========================================
        // WEBSOCKET CONNECTION
        // ========================================

        connect() {
            if (this._loggedOut || this._ws) return;
            clearTimeout(this._reconnectTimer);
            this._reconnectTimer = null;
            this._connectionState = 'connecting';

            const ws = new WebSocket(this.wsURL);
            this._ws = ws;
            ws.onopen = () => {
                if (this._ws !== ws) return;
                // Cookie auth: no token, the server reads the session cookie.
                ws.send(JSON.stringify({ type: 'auth' }));
                this._authTimer = setTimeout(() => {
                    if (this._ws !== ws) return;
                    console.error('[MiraAPIClient] No authentication answer within', AUTH_WAIT_MS, 'ms; reconnecting');
                    ws.close();
                }, AUTH_WAIT_MS);
            };
            ws.onmessage = (event) => {
                if (this._ws === ws) this._handleFrame(event.data);
            };
            ws.onerror = (event) => {
                if (this._ws === ws) this._emit('onError', event);
            };
            ws.onclose = (event) => {
                if (this._ws === ws) this._handleClose(event);
            };
        }

        // Detach the current socket and reset connection state. Callers that also
        // need the socket closed close it themselves.
        _dropSocket() {
            clearTimeout(this._authTimer);
            this._authTimer = null;
            clearInterval(this._keepaliveTimer);
            this._keepaliveTimer = null;
            this._awaitingPong = false;
            this._ws = null;
            this._connectionState = 'disconnected';
        }

        _handleClose(event) {
            this._dropSocket();
            this._emit('onDisconnect', event);

            const send = this._send;
            if (send) {
                if (send.turnId) {
                    // The server stops a turn whose socket drops (reason
                    // "disconnect") and keeps the partial reply in history.
                    this._resolveSend({
                        status: 'stopped',
                        turnId: send.turnId,
                        reason: 'disconnect',
                        streamedText: send.streamedText
                    });
                } else if (send.transmitted) {
                    this._rejectSend(
                        'CONNECTION_LOST',
                        'Connection lost before MIRA confirmed this message — check history before resending'
                    );
                } else {
                    // Never reached the server: it goes out after the reconnect.
                    this._startConnectWait(send);
                }
            }

            if (!this._loggedOut) this._scheduleReconnect();
        }

        _scheduleReconnect() {
            const n = this._reconnectAttempts;
            const delay = n < RECONNECT_DELAYS_MS.length ? RECONNECT_DELAYS_MS[n] : RECONNECT_POLL_MS;
            this._reconnectAttempts += 1;
            this._reconnectTimer = setTimeout(() => {
                this._reconnectTimer = null;
                this.connect();
            }, delay);
        }

        _startKeepalive() {
            clearInterval(this._keepaliveTimer);
            this._awaitingPong = false;
            const ws = this._ws;
            this._keepaliveTimer = setInterval(() => {
                if (this._ws !== ws) return;
                if (this._awaitingPong) {
                    console.error('[MiraAPIClient] Ping unanswered for', KEEPALIVE_INTERVAL_MS, 'ms; reconnecting');
                    ws.close();
                    return;
                }
                this._awaitingPong = true;
                ws.send(JSON.stringify({ type: 'ping' }));
            }, KEEPALIVE_INTERVAL_MS);
        }

        _markAuthenticated() {
            if (this._authenticated) return;
            this._authenticated = true;
            this._emit('onAuthChange', { type: 'login' });
        }

        // ========================================
        // INBOUND FRAMES
        // ========================================

        _handleFrame(raw) {
            let frame;
            try {
                frame = JSON.parse(raw);
            } catch (error) {
                console.error('[MiraAPIClient] Unparseable WebSocket frame:', raw, error);
                return;
            }

            switch (frame.type) {
                case 'auth_success':
                    this._onAuthSuccess();
                    return;
                case 'pong':
                    this._awaitingPong = false;
                    return;
                case 'server_shutdown':
                    // The server closes the socket next; the close path reconnects.
                    console.warn('[MiraAPIClient] Server shutting down:', frame.code, frame.message);
                    return;
                case 'protocol_error':
                    this._onProtocolError(frame);
                    return;
                case 'proactive_message':
                    this._emit('onMessage', frame);
                    return;
                case 'turn_complete':
                case 'turn_stopped':
                case 'turn_error':
                    this._onTurnEnd(frame);
                    return;
                default:
                    if (TURN_STREAM_FRAMES.has(frame.type)) {
                        this._onTurnStream(frame);
                        return;
                    }
                    console.error('[MiraAPIClient] Unknown server frame type:', frame.type, frame);
            }
        }

        _onAuthSuccess() {
            clearTimeout(this._authTimer);
            this._authTimer = null;
            this._connectionState = 'authenticated';
            this._reconnectAttempts = 0;
            this._startKeepalive();
            this._markAuthenticated();
            this._emit('onConnect');
            this._transmit();
        }

        _onProtocolError(frame) {
            const { code, message } = frame;
            if (AUTH_FAILURE_CODES.has(code)) {
                console.error('[MiraAPIClient] WebSocket authentication failed:', code, message);
                this.auth.clearToken();
                return;
            }
            if (code === 'AUTH_UNAVAILABLE') {
                // Auth infrastructure is down, not the session: the server closes
                // the socket and the close path reconnects with backoff.
                console.warn('[MiraAPIClient] Authentication temporarily unavailable; reconnecting');
                return;
            }

            const send = this._send;
            const attributed = send !== null && send.turnId === null && (
                (frame.message_id && frame.message_id === send.frame.message_id)
                || (UNATTRIBUTED_SEND_CODES.has(code) && send.transmitted)
            );
            if (attributed && code === 'TURN_BUSY') {
                this._onTurnBusy(send, message);
                return;
            }
            if (attributed) {
                this._rejectSend(code, message);
                return;
            }
            if (code === 'NO_MATCHING_ACTIVE_TURN') {
                console.warn('[MiraAPIClient] Halt arrived after the turn had already ended');
                return;
            }
            console.error('[MiraAPIClient] Unattributed protocol_error:', code, message);
        }

        _onTurnStream(frame) {
            const send = this._send;
            if (frame.type === 'turn_started') {
                if (!send || send.turnId !== null || send.frame.message_id !== frame.message_id) {
                    console.error('[MiraAPIClient] turn_started for a message that is not in flight:', frame);
                    return;
                }
                send.turnId = frame.turn_id;
                send.segmentId = frame.segment_id;
                if (send.haltRequested) this._sendHalt(send);
            } else {
                if (!send || send.turnId !== frame.turn_id) {
                    console.error(`[MiraAPIClient] ${frame.type} for a turn that is not active:`, frame);
                    return;
                }
                if (frame.type === 'assistant_delta') {
                    send.streamedText += frame.content;
                } else if (frame.type === 'context_reset') {
                    send.streamedText = '';
                }
            }
            this._emit('onMessage', frame);
        }

        _onTurnEnd(frame) {
            const send = this._send;
            if (!send || send.turnId !== frame.turn_id) {
                console.error(`[MiraAPIClient] ${frame.type} for a turn that is not active:`, frame);
                return;
            }
            if (frame.type === 'turn_complete') {
                this._resolveSend({
                    status: 'complete',
                    turnId: frame.turn_id,
                    segmentId: frame.segment_id,
                    continuumId: frame.continuum_id,
                    response: frame.response,
                    streamedText: send.streamedText,
                    toolsUsed: frame.tools_used,
                    processingTimeMs: frame.processing_time_ms,
                    emotion: frame.emotion ?? null
                });
            } else if (frame.type === 'turn_stopped') {
                this._resolveSend({
                    status: 'stopped',
                    turnId: frame.turn_id,
                    reason: frame.reason,
                    streamedText: send.streamedText
                });
            } else {
                this._rejectSend(frame.code, frame.message);
            }
        }

        // ========================================
        // SENDING
        // ========================================

        _sendMessage(text, attachment) {
            if (this._loggedOut) {
                return Promise.reject(new MiraTurnError('NOT_AUTHENTICATED', 'The session has ended'));
            }
            if (this._send) {
                return Promise.reject(new MiraTurnError('SEND_IN_FLIGHT', 'A message is already in flight; wait for it to finish or halt it'));
            }
            if (typeof text !== 'string' || text.length === 0) {
                return Promise.reject(new MiraTurnError('EMPTY_MESSAGE', 'Message content is empty'));
            }
            if (text.length > MAX_CONTENT_LENGTH) {
                return Promise.reject(new MiraTurnError(
                    'MESSAGE_TOO_LONG',
                    `Message is ${text.length} characters; the maximum is ${MAX_CONTENT_LENGTH}`
                ));
            }

            const frame = {
                type: 'message',
                message_id: newMessageId(),
                content: text,
                include_thinking: true
            };
            if (attachment) {
                if (IMAGE_MIME_TYPES.has(attachment.mimeType)) {
                    frame.image = attachment.base64;
                    frame.image_type = attachment.mimeType;
                } else {
                    frame.document = attachment.base64;
                    frame.document_type = attachment.mimeType;
                }
            }

            return new Promise((resolve, reject) => {
                const send = {
                    frame,
                    resolve,
                    reject,
                    turnId: null,
                    segmentId: null,
                    streamedText: '',
                    transmitted: false,
                    haltRequested: false,
                    retries: 0,
                    firstBounce: null,
                    retryTimer: null,
                    connectTimer: null
                };
                this._send = send;
                if (this._connectionState === 'authenticated') {
                    this._transmit();
                } else {
                    this._startConnectWait(send);
                    this.connect();
                }
            });
        }

        // Put the in-flight message frame on the wire if the socket is
        // authenticated and the send is not already out or waiting in a TURN_BUSY
        // backoff. Called on send, on auth_success, and when a backoff ends.
        _transmit() {
            const send = this._send;
            if (!send || send.transmitted || send.retryTimer || send.turnId) return;
            if (this._connectionState !== 'authenticated') return;
            this._ws.send(JSON.stringify(send.frame));
            send.transmitted = true;
            clearTimeout(send.connectTimer);
            send.connectTimer = null;
        }

        _startConnectWait(send) {
            if (send.connectTimer) return;
            send.connectTimer = setTimeout(() => {
                send.connectTimer = null;
                if (this._send === send && !send.transmitted) {
                    this._rejectSend(
                        'CONNECTION_LOST',
                        `Could not reach MIRA within ${SEND_CONNECT_MAX_WAIT_MS / 1000} s; the message was not sent`
                    );
                }
            }, SEND_CONNECT_MAX_WAIT_MS);
        }

        // Another turn holds this user's lock (usually a heartbeat turn the server
        // could not preempt). The bounced frame never reached the orchestrator, so
        // the same frame — same message_id — is resent after the backoff.
        _onTurnBusy(send, message) {
            if (send.haltRequested) {
                this._resolveSend({ status: 'stopped', turnId: null, reason: 'halt', streamedText: '' });
                return;
            }
            const now = Date.now();
            if (send.firstBounce === null) send.firstBounce = now;
            const waited = now - send.firstBounce;
            if (waited >= TURN_BUSY_MAX_WAIT_MS) {
                this._rejectSend('TURN_BUSY', `${message} (waited ${Math.round(waited / 1000)} s)`);
                return;
            }
            const n = send.retries;
            const delay = n < TURN_BUSY_RETRY_DELAYS_MS.length ? TURN_BUSY_RETRY_DELAYS_MS[n] : TURN_BUSY_POLL_MS;
            send.retries += 1;
            send.transmitted = false;
            send.retryTimer = setTimeout(() => {
                send.retryTimer = null;
                if (this._send !== send) return;
                if (this._connectionState === 'authenticated') {
                    this._transmit();
                } else {
                    this._startConnectWait(send);
                }
            }, delay);
        }

        _halt() {
            const send = this._send;
            if (!send || send.haltRequested) return;
            if (send.turnId) {
                this._sendHalt(send);
            } else if (send.transmitted) {
                // The server has the frame but has not announced the turn yet:
                // the halt goes out the moment turn_started arrives.
                send.haltRequested = true;
            } else {
                // Waiting in a TURN_BUSY backoff or for a connection: the server
                // holds nothing to halt.
                this._resolveSend({ status: 'stopped', turnId: null, reason: 'halt', streamedText: '' });
            }
        }

        _sendHalt(send) {
            send.haltRequested = true;
            if (this._connectionState !== 'authenticated') return; // the drop stops the turn server-side
            this._ws.send(JSON.stringify({
                type: 'halt',
                turn_id: send.turnId,
                message_id: send.frame.message_id
            }));
        }

        _finishSend() {
            const send = this._send;
            clearTimeout(send.retryTimer);
            clearTimeout(send.connectTimer);
            this._send = null;
            return send;
        }

        _resolveSend(result) {
            this._finishSend().resolve(result);
        }

        _rejectSend(code, message) {
            this._finishSend().reject(new MiraTurnError(code, message));
        }

        // ========================================
        // HTTP
        // ========================================

        async _ensureCsrfToken() {
            if (this._csrfToken) return this._csrfToken;
            // One fetch per tab at a time: every fetch rotates the server-side
            // token, so concurrent fetches would invalidate each other.
            if (!this._csrfFetch) {
                this._csrfFetch = this._httpRequest(CSRF_PATH, { method: 'POST' })
                    .then((data) => {
                        if (!data || typeof data.csrf_token !== 'string' || !data.csrf_token) {
                            throw new MiraHttpError(200, 'BAD_CSRF_RESPONSE', `${CSRF_PATH} returned no csrf_token`);
                        }
                        this._csrfToken = data.csrf_token;
                        return data.csrf_token;
                    })
                    .finally(() => {
                        this._csrfFetch = null;
                    });
            }
            return this._csrfFetch;
        }

        async _fetchEnvelope(endpoint, method, options, csrfToken) {
            const headers = { ...options.headers };
            if (options.body !== undefined) headers['Content-Type'] = 'application/json';
            if (csrfToken) headers['X-CSRF-Token'] = csrfToken;

            let response;
            try {
                response = await fetch(`${this.baseURL}${endpoint}`, {
                    method,
                    headers,
                    body: options.body,
                    credentials: 'include'
                });
            } catch (error) {
                throw new MiraHttpError(0, 'NETWORK_ERROR', `${method} ${endpoint} failed: ${error.message}`);
            }

            const text = await response.text();
            let body = null;
            try {
                body = JSON.parse(text);
            } catch (error) {
                body = null; // non-JSON body: errorFromBody reports the raw text
            }
            if (response.ok && body && body.success === true) {
                return { status: response.status, data: body.data, error: null };
            }
            return { status: response.status, data: undefined, error: errorFromBody(response.status, body, text, endpoint) };
        }

        /**
         * The one HTTP path. `options.body` is a JSON string. Resolves to the
         * envelope's `data`; rejects with MiraHttpError.
         */
        async _httpRequest(endpoint, options = {}) {
            const method = (options.method || 'GET').toUpperCase();
            const path = endpoint.split('?')[0];
            const csrfProtected = method !== 'GET' && method !== 'HEAD' && (
                path.startsWith('/v1/api') || (path.startsWith('/v1/auth') && path !== CSRF_PATH)
            );

            let csrfToken = csrfProtected ? await this._ensureCsrfToken() : null;
            let result = await this._fetchEnvelope(endpoint, method, options, csrfToken);

            if (csrfProtected && result.status === 403 && CSRF_RETRY_CODES.has(result.error.code)) {
                // The server keeps one CSRF token per session and every
                // POST /v1/auth/csrf replaces it, so another tab's fetch
                // invalidates this tab's copy. Refetch and retry exactly once.
                // Server-side alternative: make auth/session.py:generate_csrf_token
                // get-or-create (atomic SET NX) so concurrent tabs share one token
                // and this retry becomes unnecessary.
                if (this._csrfToken === csrfToken) this._csrfToken = null;
                csrfToken = await this._ensureCsrfToken();
                result = await this._fetchEnvelope(endpoint, method, options, csrfToken);
            }

            if (result.error) {
                if (result.status === 401) this.auth.clearToken();
                throw result.error;
            }
            return result.data;
        }

        _emit(event, data) {
            for (const handler of this.eventHandlers[event]) {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`[MiraAPIClient] ${event} handler threw:`, error);
                }
            }
        }
    }

    window.MiraAPIClient = MiraAPIClient;
    window.MiraHttpError = MiraHttpError;
    window.MiraTurnError = MiraTurnError;
})();
