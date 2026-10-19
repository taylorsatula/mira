/**
 * CORE.JS — the chat page's foundation: shared state, the DOM cache, the API
 * client with its turn-independent listeners, startup, and segment collapse.
 *
 * `AppState` and `elements` are top-level `const`s. Sibling <script> tags share
 * one global lexical scope, so later modules read them directly and they are
 * not properties of `window`; `elements` is captured once, at eval time, so an
 * id added to the page after this file runs is invisible to every module.
 *
 * `initializeAPIClient` installs the listeners that outlive a turn: `tool`
 * frames mapped through TOOL_INDICATOR_STATES to ui.js's badge,
 * `proactive_message` to messaging.js, the Retry Send indicator on connect and
 * disconnect, and the interrupted-reply restore scheduled after reconnect.
 * messaging.js adds one listener per turn and removes it in the same turn, so a
 * second copy of a turn listener in that array would handle every frame twice.
 *
 * `Functions.collapseSegment` starts the continuum action and the collapse
 * animation together and reports the outcome only once both have finished:
 * `collapsed: false` means the server deferred the collapse behind an in-flight
 * turn, and `result.message` is its explanation — rendering the success text
 * there would claim a collapse that did not happen.
 *
 * Load order: after api-client.js, before every other chat module.
 */

// ========================================
// APPLICATION STATE
// ========================================

const AppState = {
	responseActive: false,

	// Messages that failed to send, persisted across reloads.
	messageQueue: JSON.parse(localStorage.getItem('mira-queue') || '[]'),

	// API client instance (also window.miraAPI)
	apiClient: null,

	// Set when a reply was cut off by a dropped connection. The server stops
	// that turn and keeps the partial reply in history; the next connect
	// restores it from there.
	responseInterrupted: false,

	// Attached files for multimodal messages
	attachedFiles: []
};

// ========================================
// DOM ELEMENTS CACHE
// ========================================

const elements = {
	// Loading
	loadingScreen: document.getElementById('loading-screen'),
	asciiContainer: document.getElementById('ascii-container'),

	// Response - responseBox is the scroll container, responseContent the inner wrapper
	responseBox: document.getElementById('response_box'),
	responseContent: document.getElementById('response_content'),

	// Input
	inputContainer: document.querySelector('#compose_bar form'),
	messageInput: document.getElementById('chat_field'),
	sendButton: document.getElementById('send-button'),
	fileInput: document.getElementById('file-input'),
	attachmentPopover: document.getElementById('attachment-popover'),
	attachmentList: document.getElementById('attachment-list'),
	attachmentAddBtn: document.getElementById('attachment-add-btn'),
	preserveResolution: document.getElementById('preserve-resolution'),
	queueIndicator: document.getElementById('queue-indicator'),
	queueCount: document.querySelector('.retry-count'),
	queuePopover: document.getElementById('queue-popover'),
	queueMessages: document.getElementById('queue-messages'),

	// Toolbar indicators
	toolBadge: document.querySelector('[data-indicator="toolcall_btn"]'),
	attachmentButton: document.querySelector('[data-indicator="attachment_btn"]'),
	toolbarRight: document.querySelector('#toolbar .rightside'),

	// Inline history
	inlineHistoryContainer: document.getElementById('inline-history-container'),
	inlineHistorySentinel: document.getElementById('inline-history-sentinel'),
	newMessageToast: document.getElementById('new-message-toast'),
	toastScrollBtn: document.getElementById('toast-scroll-btn')
};

// ========================================
// API CLIENT INITIALIZATION
// ========================================

// Server `tool` frame lifecycle event → tool indicator state (ui.js).
const TOOL_INDICATOR_STATES = {
	tool_detected: 'detected',
	tool_executing: 'executing',
	tool_completed: 'completed',
	tool_error: 'error'
};

// Wait before restoring an interrupted reply: the server stops the turn at
// its next stream or tool boundary and only then commits the partial reply.
const INTERRUPTED_RESPONSE_RESTORE_DELAY_MS = 2000;
// Rows fetched to cover a full last turn (multi-round tool use produces many
// assistant and tool rows after one user row).
const LAST_TURN_HISTORY_LIMIT = 20;

async function restoreInterruptedResponse() {
	const { messages } = await AppState.apiClient.history.getHistory({ limit: LAST_TURN_HISTORY_LIMIT });
	// messages are oldest→newest; the last turn is everything after the newest user row.
	let lastUserIndex = -1;
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role === 'user') {
			lastUserIndex = i;
			break;
		}
	}
	const turnMessages = messages.slice(lastUserIndex + 1).filter(msg =>
		(msg.role === 'assistant' || msg.role === 'tool') && !msg.metadata.is_segment_boundary
	);
	if (turnMessages.some(msg => msg.role === 'assistant')) {
		window.showResponse(window.assembleTurnContent(turnMessages));
	}
}

function initializeAPIClient() {
	const apiClient = new window.MiraAPIClient();
	AppState.apiClient = apiClient;
	window.miraAPI = apiClient;

	// Single-user mode re-mints the session at '/'.
	apiClient.auth.onAuthChange((event) => {
		if (event.type === 'logout') {
			window.location.href = '/';
		}
	});

	apiClient.eventHandlers.onConnect.push(() => {
		if (AppState.messageQueue.length > 0) {
			window.updateQueueIndicator();
		}
		if (AppState.responseInterrupted) {
			AppState.responseInterrupted = false;
			setTimeout(() => {
				restoreInterruptedResponse().catch((error) => {
					console.error('Failed to restore the interrupted response; it remains in history:', error);
				});
			}, INTERRUPTED_RESPONSE_RESTORE_DELAY_MS);
		}
	});

	apiClient.eventHandlers.onDisconnect.push((event) => {
		console.warn('WebSocket disconnected:', event.code, event.reason);
		if (AppState.messageQueue.length > 0) {
			window.updateQueueIndicator();
		}
	});

	apiClient.eventHandlers.onError.push((error) => {
		console.error('WebSocket error:', error);
	});

	apiClient.eventHandlers.onMessage.push((frame) => {
		if (frame.type === 'tool') {
			window.updateToolIndicator(frame.tool_name, TOOL_INDICATOR_STATES[frame.event]);
		} else if (frame.type === 'proactive_message') {
			window.handleProactiveMessage(frame);
		}
	});
}

// ========================================
// APPLICATION INITIALIZATION
// ========================================

async function initializeMira() {
	initializeAPIClient();

	// false: the session is gone and the logout handler is already navigating.
	if (!await AppState.apiClient.auth.validateToken()) {
		return;
	}

	window.initDomainKnowledge(AppState.apiClient);
	window.updateQueueIndicator();
	await window.loadLastSessionResponse();
	window.setupEventListeners();
	window.setupDelegatedEvents();
}

if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', initializeMira);
} else {
	initializeMira();
}

// ========================================
// SEGMENT COLLAPSE
// ========================================

const COLLAPSE_SUCCESS_HTML = `The previous conversation segment has been collapsed as-per your request. Feel free to continue the conversation in a new direction or just come back later. MIRA will be ready when you return. <br><br><br><span style="opacity: 0.4; font-family: 'Minni', serif; letter-spacing: -1px">Dhu preeveeus kanverseishun segment haz bin kulapst eisper yor rikwest. Feel free too kuntinyoo dhu kanverseishun in u noo derekshun or just kum bak leiter. Miru wil bee redee wen yoo ritern.</span>`;

const Functions = {
	collapseSegment: function() {
		const responseBox = elements.responseBox;
		const responseContent = elements.responseContent;

		// The request runs while the animation plays; its outcome is shown
		// once both the fall and the request have finished.
		const outcome = AppState.apiClient.actions.executeAction('continuum', 'collapse_segment').then(
			// collapsed:false means the collapse was deferred behind the
			// in-flight turn; the server's message says so.
			(result) => (result.collapsed ? { kind: 'collapsed' } : { kind: 'deferred', message: result.message }),
			(error) => {
				console.error('Error collapsing segment:', error);
				return { kind: 'failed', message: error.message };
			}
		);

		window.collapseAnimation(responseContent, {
			scrollContainer: responseBox,
			onFallComplete: () => {
				setTimeout(async () => {
					const result = await outcome;
					responseBox.className = 'active';
					if (result.kind === 'failed') {
						responseContent.innerHTML = '<span style="color: #ff6b6b;"></span><br><br>Please try again.';
						responseContent.querySelector('span').textContent = `Failed to collapse segment: ${result.message}`;
					} else if (result.kind === 'deferred') {
						responseContent.textContent = result.message;
					} else {
						responseContent.innerHTML = COLLAPSE_SUCCESS_HTML;
					}
				}, 500);
			}
		});
	}
};

window.Functions = Functions;
