/**
 * MESSAGING.JS - Message Sending, Reply Streaming & Rendering
 *
 * PURPOSE:
 * Owns the chat turn on the client: the retry queue for unsent messages,
 * attachments, sending through `miraAPI.chat.sendMessage`, consuming the raw
 * turn frames of the in-flight send, the streaming display filter that hides
 * `<mira:...>` tags and think blocks, markdown rendering, session start, notices,
 * and proactive messages.
 *
 * TURN FRAMES (raw server frames on `eventHandlers.onMessage`):
 * - turn_started      fixes the turn_id this send's frames are matched against
 * - assistant_delta   raw model text, fed through ReplyFilter per entry_id
 * - thinking          reasoning text, shown live then folded into a drawer
 * - tool              ends the provider step (flush) and records the tool call;
 *                     the toolbar indicator is driven by core.js
 * - context_reset     discards everything streamed so far for the turn
 * - model_error       informational notice, removed when the turn settles
 * The send promise settles once: complete | stopped (halt|disconnect|stall) |
 * MiraTurnError.
 *
 * DISPLAYED EXCHANGE:
 * The live region holds the turn in flight and the exchange it produced, which
 * is the newest thing in the thread. When the next send starts, beginTurn hands
 * it to the inline history and empties the region in the same step, so the
 * conversation never blinks between turns. history.js reads
 * `streamingState.displayedExchange` only to keep a retried first page from
 * rendering that exchange a second time.
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements
 * - ui.js: ASSET_VERSION, animations, loading screen, status badges, attachment/queue UI
 * - history.js: window.InlineHistory
 * - repulsion-feedback.js: window.repulsionFeedback
 * - inactivity-warning.js: window.inactivityWarningManager
 * - domain-knowledge.js: window.domainManager
 * - External: marked, DOMPurify
 *
 * FILTER LITERALS: THINK_OPEN and THINK_CLOSE, declared below the header, hold
 * the raw tag bytes, and every hold-back test below compares candidates against
 * them character by character. A copy whose angle brackets arrived
 * HTML-escaped matches nothing, and the failure is silent: the model's
 * reasoning renders as reply text and tags show where text should have been
 * withheld.
 *
 * LOAD ORDER:
 * After core.js and ui.js, before history.js.
 */

// ========================================
// MESSAGE QUEUE MANAGEMENT
// ========================================

function persistQueue() {
	localStorage.setItem('mira-queue', JSON.stringify(AppState.messageQueue));
}

function queueMessage(text) {
	AppState.messageQueue.push({
		text: text,
		timestamp: Date.now()
	});
	persistQueue();
	updateQueueIndicator();
}

function escapeHtml(text) {
	const div = document.createElement('div');
	div.textContent = text;
	return div.innerHTML;
}

function renderQueuedMessages() {
	if (AppState.messageQueue.length === 0) {
		elements.queueMessages.innerHTML = '<div class="empty-state">No queued messages</div>';
		return;
	}

	let html = '';
	AppState.messageQueue.forEach((msg, index) => {
		const time = new Date(msg.timestamp).toLocaleString();
		html += `
			<div class="queue-message" data-index="${index}">
				<div class="queue-message-text">${escapeHtml(msg.text)}</div>
				<div class="queue-message-time">${time}</div>
				<div class="queue-message-actions">
					<button class="queue-edit-btn" data-index="${index}">Edit</button>
					<button class="queue-send-btn" data-index="${index}">Send</button>
					<button class="queue-remove-btn" data-index="${index}">Remove</button>
				</div>
			</div>
		`;
	});

	elements.queueMessages.innerHTML = html;
}

async function sendSingleQueuedMessage(index) {
	const msg = AppState.messageQueue[index];
	if (!msg || streamingState.isGenerating) return;

	elements.messageInput.value = msg.text;
	AppState.messageQueue.splice(index, 1);
	persistQueue();
	updateQueueIndicator();
	renderQueuedMessages();

	await sendMessage();
}

function removeQueuedMessage(index) {
	AppState.messageQueue.splice(index, 1);
	persistQueue();
	updateQueueIndicator();
	renderQueuedMessages();
}

function editQueuedMessage(index) {
	const msg = AppState.messageQueue[index];
	if (!msg) return;

	const messageEl = elements.queueMessages.querySelector(`[data-index="${index}"]`);
	if (!messageEl) return;

	const textEl = messageEl.querySelector('.queue-message-text');
	const actionsEl = messageEl.querySelector('.queue-message-actions');

	textEl.classList.add('editing');
	textEl.innerHTML = `<textarea class="queue-edit-textarea" rows="3">${escapeHtml(msg.text)}</textarea>`;
	actionsEl.innerHTML = `
		<button class="queue-save-btn" data-index="${index}">Save</button>
		<button class="queue-cancel-btn" data-index="${index}">Cancel</button>
	`;

	const textarea = textEl.querySelector('.queue-edit-textarea');
	textarea.focus();
	textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function saveQueuedMessageEdit(index) {
	const messageEl = elements.queueMessages.querySelector(`[data-index="${index}"]`);
	if (!messageEl) return;

	const textarea = messageEl.querySelector('.queue-edit-textarea');
	if (!textarea) return;

	const newText = textarea.value.trim();
	if (!newText) {
		alert('Message cannot be empty');
		return;
	}

	AppState.messageQueue[index].text = newText;
	persistQueue();
	renderQueuedMessages();
}

function cancelQueuedMessageEdit() {
	renderQueuedMessages();
}

// ========================================
// FILE ATTACHMENT HANDLING
// ========================================

async function downscaleImage(file, maxLongestAxis = 512, minShortestAxis = 64) {
	let bitmap = await createImageBitmap(file);
	const { width, height } = bitmap;

	// Already within bounds — return original (no re-encoding)
	if (Math.max(width, height) <= maxLongestAxis) {
		bitmap.close();
		return file;
	}

	// Scale longest axis to target
	let scale = maxLongestAxis / Math.max(width, height);

	// Narrow content protection: if shortest axis would collapse below floor,
	// scale to the floor instead (longest axis will exceed max)
	if (Math.min(width, height) * scale < minShortestAxis) {
		scale = minShortestAxis / Math.min(width, height);
	}

	const targetW = Math.round(width * scale);
	const targetH = Math.round(height * scale);

	// Multi-step halving — each step is ≤2:1, optimal for any resampling algorithm
	let curW = width;
	let curH = height;
	while (curW / 2 >= targetW && curH / 2 >= targetH) {
		curW = Math.round(curW / 2);
		curH = Math.round(curH / 2);
		const stepped = await createImageBitmap(bitmap, { resizeWidth: curW, resizeHeight: curH, resizeQuality: 'high' });
		bitmap.close();
		bitmap = stepped;
	}

	// Final resize to exact target
	const final = await createImageBitmap(bitmap, { resizeWidth: targetW, resizeHeight: targetH, resizeQuality: 'high' });
	bitmap.close();

	const canvas = new OffscreenCanvas(targetW, targetH);
	canvas.getContext('2d').drawImage(final, 0, 0);
	final.close();

	const mimeType = file.type || 'image/jpeg';
	const blob = await canvas.convertToBlob({ type: mimeType, quality: 0.9 });
	return new File([blob], file.name, { type: mimeType });
}

async function handleFileSelect(event) {
	const file = event.target.files[0];
	if (!file) return;

	const supportedTypes = [
		'image/jpeg', 'image/png', 'image/gif', 'image/webp',
		'application/pdf',
		'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
		'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
		'text/plain',
		'text/csv'
	];

	if (!supportedTypes.includes(file.type) && !file.type.startsWith('image/')) {
		alert('Unsupported file type. Supported: Images, PDF, DOCX, XLSX, TXT, CSV');
		return;
	}

	const isImage = file.type.startsWith('image/');
	const isText = file.type === 'text/plain' || file.type === 'text/csv';

	// Size limits: text 100KB, non-image 32MB. Images handled below per toggle.
	if (!isImage) {
		const maxSize = isText ? 100 * 1024 : 32 * 1024 * 1024;
		const maxSizeLabel = isText ? '100KB' : '32MB';
		if (file.size > maxSize) {
			alert(`File must be less than ${maxSizeLabel}`);
			return;
		}
	}

	try {
		let processedFile;

		if (isImage) {
			if (elements.preserveResolution.checked) {
				// Preserve resolution: send original, backend compresses.
				// 5MB limit applies since we're not downscaling client-side.
				if (file.size > 5 * 1024 * 1024) {
					alert('File must be less than 5MB when preserving resolution');
					return;
				}
				processedFile = file;
			} else {
				// Default: downscale to 512px longest axis.
				// No size check — output is always well under 5MB.
				processedFile = await downscaleImage(file);
			}
		} else {
			processedFile = file;
		}

		AppState.attachedFiles.push(processedFile);

		updateAttachmentPopover();
		elements.attachmentButton.classList.add('has-attachment');
		updateAttachmentCount();

		event.target.value = '';
	} catch (error) {
		console.error('Failed to process file:', error);
		alert('Failed to process file. Please try another file.');
		event.target.value = '';
	}
}

function clearAttachment(index) {
	if (index !== undefined) {
		AppState.attachedFiles.splice(index, 1);
	} else {
		AppState.attachedFiles = [];
	}

	elements.fileInput.value = '';

	if (AppState.attachedFiles.length === 0) {
		elements.attachmentButton.classList.remove('has-attachment');
	}

	updateAttachmentPopover();
	updateAttachmentCount();
}

function readFileAsBase64(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = (e) => resolve(e.target.result.split(',')[1]);
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

// ========================================
// REPLY DISPLAY FILTER
// ========================================

/*
 * Raw assistant text carries think blocks, `mira:` namespaced tags (paired or
 * self-closing, any name) and an ephemeral leading `[5:47pm]` timestamp, and a
 * tag can be split across deltas or span lines. ReplyFilter mirrors the TUI's
 * ReplyStream (ui/tui/text.py):
 * - Text is committed line by line and never retracted. Text that could still
 *   become (part of) a tag is held back, never shown and corrected.
 * - finish()/flush() drop what streaming held back: an unclosed tag and
 *   everything after it in that entry, and a trailing partial tag fragment.
 * - Blank lines are held until a later non-blank line proves them interior
 *   (no leading or trailing blanks, runs collapse to one).
 * - A new entry_id (a new provider step) flushes the current entry and owes a
 *   paragraph break before the next text.
 * Output is display text: committed lines, each ending in "\n", with a blank
 * line ("\n") as the paragraph break. A reply held back entirely leaves
 * hasText false — the cue to fall back to turn_complete.response.
 */

const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';
const MIRA_OPEN_PREFIX = '<mira:';
const MIRA_CLOSE_PREFIX = '</mira:';

// Paired (name backreference, case-insensitive) or self-closing; one pass over
// both kinds so spans found while streaming equal the spans the final filter removes.
const TAG_BLOCK_RE = /<think>[\s\S]*?<\/think>|<mira:([^>\/\s]+)(?:\s[^>]*)?>[\s\S]*?<\/mira:\1>|<mira:[^>]*\/>/gi;
// An opening tag that is not self-closing: unclosed when no block span starts at it.
const TAG_OPEN_RE = /<think>|<mira:[^>\/\s]+(?:\s[^>]*)?>/gi;
const TIMESTAMP_RE = /^\[\d{1,2}:\d{2}[ap]m\]\s*/i;

const LEAD_UNDECIDED = 0;  // no display text seen yet in this entry
const LEAD_STRIPPING = 1;  // timestamp stripped; eating the whitespace after it
const LEAD_DONE = 2;

function removeTagBlocks(text) {
	return text.replace(TAG_BLOCK_RE, '');
}

function isCovered(spans, pos) {
	return spans.some(([start, end]) => start <= pos && pos < end);
}

// True when `tail` (starts with "<", contains no ">") is, or could grow into, a think or mira tag.
function isTagFragment(tail) {
	const lower = tail.toLowerCase();
	return THINK_OPEN.startsWith(lower)
		|| THINK_CLOSE.startsWith(lower)
		|| MIRA_OPEN_PREFIX.startsWith(lower)
		|| MIRA_CLOSE_PREFIX.startsWith(lower)
		|| lower.startsWith(MIRA_OPEN_PREFIX)
		|| lower.startsWith(MIRA_CLOSE_PREFIX);
}

// Index of the last "\n" in text[0, end), or -1.
function lastNewlineBefore(text, end) {
	return end > 0 ? text.lastIndexOf('\n', end - 1) : -1;
}

// The complete tag-block spans of `raw` and `hold`, the index where text stops being safe to show.
function scanTags(raw) {
	const spans = Array.from(raw.matchAll(TAG_BLOCK_RE), (m) => [m.index, m.index + m[0].length]);
	let hold = raw.length;
	for (const match of raw.matchAll(TAG_OPEN_RE)) {
		if (!isCovered(spans, match.index)) {
			hold = match.index;
			break;
		}
	}
	// A trailing fragment can only start after the last ">" (it has none).
	let pos = raw.indexOf('<', raw.lastIndexOf('>') + 1);
	while (pos !== -1 && pos < hold) {
		if (isTagFragment(raw.slice(pos))) {
			hold = pos;
			break;
		}
		pos = raw.indexOf('<', pos + 1);
	}
	return { spans, hold };
}

class ReplyFilter {
	constructor() {
		this.raw = '';            // uncommitted text of the current entry
		this.entryId = null;
		this.lead = LEAD_UNDECIDED;
		this.hasText = false;     // a non-blank line has been committed
		this.blankPending = false;
	}

	// Append `delta` to entry `entryId`; return the display text this newly committed.
	feed(entryId, delta) {
		let out = '';
		if (this.entryId !== null && entryId !== this.entryId) {
			out += this.flush();
		}
		this.entryId = entryId;
		this.raw += delta.replace(/\r\n/g, '\n');
		const { spans, hold } = scanTags(this.raw);
		let cut = lastNewlineBefore(this.raw, hold);
		while (cut !== -1 && isCovered(spans, cut)) {
			cut = lastNewlineBefore(this.raw, cut);
		}
		if (cut !== -1) {
			const chunk = this.raw.slice(0, cut + 1);
			this.raw = this.raw.slice(cut + 1);
			out += this.commit(this.stripLead(removeTagBlocks(chunk)));
		}
		return out;
	}

	// End the current provider step: commit up to the hold-back point, drop
	// what was held, and owe a paragraph break before the next text.
	flush() {
		const { hold } = scanTags(this.raw);
		const out = this.commit(this.stripLead(removeTagBlocks(this.raw.slice(0, hold))));
		this.raw = '';
		this.lead = LEAD_UNDECIDED;
		this.blankPending = this.hasText;
		return out;
	}

	finish() {
		return this.flush();
	}

	stripLead(text) {
		if (this.lead === LEAD_UNDECIDED && text) {
			const match = text.match(TIMESTAMP_RE);
			if (match) {
				text = text.slice(match[0].length);
				this.lead = LEAD_STRIPPING;
			} else {
				this.lead = LEAD_DONE;
			}
		}
		if (this.lead === LEAD_STRIPPING) {
			text = text.trimStart();
			if (text) this.lead = LEAD_DONE;
		}
		return text;
	}

	commit(text) {
		const lines = text.split('\n');
		if (lines[lines.length - 1] === '') lines.pop();
		let out = '';
		for (const rawLine of lines) {
			const line = rawLine.trimEnd();
			if (!line) {
				this.blankPending = this.hasText;
				continue;
			}
			if (this.blankPending) {
				out += '\n';
				this.blankPending = false;
			}
			out += line + '\n';
			this.hasText = true;
		}
		return out;
	}
}

// One-shot display text for a complete raw text (same path as streaming).
function filterReplyText(text) {
	const filter = new ReplyFilter();
	return (filter.feed('', text) + filter.finish()).trimEnd();
}

// ========================================
// RENDERING HELPERS
// ========================================

marked.setOptions({ breaks: true, gfm: true });

const MARKDOWN_SANITIZE_CONFIG = {
	ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'u', 's', 'code', 'pre',
				   'blockquote', 'ul', 'ol', 'li', 'a', 'h1', 'h2',
				   'h3', 'h4', 'h5', 'h6', 'hr', 'table', 'thead',
				   'tbody', 'tr', 'th', 'td', 'img', 'span', 'div'],
	ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id', 'src', 'alt',
				   'title', 'data-language', 'data-copy'],
	ALLOW_DATA_ATTR: true
};

function renderMarkdown(text) {
	return DOMPurify.sanitize(marked.parse(text), MARKDOWN_SANITIZE_CONFIG);
}

// Displayable text of one history row's content: a plain string, or a
// JSON-stringified array of provider content blocks (only `text` blocks are
// conversation; `reasoning` and `tool_call` blocks are not).
// A row's content is either plain text or a JSON array of content blocks. An
// assistant row that called a tool is stored as blocks — reasoning, text, tool
// calls — which is the only place a past turn's thinking survives.
function contentBlocksOf(content) {
	if (Array.isArray(content)) return content;
	if (typeof content !== 'string') return [];
	const trimmed = content.trim();
	if (!trimmed.startsWith('[')) return [{ type: 'text', text: content }];
	try {
		const parsed = JSON.parse(trimmed);
		return Array.isArray(parsed) ? parsed : [{ type: 'text', text: content }];
	} catch {
		return [{ type: 'text', text: content }];
	}
}

function messageDisplayText(content) {
	return contentBlocksOf(content)
		.filter(block => block && block.type === 'text' && typeof block.text === 'string' && block.text)
		.map(block => block.text)
		.join('\n');
}

function isSegmentSentinel(row) {
	const value = row.metadata.is_segment_boundary;
	return value === true || value === 'true';
}

// Reply parts for assistant/tool history rows, in order:
// {kind: 'text', text} (display text) | {kind: 'tool', name, ok}.
function replyPartsFromRows(rows) {
	const parts = [];
	for (const row of rows) {
		if (row.role === 'tool') {
			// The tool row carries the whole record: name and arguments in
			// metadata, the result in content, failure in is_error.
			const name = row.metadata.tool_name;
			if (typeof name === 'string') {
				parts.push({
					kind: 'tool',
					name,
					input: row.metadata.tool_arguments ?? null,
					output: typeof row.content === 'string' ? row.content : '',
					ok: !row.is_error
				});
			}
			continue;
		}
		if (row.role !== 'assistant') continue;
		for (const block of contentBlocksOf(row.content)) {
			if (block.type === 'reasoning' && typeof block.text === 'string' && block.text.trim()) {
				parts.push({ kind: 'thinking', text: block.text });
			} else if (block.type === 'text' && typeof block.text === 'string') {
				const text = filterReplyText(block.text);
				if (text) parts.push({ kind: 'text', text });
			}
		}
	}
	return parts;
}

// Pretty-print a tool payload for the expandable body: arguments arrive as an
// object live and as a JSON string in history, results are always strings.
function toolPayloadText(value) {
	if (value === null || value === undefined || value === '') return '';
	if (typeof value === 'string') {
		const trimmed = value.trim();
		if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
			try {
				return JSON.stringify(JSON.parse(trimmed), null, 2);
			} catch {
				return value;
			}
		}
		return value;
	}
	try {
		return JSON.stringify(value, null, 2);
	} catch {
		return String(value);
	}
}

function renderThinkingDrawer(text) {
	return '<div class="thinking-drawer">'
		+ '<button type="button" class="thinking-drawer-toggle" aria-expanded="false">'
		+ '<span class="thinking-drawer-arrow" aria-hidden="true">&#x25B8;</span> Thinking</button>'
		+ `<div class="thinking-drawer-body">${escapeHtml(text)}</div>`
		+ '</div>';
}

function renderToolCall(part) {
	const input = toolPayloadText(part.input);
	const output = toolPayloadText(part.output);
	const status = part.ok ? 'ok' : 'failed';
	return `<div class="tool-call" data-status="${part.ok ? 'ok' : 'error'}">`
		+ '<button type="button" class="tool-call-header" aria-expanded="false">'
		+ '<span class="tool-call-arrow" aria-hidden="true">&#x25B8;</span>'
		+ `<span class="tool-call-name">${escapeHtml(part.name)}</span>`
		+ `<span class="tool-call-status">${status}</span>`
		+ '</button>'
		+ '<div class="tool-call-body">'
		+ (input ? `<div class="tool-call-section"><span class="tool-call-label">input</span><pre class="tool-call-pre">${escapeHtml(input)}</pre></div>` : '')
		+ (output ? `<div class="tool-call-section"><span class="tool-call-label">output</span><pre class="tool-call-pre">${escapeHtml(output)}</pre></div>` : '')
		+ '</div></div>';
}

function renderReplyParts(parts) {
	return parts.map(part => {
		if (part.kind === 'tool') return renderToolCall(part);
		if (part.kind === 'thinking') return renderThinkingDrawer(part.text);
		return renderMarkdown(part.text);
	}).join('');
}

// Index of the last user row, or -1.
function lastUserIndex(rows) {
	for (let i = rows.length - 1; i >= 0; i--) {
		if (rows[i].role === 'user') return i;
	}
	return -1;
}

// ========================================
// NOTICES
// ========================================

function showNotice(text, kind) {
	const notice = document.createElement('div');
	notice.className = `system-alert ${kind}`;

	const content = document.createElement('div');
	content.className = 'system-alert-content';
	const paragraph = document.createElement('p');
	paragraph.textContent = text;
	content.appendChild(paragraph);

	const dismiss = document.createElement('button');
	dismiss.type = 'button';
	dismiss.className = 'system-alert-dismiss';
	dismiss.setAttribute('aria-label', 'Dismiss');
	dismiss.textContent = 'D I S M I S S';
	dismiss.addEventListener('click', () => removeNotice(notice));

	notice.append(content, dismiss);
	document.getElementById('notifications-center').appendChild(notice);
	return notice;
}

function removeNotice(notice) {
	notice.classList.add('dismissing');
	notice.addEventListener('animationend', () => notice.remove(), { once: true });
}

// ========================================
// MESSAGE SENDING
// ========================================

async function sendMessage() {
	const message = elements.messageInput.value.trim();
	if (!message || streamingState.isGenerating) return;

	let attachment = null;
	if (AppState.attachedFiles.length > 0) {
		const file = AppState.attachedFiles[0];
		try {
			attachment = { base64: await readFileAsBase64(file), mimeType: file.type };
		} catch (error) {
			console.error('Failed to read attachment:', error);
			showNotice(`Could not read the attachment: ${error.message}`, 'error');
			return;
		}
		clearAttachment();
	}

	window.repulsionFeedback.noteUserMessage(message);

	// Hides the session status bar optimistically, refreshes from server
	window.inactivityWarningManager.trackActivity('message_sent');

	hapticFeedback(100);

	// Don't clear textarea here — animateTextDissolve fades it out then clears
	if (document.activeElement === elements.messageInput) {
		elements.messageInput.blur();
	}

	elements.sendButton.disabled = true;
	elements.inputContainer.classList.add('firing');

	resetBadges();

	// Start the turn immediately (don't wait for animation)
	runTurn(message, attachment);

	// Dissolve → plunger pullback → projectile launch (adaptive duration)
	await runPlungerAnimation(message);

	// 500ms settle pause after plunger completes
	await new Promise((resolve) => setTimeout(resolve, 500));

	scheduleLoadingScreen();

	elements.sendButton.disabled = false;
	elements.inputContainer.classList.remove('firing');
}

// ========================================
// STREAMING STATE
// ========================================

const streamingState = {
	filter: new ReplyFilter(),
	buffer: '',                  // committed display text of the current turn
	turnId: null,                // turn_id from this send's turn_started
	tools: [],                   // [{kind: 'tool', name, ok}] finished tool calls of the turn
	toolCalls: new Map(),        // tool_id -> in-flight card, live only
	settledTools: new Map(),     // tool_id -> settled card, kept for late tool_summary frames
	userMessage: null,           // the message that started the current turn
	inCodeFence: false,
	fenceLanguage: '',
	renderedLength: 0,
	firstBlockRendered: false,
	activeAnimations: 0,
	streamFinished: false,
	containerReady: false,
	pendingSettle: null,         // turn-end display work waiting for the container
	thinkingActive: false,       // true while receiving thinking tokens
	thinkingBuffer: '',          // accumulated thinking text of the current step
	thinkingElement: null,       // the .thinking-stream-content DOM node
	renderBlocked: false,        // true during the thinking→text transition
	isGenerating: false,         // true from send until the turn's display settles
	turnNotices: [],             // model_error notices removed when the turn settles
	displayedExchange: null,     // {user, parts, persisted} on screen in the response box
	deferredProactive: []        // proactive message contents held while a turn runs
};

const DEFAULT_EMOTION = '🤖';
const TURN_FRAME_TYPES = new Set(['assistant_delta', 'thinking', 'tool', 'context_reset', 'model_error', 'provider_notice']);

// Resolved once the page-load restore has decided what the response box shows;
// history.js waits on it before computing its first page.
let resolveSessionRestored;
const sessionRestored = new Promise((resolve) => { resolveSessionRestored = resolve; });

let loadingScreenTimer = null;

function clearPendingLoadingScreen() {
	if (!loadingScreenTimer) return;
	clearTimeout(loadingScreenTimer);
	loadingScreenTimer = null;
}

function showLoadingScreenIfStillPending() {
	if (
		streamingState.firstBlockRendered ||
		streamingState.thinkingElement ||
		streamingState.pendingSettle ||
		streamingState.streamFinished
	) {
		return;
	}

	showLoadingScreen();
}

function scheduleLoadingScreen(delayMs = 0) {
	clearPendingLoadingScreen();

	if (delayMs > 0) {
		loadingScreenTimer = setTimeout(() => {
			loadingScreenTimer = null;
			showLoadingScreenIfStillPending();
		}, delayMs);
		return;
	}

	showLoadingScreenIfStillPending();
}

function stopLoadingAnimation() {
	clearPendingLoadingScreen();
	if (elements.loadingScreen._stopLoading) {
		elements.loadingScreen._stopLoading();
	}
}

// ========================================
// GENERATION CANCEL
// ========================================

function cancelGeneration() {
	if (!streamingState.isGenerating) return;
	AppState.apiClient.chat.halt();
}

function setGenerating(active) {
	streamingState.isGenerating = active;
	const btn = elements.sendButton;

	if (active) {
		btn.classList.add('stop-mode');
		btn.setAttribute('aria-label', 'Stop generation');
	} else {
		btn.classList.remove('stop-mode');
		btn.setAttribute('aria-label', 'Send message');
	}
}

// ========================================
// BLOCK RENDERING
// ========================================

function extractCompleteBlocks(text, state) {
	const blocks = [];
	let currentPos = 0;

	if (!state.inCodeFence) {
		while (currentPos < text.length && text[currentPos] === '\n') {
			currentPos++;
		}
	}

	while (currentPos < text.length) {
		if (state.inCodeFence) {
			// Look for closing fence - must be at line start or buffer start
			const fenceEnd = text.indexOf('```', currentPos);

			if (fenceEnd !== -1) {
				const beforeFence = fenceEnd > 0 ? text[fenceEnd - 1] : '\n';
				if (beforeFence === '\n' || fenceEnd === 0) {
					const blockEnd = fenceEnd + 3;
					const nextNewline = text.indexOf('\n', blockEnd);
					const actualEnd = nextNewline !== -1 ? nextNewline + 1 : blockEnd;

					blocks.push({
						type: 'code',
						content: text.substring(currentPos, actualEnd),
						language: state.fenceLanguage
					});
					currentPos = actualEnd;
					state.inCodeFence = false;
					state.fenceLanguage = '';
				} else {
					// Not a real fence at line start, wait for more data
					break;
				}
			} else {
				break;
			}
		} else {
			while (currentPos < text.length && text[currentPos] === '\n') {
				currentPos++;
			}
			if (currentPos >= text.length) break;

			const fenceMatch = text.substring(currentPos).match(/^```(\w*)\n/);
			if (fenceMatch) {
				state.inCodeFence = true;
				state.fenceLanguage = fenceMatch[1] || '';
				currentPos += fenceMatch[0].length;
				continue;
			}

			const paraEnd = text.indexOf('\n\n', currentPos);

			if (paraEnd !== -1) {
				const nextFence = text.indexOf('\n```', currentPos);

				if (nextFence === -1 || nextFence > paraEnd) {
					const paraContent = text.substring(currentPos, paraEnd).trim();
					if (paraContent.length > 0) {
						blocks.push({
							type: 'paragraph',
							content: paraContent
						});
					}
					currentPos = paraEnd + 2;
				} else {
					break;
				}
			} else {
				break;
			}
		}
	}

	return { blocks, consumedLength: currentPos };
}

// Follow the newest content unless the reader has scrolled away from it.
const FOLLOW_BOTTOM_SLACK_PX = 80;

function isNearBottom() {
	const box = elements.responseBox;
	return box.scrollHeight - box.scrollTop - box.clientHeight <= FOLLOW_BOTTOM_SLACK_PX;
}

function followNewest(wasNearBottom) {
	if (wasNearBottom) elements.responseBox.scrollTop = elements.responseBox.scrollHeight;
}

function appendToResponse(html) {
	const wrapper = document.createElement('div');
	wrapper.className = 'response-block';
	wrapper.innerHTML = html;

	const stick = isNearBottom();

	streamingState.activeAnimations++;
	elements.responseBox.classList.add('no-scroll');

	// Haptic feedback when message block appears
	hapticFeedback(30);

	const onAnimEnd = (e) => {
		if (e && e.animationName && !/bubble-up/.test(e.animationName)) return;
		wrapper.removeEventListener('animationend', onAnimEnd);
		streamingState.activeAnimations = Math.max(0, streamingState.activeAnimations - 1);
		if (streamingState.streamFinished && streamingState.activeAnimations === 0) {
			elements.responseBox.classList.remove('no-scroll');
		}
	};
	wrapper.addEventListener('animationend', onAnimEnd);

	elements.responseContent.appendChild(wrapper);
	followNewest(stick);
	ensureResponseStatusBadgePosition();
	scheduleInitialStatusBadge();

	if (!streamingState.firstBlockRendered) {
		streamingState.firstBlockRendered = true;
		stopLoadingAnimation();
	}
}

function renderStreamingBlocks(options = {}) {
	const { force = false } = options;
	const pendingText = streamingState.buffer.substring(streamingState.renderedLength);
	if (!pendingText) return;

	const { blocks, consumedLength } = extractCompleteBlocks(pendingText, streamingState);
	blocks.forEach(block => appendToResponse(renderMarkdown(block.content)));
	streamingState.renderedLength += consumedLength;

	if (force) {
		const leftover = streamingState.buffer.substring(streamingState.renderedLength);
		if (leftover.trim()) {
			appendToResponse(renderMarkdown(leftover.trim()));
		}
		streamingState.renderedLength = streamingState.buffer.length;
		streamingState.inCodeFence = false;
		streamingState.fenceLanguage = '';
	}
}

// A live tool card is appended in the thread's own markup, so it carries the
// same expandable input/output a reload renders from the stored row.
function appendToolCall(part) {
	const stick = isNearBottom();
	const holder = document.createElement('div');
	holder.innerHTML = renderToolCall(part);
	const el = holder.firstElementChild;
	part.el = el;
	elements.responseContent.appendChild(el);
	followNewest(stick);
	return el;
}

function appendDisplayText(text) {
	if (!text) return;
	streamingState.buffer += text;
	if (streamingState.containerReady && !streamingState.renderBlocked) {
		renderStreamingBlocks();
	}
}

function resetStreamDisplay() {
	streamingState.buffer = '';
	streamingState.inCodeFence = false;
	streamingState.fenceLanguage = '';
	streamingState.renderedLength = 0;
	streamingState.firstBlockRendered = false;
	streamingState.activeAnimations = 0;
	streamingState.streamFinished = false;
	streamingState.thinkingActive = false;
	streamingState.thinkingBuffer = '';
	streamingState.thinkingElement = null;
	streamingState.renderBlocked = false;
	streamingState.toolCalls = new Map();
	streamingState.settledTools = new Map();
}

// Render the rest of the turn's text and release the response box.
function finishStreamDisplay() {
	if (streamingState.thinkingActive) {
		transitionThinkingToText();
	}
	streamingState.renderBlocked = false;
	renderStreamingBlocks({ force: true });
	addCodeBlockButtons(elements.responseContent);
	stopLoadingAnimation();
	streamingState.streamFinished = true;
	streamingState.activeAnimations = 0;
	elements.responseBox.classList.remove('no-scroll');
}

// ========================================
// TURN LIFECYCLE
// ========================================

// Every turn renders into a bubble in the thread, so a reply wears one chrome
// whether it is live or a later page load re-renders it from history.
function openExchange(userText) {
	elements.responseContent = window.InlineHistory.beginExchange(userText);
	return elements.responseContent;
}

function ensureExchange(userText) {
	if (elements.responseContent === elements.responseRoot) openExchange(userText);
	return elements.responseContent;
}

// A turn that produced nothing leaves no pair behind: the server holds no such
// exchange, so an empty bubble would be the one thing a reload could not show.
function discardExchange() {
	if (elements.responseContent === elements.responseRoot) return;
	elements.responseContent.parentElement.remove();
	elements.responseContent = elements.responseRoot;
}

function beginTurn(userMessage) {
	streamingState.displayedExchange = null;
	resetStreamDisplay();
	streamingState.filter = new ReplyFilter();
	streamingState.turnId = null;
	streamingState.tools = [];
	streamingState.userMessage = userMessage;
	streamingState.containerReady = true;
	streamingState.pendingSettle = null;
	setGenerating(true);

	clearCompletionBadgeTimer();
	removeExistingCompletionBadge();

	// The finished exchange already sits in the thread as a pair; this turn opens
	// a fresh bubble below it, which is also where the user's own message shows.
	openExchange(userMessage);
	elements.responseBox.classList.add('no-scroll');
	AppState.responseActive = true;

	if (streamingState.thinkingActive) {
		renderThinkingBuffer();
	} else {
		renderStreamingBlocks();
	}

	if (streamingState.pendingSettle) {
		const settle = streamingState.pendingSettle;
		streamingState.pendingSettle = null;
		settle();
	}
}

async function runTurn(message, attachment) {
	beginTurn(message);
	const handlers = AppState.apiClient.eventHandlers.onMessage;
	handlers.push(handleTurnFrame);

	let result;
	try {
		result = await AppState.apiClient.chat.sendMessage(message, attachment);
	} catch (error) {
		failTurn(error, message);
		return;
	} finally {
		handlers.splice(handlers.indexOf(handleTurnFrame), 1);
	}

	if (result.status === 'complete') {
		completeTurn(result);
	} else {
		stopTurn(result);
	}
}

function handleTurnFrame(frame) {
	if (frame.type === 'turn_started') {
		if (streamingState.turnId === null) streamingState.turnId = frame.turn_id;
		return;
	}
	if (!TURN_FRAME_TYPES.has(frame.type) || frame.turn_id !== streamingState.turnId) return;

	switch (frame.type) {
		case 'assistant_delta':
			if (streamingState.thinkingActive) {
				if (streamingState.containerReady) {
					transitionThinkingToText();
				} else {
					streamingState.thinkingActive = false;
					streamingState.thinkingBuffer = '';
				}
			}
			appendDisplayText(streamingState.filter.feed(frame.entry_id, frame.content));
			break;
		case 'thinking':
			updateThinkingIndicator(true);
			handleThinkingToken(frame.content);
			break;
		case 'tool':
			if (frame.event === 'tool_summary') {
				// A late display description, not a lifecycle event: swap it into
				// the card in place and do not flush the provider step. The card
				// may already be settled or still running (a slow tool).
				const part = streamingState.settledTools.get(frame.tool_id)
					|| streamingState.toolCalls.get(frame.tool_id);
				if (part) {
					part.input = frame.summary;
					if (part.el && part.el.isConnected) {
						const holder = document.createElement('div');
						holder.innerHTML = renderToolCall(part);
						part.el.replaceWith(holder.firstElementChild);
						part.el = holder.firstElementChild;
					}
				}
				break;
			}
			// A tool event ends the provider step.
			appendDisplayText(streamingState.filter.flush());
			if (frame.event === 'tool_executing') {
				const part = { name: frame.tool_name, input: frame.arguments ?? null, output: '', ok: true, settled: false };
				streamingState.toolCalls.set(frame.tool_id, part);
				appendToolCall(part);
			} else if (frame.event === 'tool_completed' || frame.event === 'tool_error') {
				const ok = frame.event === 'tool_completed';
				const part = streamingState.toolCalls.get(frame.tool_id) || { name: frame.tool_name, input: null };
				part.ok = ok;
				part.output = frame.result ?? '';
				part.settled = true;
				// Re-render in place: the card grows the result and never keeps a
				// stale "ok" while the tool was still running.
				if (part.el && part.el.isConnected) {
					const holder = document.createElement('div');
					holder.innerHTML = renderToolCall(part);
					part.el.replaceWith(holder.firstElementChild);
					part.el = holder.firstElementChild;
				}
				// Keep the settled card handle so a late tool_summary frame can
				// swap its description in place; history rows carry no summary,
				// so a reload shows the raw input again.
				streamingState.toolCalls.delete(frame.tool_id);
				streamingState.settledTools.set(frame.tool_id, part);
				streamingState.tools.push({ kind: 'tool', name: frame.tool_name, ok });
			}
			break;
		case 'context_reset':
			// The server is regenerating the reply: everything streamed so far is void.
			streamingState.filter = new ReplyFilter();
			resetStreamDisplay();
			if (streamingState.containerReady) {
				elements.responseContent.innerHTML = '';
			}
			break;
		case 'model_error':
			appendDisplayText(streamingState.filter.flush());
			streamingState.turnNotices.push(showNotice(`Model error: ${frame.message} — retrying`, 'alert'));
			break;
		case 'provider_notice':
			// Transient provider retry/fallback status from the server; removed at turn end
			// with the other turn notices.
			streamingState.turnNotices.push(showNotice(frame.message, 'alert'));
			break;
	}
}

// Run turn-end display work once the response container is ready, then
// release the send button and show proactive messages held during the turn.
function endTurn(finalize) {
	const settle = () => {
		finalize();
		setGenerating(false);
		streamingState.turnNotices.splice(0).forEach(removeNotice);
		streamingState.deferredProactive.splice(0).forEach(showProactiveMessage);
	};
	if (streamingState.containerReady) {
		settle();
	} else {
		streamingState.pendingSettle = settle;
	}
}

function completeTurn(result) {
	appendDisplayText(streamingState.filter.finish());
	if (!streamingState.filter.hasText && result.response) {
		appendDisplayText(filterReplyText(result.response));
	}

	// CROSS-FILE-COUPLING: tools/implementations/domaindoc_tool.py's tool name must stay
	// `domaindoc_tool`, or this literal stops matching and the domain list never refreshes.
	if (result.toolsUsed.includes('domaindoc_tool')) {
		window.domainManager.fetchDomains().catch((error) => {
			console.error('Domain list refresh after domaindoc_tool failed; the popover shows the previous list:', error);
		});
	}

	endTurn(() => {
		finishStreamDisplay();
		scheduleCompletionBadge();
		updateThinkingIndicator(false, result.emotion || DEFAULT_EMOTION);

		const text = streamingState.buffer.trimEnd();
		streamingState.displayedExchange = {
			user: streamingState.userMessage,
			parts: [...streamingState.tools, ...(text ? [{ kind: 'text', text }] : [])],
			persisted: true
		};
		window.repulsionFeedback.scanResponse(text);
		window.InlineHistory.notifyNewMessage();
	});
}

// A stopped turn (halt, disconnect, stall) keeps its partial reply server-side.
function stopTurn(result) {
	appendDisplayText(streamingState.filter.finish());
	if (result.reason === 'stall') {
		showNotice('MIRA stalled — the server stopped this reply', 'error');
	} else if (result.reason === 'disconnect') {
		// core.js re-reads the saved partial reply after reconnecting.
		AppState.responseInterrupted = true;
		showNotice('The connection dropped mid-reply — the server stopped this reply', 'alert');
	}
	endTurn(() => settleUnfinishedTurn(true));
}

// Codes for a send the server never took: the message goes to Retry Send.
const UNSENT_TURN_CODES = new Set(['TURN_BUSY', 'CONNECTION_LOST', 'NOT_AUTHENTICATED']);

function failTurn(error, message) {
	console.error('Turn failed:', error);
	appendDisplayText(streamingState.filter.finish());
	if (UNSENT_TURN_CODES.has(error.code)) {
		queueMessage(message);
		showNotice(`${error.message} — your message is in the Retry Send queue`, 'error');
	} else {
		showNotice(error.message, 'error');
	}
	endTurn(() => settleUnfinishedTurn(false));
}

// Display tail of a stopped or failed turn: whatever streamed stays on screen.
function settleUnfinishedTurn(persisted) {
	finishStreamDisplay();
	updateThinkingIndicator(false);
	const text = streamingState.buffer.trimEnd();
	streamingState.displayedExchange = (text || streamingState.tools.length)
		? { user: streamingState.userMessage, parts: [...streamingState.tools, ...(text ? [{ kind: 'text', text }] : [])], persisted }
		: null;
	if (!streamingState.displayedExchange) discardExchange();
}

// Reply parts for the persisted rows of one turn (core.js restores an
// interrupted reply with assembleTurnContent → showResponse).
function assembleTurnContent(turnMessages) {
	return replyPartsFromRows(turnMessages);
}

// Replace the reply on screen with the persisted version of the last turn.
// A turn started since then owns the response box, so it is left alone.
function showResponse(parts) {
	if (streamingState.isGenerating || parts.length === 0) return;
	ensureExchange(streamingState.userMessage);
	elements.responseContent.innerHTML = renderReplyParts(parts);
	addCodeBlockButtons(elements.responseContent);
	elements.responseBox.style.display = 'block';
	AppState.responseActive = true;
	streamingState.displayedExchange = { user: streamingState.userMessage, parts, persisted: true };
}

// ========================================
// PROACTIVE MESSAGES
// ========================================

// An assistant message MIRA sent on its own (core.js forwards the frame).
// Held while a turn runs, then shown below that turn's reply.
function handleProactiveMessage(frame) {
	if (streamingState.isGenerating) {
		streamingState.deferredProactive.push(frame.content);
		return;
	}
	showProactiveMessage(frame.content);
}

function showProactiveMessage(content) {
	const text = filterReplyText(content);
	if (!text) return;

	if (!streamingState.displayedExchange) {
		openExchange(null);
		streamingState.displayedExchange = { user: null, parts: [], persisted: true };
	}
	streamingState.displayedExchange.parts.push({ kind: 'text', text });

	const block = document.createElement('div');
	const stick = isNearBottom();

	block.className = 'response-block';
	block.innerHTML = renderMarkdown(text);
	elements.responseContent.appendChild(block);
	addCodeBlockButtons(block);
	elements.responseBox.style.display = 'block';
	AppState.responseActive = true;
	followNewest(stick);

	hapticFeedback(30);
	window.InlineHistory.notifyNewMessage();
}

// ========================================
// THINKING / EMOTION INDICATOR
// ========================================

function updateThinkingIndicator(isThinking, emotion = DEFAULT_EMOTION) {
	const indicator = document.getElementById('thinking-indicator');

	if (isThinking) {
		indicator.classList.remove('standby');
		indicator.classList.add('active');
		indicator.innerHTML = `<img src="/assets/images/icons/think.png?v=${ASSET_VERSION}" alt="">`;
	} else {
		indicator.classList.remove('active');
		indicator.classList.add('standby');
		const emoji = document.createElement('span');
		emoji.className = 'emotion-emoji';
		emoji.style.paddingLeft = '0';
		emoji.textContent = emotion;
		indicator.replaceChildren(emoji);
	}
}

// ========================================
// CODE BLOCK COPY BUTTONS
// ========================================

function addCodeBlockButtons(container) {
	container.querySelectorAll('pre code').forEach((block) => {
		const preElement = block.parentElement;
		if (preElement.parentElement.classList.contains('code-block-wrapper')) return;

		const wrapper = document.createElement('div');
		wrapper.className = 'code-block-wrapper';
		preElement.parentElement.insertBefore(wrapper, preElement);
		wrapper.appendChild(preElement);

		const copyBtn = document.createElement('button');
		copyBtn.className = 'code-copy-btn';
		copyBtn.textContent = 'Copy';

		copyBtn.addEventListener('click', async () => {
			const codeText = block.textContent;

			try {
				if (navigator.clipboard && window.isSecureContext) {
					await navigator.clipboard.writeText(codeText);
				} else {
					const textarea = document.createElement('textarea');
					textarea.value = codeText;
					textarea.style.position = 'absolute';
					textarea.style.left = '-9999px';
					document.body.appendChild(textarea);
					textarea.select();
					document.execCommand('copy');
					document.body.removeChild(textarea);
				}

				copyBtn.textContent = 'Copied!';
				copyBtn.classList.add('copied');

				setTimeout(() => {
					copyBtn.textContent = 'Copy';
					copyBtn.classList.remove('copied');
				}, 2000);
			} catch (err) {
				console.error('Failed to copy code:', err);
				copyBtn.textContent = 'Failed';
				setTimeout(() => {
					copyBtn.textContent = 'Copy';
				}, 2000);
			}
		});

		wrapper.appendChild(copyBtn);
	});
}

// ========================================
// SESSION RESTORE
// ========================================

function showWelcomeMessage() {
	elements.responseRoot.innerHTML = `
		<div id="welcome-message">
			<p>Welcome to MIRA! I'm ready to assist you.</p>
			<p>Start by typing a message below.</p>
		</div>
	`;
	elements.responseBox.classList.add('active');
	AppState.responseActive = true;
}

// Show the newest exchange from history: the rows after the last user row.
// The thread is the conversation: history.js renders it, including the empty
// case. Startup only releases the pager and fetches the announcement.
async function prepareSession() {
	// history.js awaits this before its first page, so it resolves first.
	resolveSessionRestored();
	await showAnnouncement();
}

// Operator announcement from user preferences, unless this browser dismissed it.
async function showAnnouncement() {
	let announcement;
	try {
		const userData = await AppState.apiClient.data.getData('user');
		announcement = userData.preferences.announcement;
	} catch (error) {
		console.warn('Announcement check failed; no announcement banner this page load:', error);
		return;
	}
	if (!announcement || localStorage.getItem('dismissed-announcement') === announcement.id) return;

	const banner = document.getElementById('announcement-banner');
	document.getElementById('announcement-text').innerHTML = announcement.message;
	banner.style.display = '';
	banner.querySelector('.system-alert-dismiss').addEventListener('click', () => {
		localStorage.setItem('dismissed-announcement', announcement.id);
	}, { once: true });
}

// ========================================
// THINKING STREAM & TRANSITION
// ========================================

function handleThinkingToken(content) {
	streamingState.thinkingActive = true;
	streamingState.thinkingBuffer += content;

	if (!streamingState.containerReady) {
		return;
	}

	renderThinkingBuffer();
}

function renderThinkingBuffer() {
	if (!streamingState.thinkingElement) {
		// First thinking token of the step — dismiss loading screen, show response area
		stopLoadingAnimation();

		const el = document.createElement('div');
		el.className = 'thinking-stream-content';
		elements.responseContent.appendChild(el);
		elements.responseBox.style.display = 'block';
		elements.responseBox.classList.add('no-scroll');
		AppState.responseActive = true;
		streamingState.thinkingElement = el;
	}

	streamingState.thinkingElement.textContent = streamingState.thinkingBuffer;
	elements.responseBox.scrollTop = elements.responseBox.scrollHeight;
}

function transitionThinkingToText() {
	if (!streamingState.thinkingActive) return;
	streamingState.thinkingActive = false;

	const thinkingEl = streamingState.thinkingElement;
	const thinkingText = streamingState.thinkingBuffer;
	streamingState.thinkingElement = null;
	streamingState.thinkingBuffer = '';

	if (!thinkingEl || !thinkingText) {
		return;
	}

	// Gate output blocks for the full transition: 200ms pause + 300ms collapse
	streamingState.renderBlocked = true;

	setTimeout(() => {
		// Measure streaming text height as the collapse start point
		const startHeight = thinkingEl.offsetHeight;

		// Create drawer at that height so the swap is visually seamless
		const drawer = createThinkingDrawer(thinkingText);
		drawer.classList.add('active');
		drawer.style.height = startHeight + 'px';
		thinkingEl.replaceWith(drawer);

		// Zip closed: collapse to 28px (CSS transition: height 0.3s ease)
		requestAnimationFrame(() => {
			drawer.style.height = '';
		});

		// Unblock once the collapse finishes
		drawer.addEventListener('transitionend', function handler(e) {
			if (e.propertyName === 'height') {
				drawer.removeEventListener('transitionend', handler);
				streamingState.renderBlocked = false;
				renderStreamingBlocks();
			}
		});

		// Unblock even if transitionend doesn't fire
		setTimeout(() => {
			if (streamingState.renderBlocked) {
				streamingState.renderBlocked = false;
				renderStreamingBlocks();
			}
		}, 500);
	}, 200);
}

// The same markup history renders for a past turn's reasoning. The expand is a
// delegated click on .thinking-drawer-toggle, so a re-rendered drawer behaves
// identically without rebinding.
function createThinkingDrawer(thinkingText) {
	const holder = document.createElement('div');
	holder.innerHTML = renderThinkingDrawer(thinkingText);
	return holder.firstElementChild;
}

// ========================================
// MODULE EXPORTS
// ========================================

window.sendMessage = sendMessage;
window.cancelGeneration = cancelGeneration;
window.handleFileSelect = handleFileSelect;
window.downscaleImage = downscaleImage;
window.clearAttachment = clearAttachment;
window.renderQueuedMessages = renderQueuedMessages;
window.sendSingleQueuedMessage = sendSingleQueuedMessage;
window.removeQueuedMessage = removeQueuedMessage;
window.editQueuedMessage = editQueuedMessage;
window.saveQueuedMessageEdit = saveQueuedMessageEdit;
window.cancelQueuedMessageEdit = cancelQueuedMessageEdit;
window.prepareSession = prepareSession;
window.showWelcomeMessage = showWelcomeMessage;
window.handleProactiveMessage = handleProactiveMessage;
window.assembleTurnContent = assembleTurnContent;
window.showResponse = showResponse;
