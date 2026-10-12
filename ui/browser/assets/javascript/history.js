/**
 * HISTORY.JS - Inline Conversation History
 *
 * PURPOSE:
 * Shows earlier conversation inside #response_box, above the current
 * response, paging backward through `history.getHistory({before, limit})`
 * (pages arrive oldest→newest; `nextBefore` pages further back).
 *
 * ROW PRESENTATION (mirrors ui/tui/history.py):
 * - A `user` row starts a pair; the `assistant` and `tool` rows that share a
 *   `metadata.turn_id` form one MIRA reply (rows without a turn_id are a reply
 *   each). A tool row is a pill (`metadata.tool_name`, failed when `is_error`).
 * - Assistant text goes through the same display filter as live replies.
 * - A collapsed segment sentinel (`metadata.status === 'collapsed'`) is a
 *   summary block: `metadata.display_title` plus its `content`. Active and
 *   paused boundaries are skipped.
 * - Leading reply rows of a page whose user row lives in the next older page
 *   are carried over and rendered with that page.
 *
 * The exchange on screen in the response box is left out of the first page
 * (rows from the last user row onward) while it is a persisted exchange; when
 * a send knocks it off screen messaging.js hands it to appendExchange.
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements
 * - messaging.js: streamingState, sessionRestored, ReplyFilter helpers
 *   (filterReplyText, messageDisplayText, isSegmentSentinel,
 *   replyPartsFromRows, renderReplyParts, renderMarkdown, lastUserIndex)
 * - ui.js: hapticFeedback
 *
 * DEPENDENTS:
 * - sidebar.js (toggle, expand, isExpanded), messaging.js (appendExchange, notifyNewMessage)
 *
 * LOAD ORDER:
 * After messaging.js.
 */

const HISTORY_PAGE_LIMIT = 30;
const HISTORY_LOADER_SELECTOR = '.inline-history-loading:not(.inline-history-message)';

const InlineHistoryState = {
	isExpanded: false,
	loaded: false,           // the newest page has been fetched
	hasMore: true,
	nextBefore: null,        // keyset cursor for the next older page
	carry: [],               // leading reply rows held for the next older page
	renderedAny: false,
	isLoading: false,
	savedScrollPosition: null,
	scrollObserver: null
};

function toggleInlineHistory() {
	if (InlineHistoryState.isExpanded) {
		collapseInlineHistory();
	} else {
		expandInlineHistory();
	}
}

function expandInlineHistory() {
	InlineHistoryState.isExpanded = true;
	elements.inlineHistoryContainer.classList.remove('hidden');

	if (InlineHistoryState.savedScrollPosition !== null) {
		elements.responseBox.scrollTop = InlineHistoryState.savedScrollPosition;
	}

	// The observer's first callback loads the newest page when the sentinel is visible
	setupInlineScrollObserver();

	hapticFeedback(30);
}

function collapseInlineHistory() {
	InlineHistoryState.savedScrollPosition = elements.responseBox.scrollTop;

	InlineHistoryState.isExpanded = false;
	elements.inlineHistoryContainer.classList.add('hidden');

	InlineHistoryState.scrollObserver.disconnect();
	InlineHistoryState.scrollObserver = null;

	hideNewMessageToast();

	hapticFeedback(30);
}

// Drop the persisted exchange that is on screen in the response box from the newest page.
function withoutDisplayedExchange(rows) {
	const exchange = streamingState.displayedExchange;
	if (!exchange || !exchange.persisted || streamingState.isGenerating) return rows;
	const userIndex = lastUserIndex(rows);
	return rows.filter((row, i) => i < userIndex || isSegmentSentinel(row));
}

/**
 * Fetch and render the next older page. A page whose rows are all carried
 * over renders nothing, so fetching continues until something renders or
 * history runs out.
 */
async function loadInlineHistory() {
	if (InlineHistoryState.isLoading || !InlineHistoryState.hasMore) return;

	InlineHistoryState.isLoading = true;
	showInlineHistoryLoading();

	// Scroll anchoring: content is inserted above what the user is reading
	const scrollHeightBefore = elements.responseBox.scrollHeight;
	const scrollTopBefore = elements.responseBox.scrollTop;

	try {
		await sessionRestored;
		let fragment;
		do {
			const params = { limit: HISTORY_PAGE_LIMIT };
			if (InlineHistoryState.nextBefore) params.before = InlineHistoryState.nextBefore;
			const page = await AppState.apiClient.history.getHistory(params);

			let rows = InlineHistoryState.loaded ? page.messages : withoutDisplayedExchange(page.messages);
			InlineHistoryState.loaded = true;
			InlineHistoryState.hasMore = page.hasMore;
			InlineHistoryState.nextBefore = page.nextBefore;

			rows = rows.concat(InlineHistoryState.carry);
			InlineHistoryState.carry = [];
			if (InlineHistoryState.hasMore) {
				const start = rows.findIndex(row => row.role === 'user' || isSegmentSentinel(row));
				const split = start === -1 ? rows.length : start;
				InlineHistoryState.carry = rows.slice(0, split);
				rows = rows.slice(split);
			}

			fragment = renderHistoryRows(rows);
		} while (!fragment.hasChildNodes() && InlineHistoryState.hasMore);

		if (fragment.hasChildNodes()) {
			elements.inlineHistoryContainer.querySelector('.inline-history-message')?.remove();
			InlineHistoryState.renderedAny = true;
			elements.inlineHistorySentinel.after(fragment);
		} else if (!InlineHistoryState.renderedAny) {
			showInlineHistoryMessage('No earlier conversation yet');
		}
	} catch (error) {
		console.error('Failed to load inline history:', error);
		showInlineHistoryMessage(`Could not load earlier conversation: ${error.message}`);
	} finally {
		InlineHistoryState.isLoading = false;
		hideInlineHistoryLoading();

		const heightDifference = elements.responseBox.scrollHeight - scrollHeightBefore;
		elements.responseBox.scrollTop = scrollTopBefore + heightDifference;

		// Re-observing re-reports a still-visible sentinel, so short pages keep loading
		const observer = InlineHistoryState.scrollObserver;
		if (observer && InlineHistoryState.hasMore) {
			observer.unobserve(elements.inlineHistorySentinel);
			observer.observe(elements.inlineHistorySentinel);
		}
	}
}

function renderHistoryRows(rows) {
	const fragment = document.createDocumentFragment();
	let pair = null;
	let pairHasReply = false;
	let replyTurnId = null;
	let parts = [];

	const flushReply = () => {
		if (parts.length === 0) return;
		if (!pair || pairHasReply) {
			pair = createHistoryPair(null);
			fragment.appendChild(pair);
		}
		pair.appendChild(createHistoryReply(parts));
		pairHasReply = true;
		parts = [];
	};

	for (const row of rows) {
		if (isSegmentSentinel(row)) {
			if (row.metadata.status !== 'collapsed') continue;
			flushReply();
			pair = null;
			const summary = createSegmentSummary(row);
			if (summary) fragment.appendChild(summary);
			continue;
		}
		if (row.role === 'user') {
			flushReply();
			pair = createHistoryPair(messageDisplayText(row.content));
			pairHasReply = false;
			fragment.appendChild(pair);
			continue;
		}
		if (row.role !== 'assistant' && row.role !== 'tool') continue;

		const turnId = typeof row.metadata.turn_id === 'string' ? row.metadata.turn_id : null;
		if (turnId === null || turnId !== replyTurnId) flushReply();
		replyTurnId = turnId;
		parts.push(...replyPartsFromRows([row]));
	}
	flushReply();

	// A user row with no displayable text and no reply leaves an empty pair
	fragment.querySelectorAll('.inline-history-pair:empty').forEach(el => el.remove());
	return fragment;
}

function createSegmentSummary(sentinel) {
	const title = typeof sentinel.metadata.display_title === 'string' ? sentinel.metadata.display_title : '';
	const body = messageDisplayText(sentinel.content);
	if (!title && !body) return null;

	const summaryBlock = document.createElement('div');
	summaryBlock.className = 'inline-history-segment-summary';

	const titleDiv = document.createElement('div');
	titleDiv.className = 'inline-history-segment-title';
	titleDiv.textContent = title || formatSegmentFallbackTitle(sentinel.timestamp);
	summaryBlock.appendChild(titleDiv);

	if (body) {
		const contentDiv = document.createElement('div');
		contentDiv.className = 'inline-history-segment-content';
		contentDiv.innerHTML = renderMarkdown(body);
		summaryBlock.appendChild(contentDiv);
	}

	return summaryBlock;
}

function formatSegmentFallbackTitle(timestamp) {
	return new Date(timestamp).toLocaleDateString('en-US', {
		weekday: 'short',
		month: 'short',
		day: 'numeric',
		hour: 'numeric',
		minute: '2-digit'
	});
}

// A pair element; `userText` null or empty means a reply with no user message above it.
function createHistoryPair(userText) {
	const pairDiv = document.createElement('div');
	pairDiv.className = 'inline-history-pair';
	if (userText) {
		const userDiv = document.createElement('div');
		userDiv.className = 'inline-history-user';
		userDiv.textContent = userText;
		pairDiv.appendChild(userDiv);
	}
	return pairDiv;
}

function createHistoryReply(parts) {
	const assistantDiv = document.createElement('div');
	assistantDiv.className = 'inline-history-assistant';
	assistantDiv.innerHTML = renderReplyParts(parts);
	return assistantDiv;
}

/**
 * Append the exchange that just left the response box. Before the first page
 * is fetched there is nothing to append to: that fetch includes it.
 */
function appendExchange(exchange) {
	if (!InlineHistoryState.loaded) return;
	const pair = createHistoryPair(exchange.user);
	if (exchange.parts.length > 0) {
		pair.appendChild(createHistoryReply(exchange.parts));
	}
	if (!pair.hasChildNodes()) return;
	elements.inlineHistoryContainer.querySelector('.inline-history-message')?.remove();
	InlineHistoryState.renderedAny = true;
	elements.inlineHistoryContainer.appendChild(pair);
}

function setupInlineScrollObserver() {
	InlineHistoryState.scrollObserver = new IntersectionObserver(
		(entries) => {
			if (entries[0].isIntersecting &&
				!InlineHistoryState.isLoading &&
				InlineHistoryState.hasMore) {
				loadInlineHistory();
			}
		},
		{
			root: elements.responseBox,
			threshold: 0.1
		}
	);

	InlineHistoryState.scrollObserver.observe(elements.inlineHistorySentinel);
}

function showInlineHistoryLoading() {
	let loader = elements.inlineHistoryContainer.querySelector(HISTORY_LOADER_SELECTOR);
	if (!loader) {
		loader = document.createElement('div');
		loader.className = 'inline-history-loading';
		loader.textContent = 'Loading older conversations...';
		elements.inlineHistorySentinel.after(loader);
	}
}

function hideInlineHistoryLoading() {
	const loader = elements.inlineHistoryContainer.querySelector(HISTORY_LOADER_SELECTOR);
	if (loader) loader.remove();
}

// One status line under the sentinel (empty history, load failure).
function showInlineHistoryMessage(text) {
	let line = elements.inlineHistoryContainer.querySelector('.inline-history-message');
	if (!line) {
		line = document.createElement('div');
		line.className = 'inline-history-loading inline-history-message';
		elements.inlineHistorySentinel.after(line);
	}
	line.textContent = text;
}

/**
 * A new message arrived while the history view is open:
 * within one viewport of the bottom auto-scroll silently, otherwise show the toast.
 */
function showNewMessageToast() {
	if (!InlineHistoryState.isExpanded) return;

	const viewportHeight = elements.responseBox.clientHeight;
	const scrollBottom = elements.responseBox.scrollHeight - elements.responseBox.scrollTop - viewportHeight;

	if (scrollBottom <= viewportHeight) {
		elements.responseBox.scrollTo({
			top: elements.responseBox.scrollHeight,
			behavior: 'smooth'
		});
		return;
	}

	elements.newMessageToast.classList.remove('hidden');
	hapticFeedback(50);
}

function hideNewMessageToast() {
	elements.newMessageToast.classList.add('hidden');
}

function scrollToLatest() {
	elements.responseBox.scrollTo({
		top: elements.responseBox.scrollHeight,
		behavior: 'smooth'
	});
	hideNewMessageToast();
}

elements.toastScrollBtn.addEventListener('click', scrollToLatest);

window.InlineHistory = {
	get isExpanded() { return InlineHistoryState.isExpanded; },
	toggle: toggleInlineHistory,
	expand: expandInlineHistory,
	notifyNewMessage: showNewMessageToast,
	appendExchange: appendExchange
};
