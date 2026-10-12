/**
 * SIDEBAR.JS - App Sidebar Navigation for All Pages
 *
 * Handles desktop collapse/expand (persisted via localStorage),
 * mobile toggle/close, swipe gestures, keyboard navigation,
 * focus trapping, and the action buttons: History (all pages), and on the
 * chat page Pause/Unpause (inactivity-warning.js) and Wrap Up
 * (core.js Functions.collapseSegment).
 */
(function() {
	'use strict';

	const sidebar = document.getElementById('app-sidebar');
	const toggle = document.getElementById('sidebar-toggle');
	const backdrop = document.getElementById('sidebar-backdrop');
	const collapseToggle = document.getElementById('sidebar-collapse-toggle');
	const expandToggle = document.getElementById('sidebar-expand-toggle');
	if (!sidebar || !toggle) return;

	const MOBILE_BP = 768;
	const STORAGE_KEY = 'mira-sidebar-collapsed';
	const SIDEBAR_WIDTH = 250;

	function isMobile() {
		return window.innerWidth <= MOBILE_BP;
	}

	function isOnChatPage() {
		const path = window.location.pathname;
		return path === '/' || path === '/chat' || path === '/chat/' || path === '/chat/index.html';
	}

	// --- Desktop collapse/expand ---

	function setCollapsed(collapsed, animate) {
		if (!animate) {
			sidebar.style.transition = 'none';
			document.body.style.transition = 'none';
		}

		if (collapsed) {
			sidebar.classList.add('collapsed');
			document.body.style.marginLeft = '0px';
			if (expandToggle) expandToggle.classList.add('visible');
		} else {
			sidebar.classList.remove('collapsed');
			document.body.style.marginLeft = SIDEBAR_WIDTH + 'px';
			if (expandToggle) expandToggle.classList.remove('visible');
		}

		if (!animate) {
			sidebar.offsetHeight;
			sidebar.style.transition = '';
			document.body.style.transition = '';
		}

		try {
			localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
		} catch (e) { /* storage unavailable */ }
	}

	function collapseSidebar() {
		if (isMobile()) { closeSidebar(); return; }
		setCollapsed(true, true);
	}

	function expandSidebar() {
		if (isMobile()) return;
		setCollapsed(false, true);
	}

	// Restore collapsed state on load (desktop only, no animation)
	// Applies visual state directly to avoid redundant localStorage write
	function restoreCollapseState() {
		if (isMobile()) {
			document.body.style.marginLeft = '';
			return;
		}
		try {
			if (localStorage.getItem(STORAGE_KEY) !== '1') return;
		} catch (e) { return; }

		sidebar.style.transition = 'none';
		document.body.style.transition = 'none';

		sidebar.classList.add('collapsed');
		document.body.style.marginLeft = '0px';
		if (expandToggle) expandToggle.classList.add('visible');

		sidebar.offsetHeight;
		sidebar.style.transition = '';
		document.body.style.transition = '';
	}

	restoreCollapseState();

	if (collapseToggle) {
		collapseToggle.addEventListener('click', collapseSidebar);
	}

	if (expandToggle) {
		expandToggle.addEventListener('click', expandSidebar);
	}

	// --- Mobile open/close ---

	function openSidebar() {
		sidebar.classList.add('open');
		sidebar.setAttribute('aria-hidden', 'false');
		backdrop.classList.add('open');
		toggle.setAttribute('aria-expanded', 'true');
		toggle.setAttribute('aria-label', 'Close navigation');
		toggle.style.display = 'none';
		const first = sidebar.querySelector('.sidebar-nav-item');
		if (first) first.focus();
	}

	function closeSidebar() {
		sidebar.classList.remove('open');
		sidebar.setAttribute('aria-hidden', 'true');
		backdrop.classList.remove('open');
		toggle.setAttribute('aria-expanded', 'false');
		toggle.setAttribute('aria-label', 'Open navigation');
		toggle.style.display = '';
		toggle.focus();
	}

	toggle.addEventListener('click', function() {
		if (sidebar.classList.contains('open')) {
			closeSidebar();
		} else {
			openSidebar();
		}
	});

	if (backdrop) {
		backdrop.addEventListener('click', closeSidebar);
	}

	// --- Keyboard: Escape closes, focus trapping ---

	document.addEventListener('keydown', function(e) {
		if (!isMobile() || !sidebar.classList.contains('open')) return;

		if (e.key === 'Escape') {
			e.preventDefault();
			closeSidebar();
			return;
		}

		if (e.key === 'Tab') {
			const focusable = sidebar.querySelectorAll(
				'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
			);
			if (focusable.length === 0) return;

			const first = focusable[0];
			const last = focusable[focusable.length - 1];

			if (e.shiftKey && document.activeElement === first) {
				e.preventDefault();
				last.focus();
			} else if (!e.shiftKey && document.activeElement === last) {
				e.preventDefault();
				first.focus();
			}
		}
	});

	// --- Swipe gestures (mobile) ---

	let touchStartX = 0;
	let touchStartY = 0;
	const SWIPE_THRESHOLD = 50;
	const EDGE_ZONE = 30;

	document.addEventListener('touchstart', function(e) {
		if (!isMobile()) return;
		touchStartX = e.touches[0].clientX;
		touchStartY = e.touches[0].clientY;
	}, { passive: true });

	document.addEventListener('touchend', function(e) {
		if (!isMobile()) return;
		const dx = e.changedTouches[0].clientX - touchStartX;
		const dy = e.changedTouches[0].clientY - touchStartY;

		if (Math.abs(dx) < SWIPE_THRESHOLD || Math.abs(dy) > Math.abs(dx)) return;

		if (dx > 0 && touchStartX < EDGE_ZONE && !sidebar.classList.contains('open')) {
			openSidebar();
		} else if (dx < 0 && sidebar.classList.contains('open')) {
			closeSidebar();
		}
	}, { passive: true });

	// --- Aria state + resize handling ---

	function handleResize() {
		if (isMobile()) {
			sidebar.setAttribute('aria-hidden', sidebar.classList.contains('open') ? 'false' : 'true');
			document.body.style.marginLeft = '';
		} else {
			sidebar.removeAttribute('aria-hidden');
			sidebar.classList.remove('open');
			backdrop.classList.remove('open');
			var isCollapsed = sidebar.classList.contains('collapsed');
			document.body.style.marginLeft = (isCollapsed ? 0 : SIDEBAR_WIDTH) + 'px';
		}
	}

	handleResize();
	var resizeRaf;
	window.addEventListener('resize', function() {
		if (resizeRaf) return;
		resizeRaf = requestAnimationFrame(function() {
			resizeRaf = null;
			handleResize();
		});
	});

	// --- Close sidebar on nav link click (mobile) ---

	sidebar.addEventListener('click', function(e) {
		if (!isMobile()) return;
		var link = e.target.closest('a.sidebar-nav-item');
		if (link) closeSidebar();
	});

	// --- History button ---

	var historyBtn = document.getElementById('sidebar-history-btn');
	historyBtn.addEventListener('click', function() {
		if (isOnChatPage()) {
			window.InlineHistory.toggle();
			historyBtn.textContent = window.InlineHistory.isExpanded
				? 'Hide Chat History'
				: 'Show Chat History';
		} else {
			window.location.href = '/chat?history=1';
		}
	});

	// --- Pause/Unpause button (chat page: state from the session status bar) ---

	var pauseBtn = document.getElementById('sidebar-pause-btn');
	var pauseLabel = document.getElementById('sidebar-pause-label');

	function updateSidebarPauseUI(isPaused, hasSegment) {
		pauseBtn.style.display = hasSegment ? '' : 'none';
		if (isPaused) {
			pauseBtn.classList.add('paused');
			pauseLabel.textContent = 'Unpause';
		} else {
			pauseBtn.classList.remove('paused');
			pauseLabel.textContent = 'Pause';
		}
	}

	// --- Collapse Segment button (chat page only) ---

	var collapseBtn = document.getElementById('sidebar-collapse-btn');

	if (isOnChatPage()) {
		pauseBtn.addEventListener('click', function() {
			window.inactivityWarningManager.togglePause();
		});
		collapseBtn.addEventListener('click', function() {
			window.Functions.collapseSegment();
		});

		// inactivity-warning.js and history.js load after this file
		document.addEventListener('DOMContentLoaded', function() {
			var manager = window.inactivityWarningManager;
			manager.onStatusChange(function(state) {
				updateSidebarPauseUI(state.isPaused, state.hasActiveSegment);
			});
			updateSidebarPauseUI(manager.isPaused, manager.hasActiveSegment);

			// ?history=1 (from the History button on another page) opens the inline history
			var params = new URLSearchParams(window.location.search);
			if (params.get('history') === '1') {
				window.history.replaceState({}, '', window.location.pathname);
				window.InlineHistory.expand();
				historyBtn.textContent = 'Hide Chat History';
			}
		});
	} else {
		collapseBtn.style.display = 'none';
	}
})();
