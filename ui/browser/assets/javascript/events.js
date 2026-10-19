/**
 * EVENTS.JS — the chat page's DOM wiring: send and stop, keyboard, the file
 * picker and image paste, popover open/close, and the Retry Send buttons.
 *
 * Three paths reach a send — the form's submit, the send button's click, and
 * Enter on a desktop keyboard — and each one checks `.stop-mode` first, so the
 * same button stops a running turn instead of starting a second one.
 *
 * The send button's `pointerdown` preventDefault is load-bearing on iOS: focus
 * has to stay in the textarea, or tapping collapses the keyboard, the visual
 * viewport resizes mid-tap, and the synthetic click lands where the button no
 * longer is.
 *
 * `setupDelegatedEvents` binds the retry-queue buttons once, on
 * `#queue-messages`, so the rows messaging.js re-renders stay clickable without
 * rebinding. Everything else binds directly at startup against ids captured by
 * core.js's `elements`, so it must run after `initializeMira` has built the
 * client.
 *
 * Load order: last of the chat modules.
 */

// ========================================
// UTILITY FUNCTIONS
// ========================================

function isMobile() {
	const isSmallScreen = window.innerWidth <= 768;
	const isTouchFirst = window.matchMedia('(pointer: coarse)').matches;
	return isSmallScreen || isTouchFirst;
}

// ========================================
// EVENT HANDLING
// ========================================

function handleClickOutside(e) {
	const queueContainer = document.querySelector('.queue-container');
	if (queueContainer && elements.queuePopover &&
		!queueContainer.contains(e.target) &&
		!elements.queuePopover.contains(e.target)) {
		elements.queuePopover.classList.remove('active');
	}
}

// ========================================
// EVENT LISTENER SETUP
// ========================================

function setupEventListeners() {
	document.addEventListener('click', handleClickOutside);

	elements.inputContainer.addEventListener('submit', (e) => {
		e.preventDefault();
		if (elements.sendButton.classList.contains('stop-mode')) {
			window.cancelGeneration();
		} else if (!elements.sendButton.disabled) {
			window.sendMessage();
		}
	});

	// iOS Safari: tapping the send button while the textarea is focused steals focus,
	// which dismisses the virtual keyboard and resizes the visual viewport mid-tap. The
	// synthetic click then lands at the original coordinates — where the button no longer
	// is — and the tap is lost. Preventing pointerdown's default keeps focus on the
	// textarea, the keyboard stays open, and the click fires on the intended target.
	elements.sendButton.addEventListener('pointerdown', (e) => {
		e.preventDefault();
	});

	elements.sendButton.addEventListener('click', (e) => {
		e.preventDefault();
		if (elements.sendButton.classList.contains('stop-mode')) {
			window.cancelGeneration();
		} else if (!elements.sendButton.disabled) {
			window.sendMessage();
		}
	});

	elements.messageInput.addEventListener('keydown', (e) => {
		// On mobile, Enter adds newline (send via button); on desktop, Enter sends (Shift+Enter for newline)
		if (e.key === 'Enter' && !e.shiftKey && !isMobile()) {
			e.preventDefault();
			if (!elements.sendButton.disabled) window.sendMessage();
		}
	});

	if (elements.attachmentButton) {
		elements.attachmentButton.addEventListener('click', (e) => {
			e.stopPropagation();
			if (AppState.attachedFiles.length > 0) {
				window.toggleAttachmentPopover();
			} else {
				elements.fileInput.click();
			}
		});
	}

	if (elements.attachmentAddBtn) {
		elements.attachmentAddBtn.addEventListener('click', () => {
			elements.fileInput.click();
		});
	}

	elements.fileInput.addEventListener('change', window.handleFileSelect);

	const attachmentPopoverClose = elements.attachmentPopover?.querySelector('.popover-close');
	if (attachmentPopoverClose) {
		attachmentPopoverClose.addEventListener('click', (e) => {
			e.stopPropagation();
			window.closeAttachmentPopover();
		});
	}

	document.addEventListener('click', (e) => {
		if (elements.attachmentPopover &&
			elements.attachmentPopover.classList.contains('active') &&
			!elements.attachmentPopover.contains(e.target) &&
			!elements.attachmentButton.contains(e.target)) {
			window.closeAttachmentPopover();
		}
	});

	document.addEventListener('paste', async (e) => {
		const items = e.clipboardData?.items;
		if (!items) return;

		for (const item of items) {
			if (item.type.startsWith('image/')) {
				e.preventDefault();
				const file = item.getAsFile();
				if (file) {
					const maxSize = 5 * 1024 * 1024;
					if (file.size > maxSize) {
						alert('Image must be less than 5MB');
						return;
					}

					try {
						const resizedFile = await window.downscaleImage(file, 540, 120);

						AppState.attachedFiles.push(resizedFile);

						window.updateAttachmentPopover();
						elements.attachmentButton.classList.add('has-attachment');
						window.updateAttachmentCount();
					} catch (error) {
						console.error('Failed to process pasted image:', error);
						alert('Failed to process pasted image. Please try again.');
					}
				}
				break;
			}
		}
	});

	if (elements.queueIndicator) {
		elements.queueIndicator.addEventListener('click', (e) => {
			e.stopPropagation();
			toggleQueuePopover();
		});
	}

}

function setupDelegatedEvents() {
	if (elements.queueMessages) {
		elements.queueMessages.addEventListener('click', async (e) => {
			const index = parseInt(e.target.dataset.index);
			if (e.target.classList.contains('queue-send-btn')) {
				await window.sendSingleQueuedMessage(index);
			} else if (e.target.classList.contains('queue-remove-btn')) {
				window.removeQueuedMessage(index);
			} else if (e.target.classList.contains('queue-edit-btn')) {
				window.editQueuedMessage(index);
			} else if (e.target.classList.contains('queue-save-btn')) {
				window.saveQueuedMessageEdit(index);
			} else if (e.target.classList.contains('queue-cancel-btn')) {
				window.cancelQueuedMessageEdit();
			}
		});
	}

	document.querySelectorAll('.popover-close').forEach(btn => {
		btn.addEventListener('click', (e) => {
			e.stopPropagation();
			const popover = btn.closest('.retry-popover');
			if (popover) {
				popover.classList.remove('active');
			}
		});
	});
}

// ========================================
// MODULE EXPORTS
// ========================================

// Export setup functions needed by core.js
window.setupEventListeners = setupEventListeners;
window.setupDelegatedEvents = setupDelegatedEvents;
