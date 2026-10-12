/**
 * EVENTS.JS - Event Handlers & User Interaction
 *
 * PURPOSE:
 * Wires DOM events on the chat page to application functions: send/stop,
 * keyboard shortcuts, attachments (picker and image paste), popover
 * open/close, and the retry-queue actions.
 *
 * DEPENDENCIES:
 * - core.js: AppState, elements
 * - messaging.js: sendMessage, cancelGeneration, handleFileSelect, downscaleImage, queue operations
 * - ui.js: popover toggles, updateAttachmentPopover, updateAttachmentCount, hapticFeedback
 *
 * DEPENDENTS:
 * - core.js calls setupEventListeners and setupDelegatedEvents at startup
 *
 * LOAD ORDER:
 * Last of the chat modules (after core.js, ui.js, messaging.js, history.js).
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
