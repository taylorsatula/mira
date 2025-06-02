// Functional JavaScript for MIRA Interface

// State
let historyOpen = false;
let responseActive = false;
let currentScope = 'today';
let theme = localStorage.getItem('mira-theme') || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
let currentCalendarDate = new Date();

// Elements
const elements = {
    // Loading
    loadingScreen: document.getElementById('loading-screen'),
    asciiContainer: document.getElementById('ascii-container'),
    
    // History
    historyDrawer: document.getElementById('history-drawer'),
    historyToggle: document.getElementById('history-toggle'),
    historyContent: document.getElementById('history-content'),
    historySearch: document.getElementById('history-search'),
    datePicker: document.getElementById('date-picker'),
    calendarPopup: document.getElementById('calendar-popup'),
    
    // Response
    responseContainer: document.getElementById('response-container'),
    responseBox: document.getElementById('response-box'),
    responseContent: document.getElementById('response-content'),
    
    // Input
    inputSection: document.querySelector('.input-section'),
    inputContainer: document.querySelector('.input-container'),
    messageInput: document.getElementById('message-input'),
    ghostText: document.getElementById('ghost-text'),
    sendButton: document.getElementById('send-button'),
    toolBadge: document.getElementById('tool-badge'),
    workflowBadge: document.getElementById('workflow-badge'),
    workflowPopover: document.getElementById('workflow-popover'),
    workflowSteps: document.getElementById('workflow-steps'),
    queueIndicator: document.getElementById('queue-indicator'),
    queueCount: document.querySelector('.queue-count'),
    queuePopover: document.getElementById('queue-popover'),
    queueMessages: document.getElementById('queue-messages'),
    
    // Menu
    themeToggle: document.getElementById('theme-toggle'),
    settingsButton: document.getElementById('settings-button'),
    settingsModal: document.getElementById('settings-modal')
};

// Theme
function applyTheme(newTheme) {
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('mira-theme', newTheme);
    theme = newTheme;
}

// Event handlers
function toggleHistory() {
    historyOpen = !historyOpen;
    elements.historyDrawer.classList.toggle('open');
    elements.historyToggle.querySelector('img').src = historyOpen ? 'images/icons/read-less.png' : 'images/icons/read-more.png';
}

function toggleTheme() {
    applyTheme(theme === 'dark' ? 'light' : 'dark');
}

function toggleCalendar() {
    const isActive = elements.calendarPopup.classList.contains('active');
    if (!isActive) renderCalendar();
    elements.calendarPopup.classList.toggle('active');
}

function isMobile() {
    // Primary: Small screen size (most reliable)
    const isSmallScreen = window.innerWidth <= 768;
    
    // Secondary: Touch-first devices (excludes touchscreen laptops)
    const isTouchFirst = window.matchMedia('(pointer: coarse)').matches;
    
    // Combine: small screen OR touch-first device
    return isSmallScreen || isTouchFirst;
}

function toggleWorkflowPopover() {
    const isActive = elements.workflowPopover.classList.contains('active');
    
    if (!isActive && isMobile()) {
        // Move to body for fullscreen display
        document.body.appendChild(elements.workflowPopover);
        elements.workflowPopover.classList.add('mobile-fullscreen');
    } else if (!isActive) {
        // Keep in original container for desktop
    }
    
    elements.workflowPopover.classList.toggle('active');
    
    if (!elements.workflowPopover.classList.contains('active')) {
        elements.workflowPopover.classList.remove('mobile-fullscreen');
        // Move back to original container
        document.querySelector('.workflow-container').appendChild(elements.workflowPopover);
    }
}

function toggleQueuePopover() {
    const isActive = elements.queuePopover.classList.contains('active');
    
    if (!isActive) {
        if (isMobile()) {
            // Move to body for fullscreen display
            document.body.appendChild(elements.queuePopover);
            elements.queuePopover.classList.add('mobile-fullscreen');
        }
        renderQueuedMessages();
    }
    
    elements.queuePopover.classList.toggle('active');
    
    if (!elements.queuePopover.classList.contains('active')) {
        elements.queuePopover.classList.remove('mobile-fullscreen');
        // Move back to original container
        document.querySelector('.queue-container').appendChild(elements.queuePopover);
    }
}

function openSettings() {
    elements.settingsModal.classList.remove('hidden');
}

function closeSettings() {
    elements.settingsModal.classList.add('hidden');
}

// Unified click-outside handler
function handleClickOutside(e) {
    // Calendar
    if (!elements.datePicker.contains(e.target) && !elements.calendarPopup.contains(e.target)) {
        elements.calendarPopup.classList.remove('active');
    }
    
    // Workflow popover
    const workflowContainer = document.querySelector('.workflow-container');
    if (workflowContainer && !workflowContainer.contains(e.target)) {
        elements.workflowPopover.classList.remove('active');
        elements.workflowPopover.classList.remove('mobile-fullscreen');
    }
    
    // Queue popover
    const queueContainer = document.querySelector('.queue-container');
    if (queueContainer && !queueContainer.contains(e.target)) {
        elements.queuePopover.classList.remove('active');
        elements.queuePopover.classList.remove('mobile-fullscreen');
    }
    
    // History drawer
    if (historyOpen && !elements.historyDrawer.contains(e.target) && !document.querySelector('.menu-wrapper').contains(e.target)) {
        toggleHistory();
    }
}

// Conversation management
function renderConversations() {
    // Use demo data if available, otherwise show empty state
    if (typeof renderDemoConversations === 'function') {
        renderDemoConversations();
    } else {
        elements.historyContent.innerHTML = '<div class="empty-state">No conversations found</div>';
    }
}

function switchTab(scope) {
    document.querySelectorAll('.tab-button').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.scope === scope);
    });
    currentScope = scope;
    filterHistory(scope);
    elements.datePicker.querySelector('span').textContent = 'Select date';
}

function filterHistory(scope) {
    const groups = document.querySelectorAll('.date-group');
    const emptyState = document.querySelector('.empty-state');
    let hasVisibleItems = false;
    
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.style.display = '';
    });
    
    groups.forEach(group => {
        const shouldShow = 
            scope === 'all' || 
            (scope === 'today' && group.dataset.date === 'today') ||
            (scope === 'recent' && ['today', 'yesterday'].includes(group.dataset.date));
        
        group.classList.toggle('hidden', !shouldShow);
        if (shouldShow) hasVisibleItems = true;
    });
    
    emptyState.classList.toggle('hidden', hasVisibleItems);
}

function highlightText(text, query) {
    if (!query) return text;
    
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    return text.replace(regex, '<span class="search-highlight">$1</span>');
}

function searchHistory(query) {
    const items = document.querySelectorAll('.conversation-item');
    const groups = document.querySelectorAll('.date-group');
    const emptyState = document.querySelector('.empty-state');
    let hasResults = false;
    
    elements.datePicker.querySelector('span').textContent = 'Select date';
    
    if (!query) {
        // Clear highlights and restore original text
        items.forEach(item => {
            const userEl = item.querySelector('.conversation-user');
            const miraEl = item.querySelector('.conversation-mira');
            userEl.innerHTML = userEl.textContent;
            miraEl.innerHTML = miraEl.textContent;
        });
        filterHistory(currentScope);
        return;
    }
    
    const lowerQuery = query.toLowerCase();
    
    groups.forEach(group => {
        let groupHasMatch = false;
        const groupItems = group.querySelectorAll('.conversation-item');
        
        groupItems.forEach(item => {
            const userEl = item.querySelector('.conversation-user');
            const miraEl = item.querySelector('.conversation-mira');
            const userText = userEl.textContent;
            const miraText = miraEl.textContent;
            const matches = userText.toLowerCase().includes(lowerQuery) || 
                          miraText.toLowerCase().includes(lowerQuery);
            
            item.style.display = matches ? 'block' : 'none';
            if (matches) {
                groupHasMatch = true;
                hasResults = true;
                // Highlight matching text
                userEl.innerHTML = highlightText(userText, query);
                miraEl.innerHTML = highlightText(miraText, query);
            } else {
                // Restore original text for non-matches
                userEl.innerHTML = userText;
                miraEl.innerHTML = miraText;
            }
        });
        
        group.classList.toggle('hidden', !groupHasMatch);
    });
    
    emptyState.classList.toggle('hidden', hasResults);
}

function loadConversation(item) {
    const miraText = item.querySelector('.conversation-mira').textContent;
    toggleHistory();
    showLoadingScreen(() => {
        elements.responseContent.textContent = miraText;
        elements.responseContainer.classList.add('active');
        responseActive = true;
    });
}

// Calendar
function renderCalendar(date = new Date()) {
    currentCalendarDate = date;
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();
    
    let html = `
        <div class="calendar-header">
            <div class="calendar-month">${date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}</div>
            <div class="calendar-nav">
                <button class="calendar-nav-prev">‹</button>
                <button class="calendar-nav-next">›</button>
            </div>
        </div>
        <div class="calendar-grid">
    `;
    
    // Day headers
    ['S', 'M', 'T', 'W', 'T', 'F', 'S'].forEach(day => {
        html += `<div class="calendar-day-header">${day}</div>`;
    });
    
    // Empty cells
    for (let i = 0; i < startingDayOfWeek; i++) {
        html += '<div class="calendar-day disabled"></div>';
    }
    
    // Days
    const today = new Date();
    const hasDataDays = [1, 3, 5, 8, 12, 15, 18, 22, 25, 28];
    
    for (let day = 1; day <= daysInMonth; day++) {
        const cellDate = new Date(year, month, day);
        const isToday = cellDate.toDateString() === today.toDateString();
        const hasData = hasDataDays.includes(day);
        const isFuture = cellDate > today;
        
        let classes = 'calendar-day';
        if (isToday) classes += ' today';
        if (hasData && !isFuture) classes += ' has-data';
        if (isFuture) classes += ' disabled';
        
        html += `<div class="${classes}" ${!isFuture ? `data-date="${year}-${month}-${day}"` : ''}>${day}</div>`;
    }
    
    html += '</div>';
    elements.calendarPopup.innerHTML = html;
}

function selectDate(date) {
    const dateStr = date.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });
    elements.datePicker.querySelector('span').textContent = dateStr;
    elements.calendarPopup.classList.remove('active');
    
    document.querySelectorAll('.tab-button').forEach(tab => tab.classList.remove('active'));
    
    const groups = document.querySelectorAll('.date-group');
    const emptyState = document.querySelector('.empty-state');
    
    groups.forEach(group => group.classList.add('hidden'));
    emptyState.textContent = `No conversations on ${dateStr}`;
    emptyState.classList.remove('hidden');
}

// Message handling
async function sendMessage(messageText) {
    const message = messageText || elements.messageInput.value.trim();
    if (!message) return;
    
    // Clear input if message came from text box
    if (!messageText) {
        elements.messageInput.value = '';
    }
    
    elements.sendButton.disabled = true;
    elements.inputContainer.classList.add('firing');
    
    // If response is active, time the wooshOut to when projectile "hits" 
    if (responseActive) {
        // Projectile launches at ~250ms and travels for ~700ms
        // Trigger slightly early for better visual impact
        setTimeout(() => {
            elements.responseBox.classList.add('exiting');
        }, 750);
    }
    
    await runPlungerAnimation(message);
    
    elements.sendButton.disabled = false;
    elements.inputContainer.classList.remove('firing');
    
    // Let the app implementation handle the actual sending
    if (typeof handleSendMessage === 'function') {
        handleSendMessage(message);
    } else {
        // Default behavior for no backend
        if (responseActive) {
            setTimeout(() => {
                elements.responseContainer.classList.remove('active');
                elements.responseBox.classList.remove('exiting');
                responseActive = false;
                showLoadingScreen(() => showResponse());
            }, 100);
        } else {
            showLoadingScreen(() => showResponse());
        }
    }
}

function showResponse() {
    // Use demo response if available, otherwise show default message
    const response = typeof getDemoResponse === 'function' 
        ? getDemoResponse() 
        : "Hello! I'm MIRA, your AI assistant.";
    
    // Remove active class to hide container
    elements.responseContainer.classList.remove('active');
    elements.responseBox.classList.remove('exiting');
    
    // Wait for container to be fully hidden
    requestAnimationFrame(() => {
        // Update content while invisible
        elements.responseContent.textContent = response;
        
        // Then show it with animation
        requestAnimationFrame(() => {
            elements.responseContainer.classList.add('active');
            responseActive = true;
        });
    });
}

// Touch handling for swipe gestures
let touchStartX = 0;
let touchStartY = 0;
let touchStartTime = 0;
let isSwiping = false;

function handleTouchStart(e) {
    touchStartX = e.touches[0].clientX;
    touchStartY = e.touches[0].clientY;
    touchStartTime = Date.now();
    isSwiping = false;
}

function handleTouchMove(e) {
    if (!touchStartX) return;
    
    const touchX = e.touches[0].clientX;
    const touchY = e.touches[0].clientY;
    const deltaX = touchX - touchStartX;
    const deltaY = touchY - touchStartY;
    
    // Determine if this is a horizontal swipe
    if (!isSwiping && Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 10) {
        isSwiping = true;
    }
    
    if (isSwiping) {
        e.preventDefault(); // Prevent scrolling while swiping
        
        // Swipe from left edge to open
        if (!historyOpen && touchStartX < 30 && deltaX > 50) {
            toggleHistory();
            resetTouch();
        }
        // Swipe left to close
        else if (historyOpen && deltaX < -50) {
            toggleHistory();
            resetTouch();
        }
    }
}

function handleTouchEnd(e) {
    resetTouch();
}

function resetTouch() {
    touchStartX = 0;
    touchStartY = 0;
    touchStartTime = 0;
    isSwiping = false;
}

// Simple Message Queue using localStorage
let messageQueue = JSON.parse(localStorage.getItem('mira-queue') || '[]');
updateQueueIndicator();

function queueMessage(text) {
    messageQueue.push({
        text: text,
        timestamp: Date.now()
    });
    localStorage.setItem('mira-queue', JSON.stringify(messageQueue));
    updateQueueIndicator();
}

function updateQueueIndicator() {
    if (messageQueue.length > 0) {
        elements.queueIndicator.classList.remove('hidden');
        elements.queueCount.textContent = `Retry Send: ${messageQueue.length}`;
    } else {
        elements.queueIndicator.classList.add('hidden');
        elements.queuePopover.classList.remove('active');
    }
}

function renderQueuedMessages() {
    if (messageQueue.length === 0) {
        elements.queueMessages.innerHTML = '<div class="empty-state">No queued messages</div>';
        return;
    }
    
    let html = '';
    messageQueue.forEach((msg, index) => {
        const time = new Date(msg.timestamp).toLocaleString();
        html += `
            <div class="queue-message" data-index="${index}">
                <div class="queue-message-text">${msg.text}</div>
                <div class="queue-message-time">${time}</div>
                <div class="queue-message-actions">
                    <button class="queue-send-btn" data-index="${index}">Send</button>
                    <button class="queue-remove-btn" data-index="${index}">Remove</button>
                </div>
            </div>
        `;
    });
    
    elements.queueMessages.innerHTML = html;
}

async function sendSingleQueuedMessage(index) {
    const msg = messageQueue[index];
    if (!msg) return;
    
    // Set the message in input and send it
    elements.messageInput.value = msg.text;
    messageQueue.splice(index, 1);
    localStorage.setItem('mira-queue', JSON.stringify(messageQueue));
    updateQueueIndicator();
    renderQueuedMessages();
    
    // Send the message
    await sendMessage();
}

function removeQueuedMessage(index) {
    messageQueue.splice(index, 1);
    localStorage.setItem('mira-queue', JSON.stringify(messageQueue));
    updateQueueIndicator();
    renderQueuedMessages();
}

// Public API for handling errors
window.miraQueue = {
    add: queueMessage
};


// Initialize
function initializeMira() {
    applyTheme(theme);
    renderConversations();
    
    // Event delegation
    document.addEventListener('click', handleClickOutside);
    
    elements.historyToggle.addEventListener('click', toggleHistory);
    elements.themeToggle.addEventListener('click', toggleTheme);
    elements.settingsButton.addEventListener('click', openSettings);
    
    elements.messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !elements.sendButton.disabled) sendMessage();
    });
    elements.sendButton.addEventListener('click', () => {
        if (!elements.sendButton.disabled) sendMessage();
    });
    
    elements.historySearch.addEventListener('input', (e) => searchHistory(e.target.value));
    
    elements.datePicker.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleCalendar();
    });
    
    elements.workflowBadge.addEventListener('click', (e) => {
        e.stopPropagation();
        if (elements.workflowBadge.classList.contains('active')) {
            toggleWorkflowPopover();
        }
    });
    
    elements.queueIndicator.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleQueuePopover();
    });
    
    // Tab delegation
    document.querySelector('.tabs').addEventListener('click', (e) => {
        if (e.target.classList.contains('tab-button')) {
            switchTab(e.target.dataset.scope);
        }
    });
    
    // Conversation delegation
    elements.historyContent.addEventListener('click', (e) => {
        const item = e.target.closest('.conversation-item');
        if (item) loadConversation(item);
    });
    
    // Calendar delegation
    elements.calendarPopup.addEventListener('click', (e) => {
        e.stopPropagation();
        
        if (e.target.classList.contains('calendar-nav-prev')) {
            const newDate = new Date(currentCalendarDate);
            newDate.setMonth(newDate.getMonth() - 1);
            renderCalendar(newDate);
        } else if (e.target.classList.contains('calendar-nav-next')) {
            const newDate = new Date(currentCalendarDate);
            newDate.setMonth(newDate.getMonth() + 1);
            renderCalendar(newDate);
        } else if (e.target.classList.contains('calendar-day') && !e.target.classList.contains('disabled')) {
            const dateStr = e.target.dataset.date;
            if (dateStr) {
                const [y, m, d] = dateStr.split('-').map(Number);
                selectDate(new Date(y, m, d));
            }
        }
    });
    
    // Modal event listeners
    elements.settingsModal.addEventListener('click', (e) => {
        if (e.target === elements.settingsModal || e.target.closest('.modal-close')) {
            closeSettings();
        }
    });
    
    // Queue action delegation
    elements.queueMessages.addEventListener('click', async (e) => {
        const index = parseInt(e.target.dataset.index);
        if (e.target.classList.contains('queue-send-btn')) {
            await sendSingleQueuedMessage(index);
        } else if (e.target.classList.contains('queue-remove-btn')) {
            await removeQueuedMessage(index);
        }
    });
    
    // Popover close buttons
    document.querySelectorAll('.popover-close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const popover = btn.closest('.popover');
            popover.classList.remove('active');
            popover.classList.remove('mobile-fullscreen');
        });
    });
    
    // Touch event listeners for swipe gestures
    if ('ontouchstart' in window) {
        document.addEventListener('touchstart', handleTouchStart, { passive: true });
        document.addEventListener('touchmove', handleTouchMove, { passive: false });
        document.addEventListener('touchend', handleTouchEnd, { passive: true });
    }
    
    
}

// Start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeMira);
} else {
    initializeMira();
}