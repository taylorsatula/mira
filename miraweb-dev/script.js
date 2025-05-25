// MIRA Interface JavaScript Module
(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        animations: {
            responseDelay: 300,
            toolIndicatorDuration: 2000,
            workflowDelay: 500,
            workflowDuration: 4000,
            ghostTextDelay: 3000,
            ghostTextTypeSpeed: 50
        },
        api: {
            endpoint: '/api/chat' // Update with actual API endpoint
        }
    };

    // State Management
    const state = {
        isTyping: false,
        currentWorkflow: null,
        conversationHistory: []
    };

    // DOM Elements Cache
    const elements = {
        historyDrawer: null,
        historyToggle: null,
        historyContent: null,
        responseContainer: null,
        responseBox: null,
        responseContent: null,
        messageForm: null,
        messageInput: null,
        ghostText: null,
        sendButton: null,
        toolIndicator: null,
        workflowIndicator: null,
        workflowPopover: null,
        workflowSteps: null
    };

    // Sample Data (Remove when connecting to real API)
    const sampleData = {
        history: [
            { user: "What's the weather like?", mira: "I'll check the current weather for you. It's currently 72°F and sunny." },
            { user: "Set a reminder for tomorrow", mira: "I've set a reminder for tomorrow at 9:00 AM." }
        ],
        workflowSteps: [
            { text: "Analyzing request", status: "completed" },
            { text: "Gathering context", status: "completed" },
            { text: "Processing information", status: "active" },
            { text: "Generating response", status: "pending" }
        ],
        ghostPhrases: [
            "Thinking about quantum mechanics...",
            "Planning tomorrow's schedule...",
            "Checking the weather...",
            "Analyzing market trends...",
            "Composing a haiku..."
        ]
    };

    // Initialize the application
    function init() {
        cacheElements();
        bindEvents();
        loadInitialData();
        startGhostText();
    }

    // Cache DOM elements
    function cacheElements() {
        elements.historyDrawer = document.getElementById('historyDrawer');
        elements.historyToggle = document.getElementById('historyToggle');
        elements.historyContent = document.getElementById('historyContent');
        elements.responseContainer = document.getElementById('responseContainer');
        elements.responseBox = document.getElementById('responseBox');
        elements.responseContent = document.getElementById('responseContent');
        elements.messageForm = document.getElementById('messageForm');
        elements.messageInput = document.getElementById('messageInput');
        elements.ghostText = document.getElementById('ghostText');
        elements.sendButton = document.getElementById('sendButton');
        elements.toolIndicator = document.getElementById('toolIndicator');
        elements.workflowIndicator = document.getElementById('workflowIndicator');
        elements.workflowPopover = document.getElementById('workflowPopover');
        elements.workflowSteps = document.getElementById('workflowSteps');
    }

    // Bind event listeners
    function bindEvents() {
        // History drawer toggle
        elements.historyToggle.addEventListener('click', toggleHistoryDrawer);

        // Message form submission
        elements.messageForm.addEventListener('submit', handleFormSubmit);

        // Workflow popover toggle
        elements.workflowIndicator.addEventListener('click', toggleWorkflowPopover);

        // Close popover when clicking outside
        document.addEventListener('click', handleOutsideClick);

        // Handle escape key
        document.addEventListener('keydown', handleKeydown);
    }

    // Load initial data
    function loadInitialData() {
        // Load sample history (replace with API call)
        sampleData.history.forEach(item => {
            addToHistory(item.user, item.mira);
        });
    }

    // Toggle history drawer
    function toggleHistoryDrawer() {
        elements.historyDrawer.classList.toggle('open');
        const isOpen = elements.historyDrawer.classList.contains('open');
        elements.historyToggle.setAttribute('aria-expanded', isOpen);
    }

    // Handle form submission
    function handleFormSubmit(e) {
        e.preventDefault();
        const message = elements.messageInput.value.trim();
        
        if (message && !state.isTyping) {
            sendMessage(message);
            elements.messageInput.value = '';
            elements.ghostText.textContent = '';
        }
    }

    // Send message
    async function sendMessage(message) {
        state.isTyping = true;
        
        // Add user message to history
        addToHistory(message, 'Thinking...');
        
        // Show/update response
        if (elements.responseContainer.classList.contains('active')) {
            animateResponseTransition(() => {
                updateResponse(`Processing: "${message}"`);
            });
        } else {
            updateResponse(`Processing: "${message}"`);
            elements.responseContainer.classList.add('active');
        }
        
        // Simulate activity
        simulateActivity();
        
        try {
            // TODO: Replace with actual API call
            // const response = await fetchResponse(message);
            // updateResponse(response);
            
            // Simulated response
            setTimeout(() => {
                const response = `I received your message: "${message}". This is a simulated response.`;
                updateResponse(response);
                updateHistoryLatestResponse(response);
                state.isTyping = false;
            }, 2000);
        } catch (error) {
            console.error('Error sending message:', error);
            updateResponse('Sorry, I encountered an error processing your request.');
            state.isTyping = false;
        }
    }

    // Animate response transition
    function animateResponseTransition(callback) {
        elements.responseBox.classList.add('exiting');
        setTimeout(() => {
            elements.responseBox.classList.remove('exiting');
            callback();
        }, CONFIG.animations.responseDelay);
    }

    // Update response content
    function updateResponse(content) {
        elements.responseContent.textContent = content;
    }

    // Add to conversation history
    function addToHistory(userMessage, miraResponse) {
        const historyItem = document.createElement('article');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <div class="history-user">You: ${escapeHtml(userMessage)}</div>
            <div class="history-mira">MIRA: ${escapeHtml(miraResponse)}</div>
        `;
        
        elements.historyContent.insertBefore(historyItem, elements.historyContent.firstChild);
        
        // Store in state
        state.conversationHistory.push({ user: userMessage, mira: miraResponse });
    }

    // Update latest history response
    function updateHistoryLatestResponse(response) {
        const latestItem = elements.historyContent.firstChild;
        if (latestItem) {
            const miraDiv = latestItem.querySelector('.history-mira');
            if (miraDiv) {
                miraDiv.textContent = `MIRA: ${response}`;
            }
        }
    }

    // Simulate tool and workflow activity
    function simulateActivity() {
        // Show tool indicator
        showToolActivity();
        
        // Show workflow after delay
        setTimeout(() => {
            showWorkflowActivity();
        }, CONFIG.animations.workflowDelay);
    }

    // Show tool activity
    function showToolActivity() {
        elements.toolIndicator.classList.add('active');
        setTimeout(() => {
            elements.toolIndicator.classList.remove('active');
        }, CONFIG.animations.toolIndicatorDuration);
    }

    // Show workflow activity
    function showWorkflowActivity() {
        elements.workflowIndicator.classList.add('active');
        populateWorkflowSteps();
        
        setTimeout(() => {
            elements.workflowIndicator.classList.remove('active');
            elements.workflowPopover.classList.remove('active');
        }, CONFIG.animations.workflowDuration);
    }

    // Toggle workflow popover
    function toggleWorkflowPopover() {
        if (elements.workflowIndicator.classList.contains('active')) {
            elements.workflowPopover.classList.toggle('active');
        }
    }

    // Populate workflow steps
    function populateWorkflowSteps() {
        elements.workflowSteps.innerHTML = '';
        
        sampleData.workflowSteps.forEach(step => {
            const stepDiv = document.createElement('div');
            stepDiv.className = `workflow-step ${step.status}`;
            stepDiv.textContent = step.text;
            elements.workflowSteps.appendChild(stepDiv);
        });
    }

    // Ghost text animation
    function startGhostText() {
        let currentAnimation = null;
        
        setInterval(() => {
            if (elements.messageInput.value === '' && !state.isTyping && Math.random() > 0.98) {
                if (currentAnimation) {
                    clearInterval(currentAnimation);
                }
                
                const phrase = sampleData.ghostPhrases[Math.floor(Math.random() * sampleData.ghostPhrases.length)];
                animateGhostText(phrase);
            }
        }, CONFIG.animations.ghostTextDelay);
    }

    // Animate ghost text typing
    function animateGhostText(text) {
        let charIndex = 0;
        elements.ghostText.textContent = '';
        
        const typeInterval = setInterval(() => {
            if (charIndex <= text.length) {
                elements.ghostText.textContent = text.substring(0, charIndex);
                charIndex++;
            } else {
                setTimeout(() => {
                    fadeOutGhostText();
                }, 1000);
                clearInterval(typeInterval);
            }
        }, CONFIG.animations.ghostTextTypeSpeed);
        
        return typeInterval;
    }

    // Fade out ghost text
    function fadeOutGhostText() {
        elements.ghostText.style.transition = 'opacity 0.5s';
        elements.ghostText.style.opacity = '0';
        
        setTimeout(() => {
            elements.ghostText.textContent = '';
            elements.ghostText.style.opacity = '';
            elements.ghostText.style.transition = '';
        }, 500);
    }

    // Handle clicks outside popovers
    function handleOutsideClick(e) {
        if (!elements.workflowPopover.contains(e.target) && 
            !elements.workflowIndicator.contains(e.target)) {
            elements.workflowPopover.classList.remove('active');
        }
    }

    // Handle keyboard shortcuts
    function handleKeydown(e) {
        // Escape key closes drawers/popovers
        if (e.key === 'Escape') {
            elements.historyDrawer.classList.remove('open');
            elements.workflowPopover.classList.remove('active');
        }
        
        // Cmd/Ctrl + K focuses input
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            elements.messageInput.focus();
        }
    }

    // Utility: Escape HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // API: Fetch response (placeholder for real implementation)
    async function fetchResponse(message) {
        const response = await fetch(CONFIG.api.endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error('API request failed');
        }
        
        const data = await response.json();
        return data.response;
    }

    // Public API
    window.MiraInterface = {
        init,
        sendMessage,
        getHistory: () => state.conversationHistory
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();