// Demo data for MIRA interface
const DEMO_DATA = {
    responses: [
        "TIP: You can use the expand icon up there to look at the history menu. (keep sending messages for more tips)",
        "NAVIGATE: Try out the date picker and search bar. (keep sending messages for more tips)",
        "WORKFLOW: Click on the purple workflow indicator. It'll show the active steps for workflows. (keep sending messages for more tips)",
        "MODE SWITCH: You can switch from day mode to dark mode in the pullout drawer. (keep sending messages for more tips)"
    ],
    
    ghostPhrases: [
        "Thinking about quantum mechanics...",
        "Planning tomorrow's schedule...",
        "Checking the weather...",
        "Analyzing market trends...",
        "Reviewing your documents..."
    ],
    
    tools: [
        { name: "Email", icon: "images/icons/toolicons/email.png" },
        { name: "Calendar", icon: "images/icons/toolicons/calendar.png" },
        { name: "Maps", icon: "images/icons/toolicons/maps.png" },
        { name: "Reminders", icon: "images/icons/toolicons/reminders.png" },
        { name: "Web Access", icon: "images/icons/toolicons/webaccess.png" }
    ],
    
    workflows: ["Contacting a customer", "Analyzing data", "Scheduling meeting", "Processing request", "Generating report"],
    
    workflowSteps: [
        { text: "Analyzing request", status: "completed" },
        { text: "Gathering context", status: "completed" },
        { text: "Processing information", status: "active" },
        { text: "Generating response", status: "pending" }
    ],
    
    conversations: {
        today: [
            { time: "10:45 AM", user: "Analyze the sales data from Q3", mira: "Q3 sales show a 23% increase compared to Q2, with strongest growth in the enterprise segment." },
            { time: "9:30 AM", user: "Show me my calendar for today", mira: "You have 4 meetings today, starting with the standup at 9:30am." },
            { time: "8:15 AM", user: "What's the weather forecast for tomorrow?", mira: "I've analyzed the forecast. Tomorrow will be partly cloudy with temperatures around 72°F." }
        ],
        yesterday: [
            { time: "4:20 PM", user: "Draft an email to the marketing team", mira: "I've drafted the email. Would you like me to make any adjustments before sending?" },
            { time: "2:15 PM", user: "Schedule a call with Sarah next week", mira: "I've found several available slots next week. Tuesday at 2pm works best for both calendars." },
            { time: "11:00 AM", user: "Can you help me debug this Python code?", mira: "I found the issue in your code. The error is on line 23 where you're missing a closing bracket." }
        ],
        older: [
            { date: "Monday, May 26", time: "3:00 PM", user: "Remind me about the meeting at 3pm", mira: "I've set a reminder for your 3pm meeting with the product team." },
            { date: "Monday, May 26", time: "10:30 AM", user: "Explain quantum computing in simple terms", mira: "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously..." }
        ]
    }
};

// Demo data functions
function renderDemoConversations() {
    let html = '';
    
    // Today
    html += '<section class="date-group" data-date="today"><h3 class="date-header">Today</h3>';
    DEMO_DATA.conversations.today.forEach(conv => {
        html += `<article class="conversation-item" data-time="${conv.time}">
            <time class="conversation-time">${conv.time}</time>
            <div class="conversation-user">${conv.user}</div>
            <div class="conversation-mira">${conv.mira}</div>
        </article>`;
    });
    html += '</section>';
    
    // Yesterday
    html += '<section class="date-group hidden" data-date="yesterday"><h3 class="date-header">Yesterday</h3>';
    DEMO_DATA.conversations.yesterday.forEach(conv => {
        html += `<article class="conversation-item" data-time="${conv.time}">
            <time class="conversation-time">${conv.time}</time>
            <div class="conversation-user">${conv.user}</div>
            <div class="conversation-mira">${conv.mira}</div>
        </article>`;
    });
    html += '</section>';
    
    // Older
    html += '<section class="date-group hidden" data-date="older"><h3 class="date-header">Monday, May 26</h3>';
    DEMO_DATA.conversations.older.forEach(conv => {
        html += `<article class="conversation-item" data-time="${conv.time}">
            <time class="conversation-time">${conv.time}</time>
            <div class="conversation-user">${conv.user}</div>
            <div class="conversation-mira">${conv.mira}</div>
        </article>`;
    });
    html += '</section>';
    
    html += '<div class="empty-state hidden">No conversations found</div>';
    
    const historyContent = document.getElementById('history-content');
    if (historyContent) {
        historyContent.innerHTML = html;
    }
}

function getDemoResponse() {
    return DEMO_DATA.responses[Math.floor(Math.random() * DEMO_DATA.responses.length)];
}

function getDemoGhostPhrase() {
    return DEMO_DATA.ghostPhrases[Math.floor(Math.random() * DEMO_DATA.ghostPhrases.length)];
}

function simulateDemoActivity() {
    // Access elements from global scope
    if (typeof elements === 'undefined') return;
    
    // Tool badge
    const tool = DEMO_DATA.tools[Math.floor(Math.random() * DEMO_DATA.tools.length)];
    if (elements.toolBadge) {
        // Update the icon
        const img = elements.toolBadge.querySelector('img');
        if (img) {
            img.src = tool.icon;
        }
        elements.toolBadge.classList.add('active');
        addGlow(elements.toolBadge);
        
        setTimeout(() => {
            elements.toolBadge.classList.remove('active');
            // Switch back to default icon when inactive
            if (img) {
                img.src = 'images/icons/toolicons/default.png';
            }
        }, 2000);
    }
    
    // Workflow badge - just add active class, don't change content
    setTimeout(() => {
        if (elements.workflowBadge) {
            elements.workflowBadge.classList.add('active');
            addGlow(elements.workflowBadge);
        }
        
        // Populate workflow steps
        if (elements.workflowSteps) {
            elements.workflowSteps.innerHTML = '';
            DEMO_DATA.workflowSteps.forEach(step => {
                const stepDiv = document.createElement('div');
                stepDiv.className = `workflow-step ${step.status}`;
                stepDiv.textContent = step.text;
                elements.workflowSteps.appendChild(stepDiv);
            });
        }
    }, 500);
}

// Demo implementation of message handling
function handleSendMessage(message) {
    // Simulate network request with 10% failure rate
    const shouldFail = Math.random() < 0.1 || !navigator.onLine;
    
    if (shouldFail) {
        // Queue the message
        window.miraQueue.add(message);
        showResponse('Network error. Message saved to queue.');
        return;
    }
    
    // Success - show loading and response
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
    
    simulateDemoActivity();
}

function startDemoGhostText() {
    let currentAnimation = null;
    const messageInput = document.getElementById('message-input');
    const ghostText = document.getElementById('ghost-text');
    
    if (!messageInput || !ghostText) return;
    
    setInterval(() => {
        if (messageInput.value === "" && Math.random() > 0.98) {
            if (currentAnimation) return;
            
            const phrase = getDemoGhostPhrase();
            currentAnimation = typeDemoGhostText(phrase, ghostText, () => {
                currentAnimation = null;
            });
        }
    }, 3000);
}

function typeDemoGhostText(phrase, ghostTextElement, onComplete) {
    let charIndex = 0;
    const interval = setInterval(() => {
        if (charIndex <= phrase.length) {
            ghostTextElement.textContent = phrase.substring(0, charIndex);
            charIndex++;
        } else {
            setTimeout(() => {
                ghostTextElement.textContent = "";
                onComplete();
            }, 1000);
            clearInterval(interval);
        }
    }, 50);
    return interval;
}

// Initialize demo data when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        renderDemoConversations();
        startDemoGhostText();
    });
} else {
    renderDemoConversations();
    startDemoGhostText();
}