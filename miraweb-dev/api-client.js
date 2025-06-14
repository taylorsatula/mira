// MIRA API Client
// Handles communication with the MIRA backend API

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api';

// API client class
class MiraAPI {
    constructor() {
        this.baseURL = API_BASE_URL;
        this.conversationId = null;
    }

    async sendMessage(message, useStreaming = true) {
        const endpoint = useStreaming ? '/chat/stream' : '/chat';
        const payload = {
            message: message,
            conversation_id: this.conversationId
        };

        if (useStreaming) {
            return this.streamChat(payload);
        } else {
            return this.sendChat(payload);
        }
    }

    async sendChat(payload) {
        try {
            const response = await fetch(`${this.baseURL}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.conversationId = data.conversation_id;
            return data.response;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async *streamChat(payload) {
        try {
            const response = await fetch(`${this.baseURL}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.text) {
                                    yield data.text;
                                } else if (data.conversation_id) {
                                    this.conversationId = data.conversation_id;
                                } else if (data.error) {
                                    throw new Error(data.error);
                                }
                            } catch (e) {
                                if (line.slice(6) !== '[DONE]') {
                                    console.warn('Failed to parse SSE data:', line);
                                }
                            }
                        }
                    }
                }
            } finally {
                reader.releaseLock();
            }
        } catch (error) {
            console.error('Streaming request failed:', error);
            throw error;
        }
    }

    async getStatus() {
        try {
            const response = await fetch(`${this.baseURL}/status`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Status request failed:', error);
            throw error;
        }
    }
}

// Global API instance
const miraAPI = new MiraAPI();

// Settings management
function updateAPIEndpoint() {
    const endpointInput = document.getElementById('api-endpoint');
    if (endpointInput) {
        miraAPI.baseURL = endpointInput.value;
        localStorage.setItem('mira-api-endpoint', endpointInput.value);
    }
}

function loadSettings() {
    const savedEndpoint = localStorage.getItem('mira-api-endpoint');
    const savedStreaming = localStorage.getItem('mira-streaming-enabled') === 'true';
    
    const endpointInput = document.getElementById('api-endpoint');
    const streamingCheckbox = document.getElementById('streaming-enabled');
    
    if (savedEndpoint && endpointInput) {
        endpointInput.value = savedEndpoint;
        miraAPI.baseURL = savedEndpoint;
    }
    
    if (streamingCheckbox) {
        streamingCheckbox.checked = savedStreaming;
    }
}

function saveSettings() {
    const endpointInput = document.getElementById('api-endpoint');
    const streamingCheckbox = document.getElementById('streaming-enabled');
    
    if (endpointInput) {
        localStorage.setItem('mira-api-endpoint', endpointInput.value);
        miraAPI.baseURL = endpointInput.value;
    }
    
    if (streamingCheckbox) {
        localStorage.setItem('mira-streaming-enabled', streamingCheckbox.checked);
    }
}

async function testConnection() {
    const statusSpan = document.getElementById('connection-status');
    const testButton = document.getElementById('test-connection');
    
    if (!statusSpan || !testButton) return;
    
    testButton.disabled = true;
    statusSpan.textContent = 'Testing...';
    statusSpan.className = 'connection-status testing';
    
    try {
        // Update endpoint before testing
        updateAPIEndpoint();
        
        const response = await fetch(`${miraAPI.baseURL}/status`);
        if (response.ok) {
            statusSpan.textContent = 'Connected ✓';
            statusSpan.className = 'connection-status connected';
        } else {
            statusSpan.textContent = `Error: ${response.status}`;
            statusSpan.className = 'connection-status error';
        }
    } catch (error) {
        statusSpan.textContent = 'Connection failed';
        statusSpan.className = 'connection-status error';
        console.error('Connection test failed:', error);
    } finally {
        testButton.disabled = false;
    }
}

// Real message handling function
async function handleSendMessage(message) {
    try {
        // Check if streaming is preferred (default to false)
        const streamingCheckbox = document.getElementById('streaming-enabled');
        const useStreaming = streamingCheckbox ? streamingCheckbox.checked : false;
        
        if (useStreaming) {
            // Prepare response container for streaming
            let responseText = '';
            
            // Handle existing response wooshOut timing
            if (responseActive) {
                setTimeout(() => {
                    elements.responseContainer.classList.remove('active');
                    elements.responseBox.classList.remove('exiting');
                    responseActive = false;
                    startStreamingResponse();
                }, 100);
            } else {
                startStreamingResponse();
            }
            
            async function startStreamingResponse() {
                showLoadingScreen(async () => {
                    // Clear content and show container
                    elements.responseContent.textContent = '';
                    elements.responseContainer.classList.add('active');
                    responseActive = true;
                    
                    try {
                        for await (const chunk of miraAPI.sendMessage(message, true)) {
                            responseText += chunk;
                            elements.responseContent.textContent = responseText;
                        }
                    } catch (error) {
                        console.error('Streaming failed:', error);
                        window.miraQueue.add(message);
                        elements.responseContent.textContent = 'Connection error. Message saved to queue.';
                        return;
                    }
                });
            }
        } else {
            // Handle single response with existing flow
            if (responseActive) {
                setTimeout(() => {
                    elements.responseContainer.classList.remove('active');
                    elements.responseBox.classList.remove('exiting');
                    responseActive = false;
                    showLoadingScreen(() => handleSingleResponse());
                }, 100);
            } else {
                showLoadingScreen(() => handleSingleResponse());
            }
            
            async function handleSingleResponse() {
                try {
                    const response = await miraAPI.sendMessage(message, false);
                    showResponse(response);
                } catch (error) {
                    console.error('API request failed:', error);
                    window.miraQueue.add(message);
                    showResponse('Connection error. Message saved to queue.');
                }
            }
        }

        // Activate tool badges (keeping existing visual functionality)
        if (typeof activateBadges === 'function') {
            activateBadges();
        }

    } catch (error) {
        console.error('Error in handleSendMessage:', error);
        window.miraQueue.add(message);
        showResponse('An error occurred. Message saved to queue.');
    }
}