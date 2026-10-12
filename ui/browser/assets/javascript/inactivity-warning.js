/**
 * INACTIVITY-WARNING.JS - Session Status Bar
 *
 * Full-width bar above compose input. Single UI for all session states:
 * - Idle (cyan, pulsing): No active segment — "Type a message to begin"
 * - Hidden: Active segment, user recently active — nothing to show
 * - Warning (orange): Inactive 5+ min — countdown, pause, wrap up
 * - Urgent (red, pulsing): < 10 min remaining — countdown, pause, wrap up
 * - Collapsed (red): Segment timed out — "Session ended"
 * - Paused (lime): Session paused — unpause button
 *
 * Server state comes from the `continuum` actions get_segment_status,
 * pause_session and resume_session (cns/api/actions.py), polled every 30 s.
 * Provides onStatusChange() + togglePause() for the sidebar pause button.
 * The manager exists from script load (sidebar.js subscribes on
 * DOMContentLoaded); init() runs on DOMContentLoaded, after core.js has
 * created window.miraAPI.
 */

class InactivityWarningManager {
    constructor() {
        this.isBarVisible = false;
        this.currentState = null; // 'idle', 'warning', 'urgent', 'collapsed', 'paused'

        this.INACTIVITY_THRESHOLD = 5 * 60 * 1000; // 5 minutes
        this.POLL_INTERVAL = 30 * 1000;

        // Segment state from server
        this.hasActiveSegment = false;
        this.collapseAt = null;
        this.lastActivity = null;
        this.isPaused = false;

        // Subscribers (sidebar pause button)
        this.statusCallbacks = [];

        // DOM elements (created dynamically)
        this.bar = null;
        this.iconEl = null;
        this.textEl = null;
        this.countdownEl = null;
        this.actionsEl = null;
    }

    /**
     * Subscribe to status changes (used by sidebar.js for pause button sync)
     */
    onStatusChange(callback) {
        this.statusCallbacks.push(callback);
    }

    notifyStatusChange() {
        const state = {
            hasActiveSegment: this.hasActiveSegment,
            isPaused: this.isPaused
        };
        this.statusCallbacks.forEach(cb => cb(state));
    }

    async init() {
        this.createBarDOM();

        // Show idle state immediately — don't wait for API
        this.showBar('idle');

        // Fetch real state, which will update the bar
        await this.fetchSegmentStatus();

        setInterval(() => this.fetchSegmentStatus(), this.POLL_INTERVAL);
        setInterval(() => this.updateCountdown(), 1000);
    }

    createBarDOM() {
        const composeBar = document.getElementById('compose_bar');

        this.bar = document.createElement('div');
        this.bar.className = 'session-status-bar';
        this.bar.id = 'session-status-bar';
        this.bar.innerHTML = `
            <div class="session-status-content">
                <span class="session-status-icon"></span>
                <span class="session-status-text"></span>
                <span class="session-status-countdown"></span>
            </div>
            <div class="session-status-actions"></div>
        `;

        composeBar.insertBefore(this.bar, composeBar.firstChild);

        this.iconEl = this.bar.querySelector('.session-status-icon');
        this.textEl = this.bar.querySelector('.session-status-text');
        this.countdownEl = this.bar.querySelector('.session-status-countdown');
        this.actionsEl = this.bar.querySelector('.session-status-actions');
    }

    async fetchSegmentStatus() {
        let status;
        try {
            status = await window.miraAPI.actions.executeAction('continuum', 'get_segment_status');
        } catch (error) {
            console.error('[SessionStatus] Status fetch failed; the bar keeps its last state until the next poll:', error);
            return;
        }

        this.hasActiveSegment = status.has_active_segment;
        this.lastActivity = status.last_activity ? new Date(status.last_activity) : null;
        this.collapseAt = status.collapse_at ? new Date(status.collapse_at) : null;
        this.isPaused = status.is_paused;

        this.evaluateState();
        this.notifyStatusChange();
    }

    /**
     * Called by messaging.js on message send — hide bar immediately, then refresh
     */
    trackActivity(source) {
        if (source === 'message_sent') {
            this.hideBar();
            this.fetchSegmentStatus();
        }
    }

    /**
     * Determine which state to show based on current segment data
     */
    evaluateState() {
        // No active segment → idle
        if (!this.hasActiveSegment) {
            this.showBar('idle');
            return;
        }

        // Paused → show paused state
        if (this.isPaused) {
            this.showBar('paused');
            return;
        }

        // Check if segment already collapsed (timer expired)
        if (this.collapseAt) {
            const diff = this.collapseAt - new Date();
            if (diff <= 0) {
                this.showBar('collapsed');
                return;
            }
            // < 10 min → urgent
            if (diff < 10 * 60 * 1000) {
                this.showBar('urgent');
                return;
            }
        }

        // Active segment — show warning after inactivity threshold
        if (this.lastActivity) {
            const inactiveDuration = Date.now() - this.lastActivity.getTime();
            if (inactiveDuration >= this.INACTIVITY_THRESHOLD) {
                this.showBar('warning');
                return;
            }
        }

        // Active and recently active — hide bar while chatting
        this.hideBar();
    }

    showBar(state) {
        // Skip redundant updates
        if (this.currentState === state && this.isBarVisible) return;

        this.bar.classList.remove('idle', 'warning', 'urgent', 'collapsed', 'paused');
        this.bar.classList.add('active', state);
        this.bar.closest('#compose_bar').classList.add('has-status-bar');
        this.isBarVisible = true;
        this.currentState = state;

        switch (state) {
            case 'idle':
                this.iconEl.textContent = '○';
                this.textEl.textContent = 'Type a message to begin a new session';
                this.countdownEl.textContent = '';
                this.actionsEl.innerHTML = '';
                break;

            case 'warning':
            case 'urgent':
                this.iconEl.textContent = '⏱';
                this.textEl.textContent = 'Session expires in';
                this.updateCountdown();
                this.renderWarningActions();
                break;

            case 'collapsed':
                this.iconEl.textContent = '●';
                this.textEl.textContent = 'Session ended — type to start a new one';
                this.countdownEl.textContent = '';
                this.actionsEl.innerHTML = '';
                break;

            case 'paused':
                this.iconEl.textContent = '⏸';
                this.textEl.textContent = 'Session paused — resumes on next message';
                this.countdownEl.textContent = '';
                this.actionsEl.innerHTML = `
                    <button class="session-action-btn" id="session-unpause-btn">Unpause</button>
                `;
                this.bar.querySelector('#session-unpause-btn').addEventListener('click', () => this.togglePause());
                break;
        }
    }

    renderWarningActions() {
        this.actionsEl.innerHTML = `
            <button class="session-action-btn" id="session-pause-btn">Pause</button>
            <button class="session-action-btn" id="session-wrapup-btn">Wrap Up</button>
        `;
        this.bar.querySelector('#session-pause-btn').addEventListener('click', () => this.togglePause());
        this.bar.querySelector('#session-wrapup-btn').addEventListener('click', () => {
            this.hideBar();
            window.Functions.collapseSegment();
        });
    }

    hideBar() {
        this.bar.classList.remove('active', 'idle', 'warning', 'urgent', 'collapsed', 'paused');
        this.bar.closest('#compose_bar').classList.remove('has-status-bar');
        this.isBarVisible = false;
        this.currentState = null;
    }

    async togglePause() {
        const action = this.isPaused ? 'resume_session' : 'pause_session';
        try {
            await window.miraAPI.actions.executeAction('continuum', action);
            window.hapticFeedback(50);
        } catch (error) {
            // Another tab may have changed the state; the refresh below shows the real one.
            console.error(`[SessionStatus] ${action} failed:`, error);
        }
        await this.fetchSegmentStatus();
    }

    updateCountdown() {
        if (!this.countdownEl || !this.collapseAt) return;
        if (this.currentState !== 'warning' && this.currentState !== 'urgent') return;

        const diff = this.collapseAt - new Date();

        if (diff <= 0) {
            this.showBar('collapsed');
            return;
        }

        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);

        let timeStr;
        if (hours > 0) {
            timeStr = `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        } else {
            timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }

        this.countdownEl.textContent = timeStr;

        // Auto-escalate warning → urgent at 10 min
        if (diff < 10 * 60 * 1000 && this.currentState === 'warning') {
            this.showBar('urgent');
        }
    }
}

window.inactivityWarningManager = new InactivityWarningManager();

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => window.inactivityWarningManager.init());
} else {
    window.inactivityWarningManager.init();
}
