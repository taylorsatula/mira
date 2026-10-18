/**
 * DOMAIN-KNOWLEDGE.JS - Domaindoc client and chat-page domain popover
 *
 * PURPOSE:
 * Domaindocs are user-defined documents MIRA loads into context while enabled.
 * DomainKnowledgeManager wraps the `domain_knowledge` action domain
 * (POST /v1/api/actions) and the `domaindocs` data type
 * (GET /v1/api/data?type=domaindocs). Any number of domaindocs can be enabled.
 *
 * CONSUMERS:
 * - Chat page: core.js calls initDomainKnowledge(apiClient), which wires the
 *   domain indicator button and its enable/disable popover, and publishes
 *   window.domainManager (messaging.js refreshes it after domaindoc tool use).
 * - Domaindocs page: its DomaindocsCompanion constructs a DomainKnowledgeManager
 *   for document and section CRUD.
 *
 * Section actions take an optional `parent` (the parent section's header) to
 * address subsections; the server resolves it at any depth.
 *
 * LOAD ORDER:
 * After api-client.js.
 */

class DomainKnowledgeManager {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.domains = [];
        this.enabledDomains = new Set();
    }

    _action(action, data) {
        return this.apiClient.actions.executeAction('domain_knowledge', action, data);
    }

    _sectionParams(label, section, parent) {
        const params = { label, section };
        if (parent) params.parent = parent;
        return params;
    }

    /**
     * Fetch all non-archived domaindocs (own and accepted shares)
     */
    async fetchDomains() {
        const response = await this.apiClient.data.getData('domaindocs');
        this.domains = response.domaindocs;
        this.enabledDomains = new Set(
            this.domains.filter(d => d.enabled).map(d => d.label)
        );
        this.updateIndicatorStatus();
        return this.domains;
    }

    async createDomain(label, description) {
        const response = await this._action('create', { label, description });
        await this.fetchDomains();
        return response;
    }

    async enableDomain(label) {
        const response = await this._action('enable', { label });
        await this.fetchDomains();
        return response;
    }

    async disableDomain(label) {
        const response = await this._action('disable', { label });
        await this.fetchDomains();
        return response;
    }

    async deleteDomain(label) {
        const response = await this._action('delete', { label });
        await this.fetchDomains();
        return response;
    }

    async archiveDomain(label) {
        const response = await this._action('archive', { label });
        await this.fetchDomains();
        return response;
    }

    async unarchiveDomain(label) {
        const response = await this._action('unarchive', { label });
        await this.fetchDomains();
        return response;
    }

    // =========================================================================
    // Section Management
    // =========================================================================

    /**
     * Get a domaindoc with all its sections (flat list; parent_section_id links subsections)
     */
    getDomainWithSections(label) {
        return this._action('get', { label });
    }

    updateSection(label, section, content, parent = null) {
        return this._action('update_section', { ...this._sectionParams(label, section, parent), content });
    }

    createSection(label, section, content, parent = null) {
        return this._action('create_section', { ...this._sectionParams(label, section, parent), content });
    }

    renameSection(label, section, newName, parent = null) {
        return this._action('rename_section', { ...this._sectionParams(label, section, parent), new_name: newName });
    }

    deleteSection(label, section, parent = null) {
        return this._action('delete_section', this._sectionParams(label, section, parent));
    }

    /**
     * Reorder the sections at one level; `order` is the full list of sibling headers
     */
    reorderSections(label, order, parent = null) {
        const params = { label, order };
        if (parent) params.parent = parent;
        return this._action('reorder_sections', params);
    }

    expandSection(label, section, parent = null) {
        return this._action('expand_section', this._sectionParams(label, section, parent));
    }

    collapseSection(label, section, parent = null) {
        return this._action('collapse_section', this._sectionParams(label, section, parent));
    }

    /**
     * Reflect enabled-domain state on the chat page's domain indicator button
     */
    updateIndicatorStatus() {
        const domainBtn = document.querySelector('[data-indicator="domain_btn"]');
        if (!domainBtn) return;

        if (this.enabledDomains.size > 0) {
            domainBtn.classList.remove('standby');
            domainBtn.classList.add('active');
        } else {
            domainBtn.classList.remove('active');
            domainBtn.classList.add('standby');
        }
    }

    /**
     * Render domain list in popover (chat interface)
     */
    renderDomainPopover(containerElement) {
        containerElement.innerHTML = '';

        if (this.domains.length === 0) {
            containerElement.innerHTML = '<div class="domain-item empty">No domains created yet</div>';
            return;
        }

        this.domains.forEach(domain => {
            const domainItem = document.createElement('div');
            domainItem.className = 'domain-item';
            if (domain.enabled) {
                domainItem.classList.add('enabled');
            }

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = Boolean(domain.enabled);

            checkbox.addEventListener('change', async () => {
                try {
                    if (checkbox.checked) {
                        await this.enableDomain(domain.label);
                    } else {
                        await this.disableDomain(domain.label);
                    }
                    this.renderDomainPopover(containerElement);
                } catch (error) {
                    alert(`Failed to ${checkbox.checked ? 'enable' : 'disable'} domain: ${error.message}`);
                    checkbox.checked = !checkbox.checked;
                }
            });

            const labelEl = document.createElement('label');
            labelEl.textContent = domain.label;
            if (domain.shared) {
                const sharedTag = document.createElement('span');
                sharedTag.textContent = ` (shared by ${domain.shared_by})`;
                sharedTag.style.cssText = 'color: #0af; font-size: 11px;';
                labelEl.appendChild(sharedTag);
            }

            domainItem.appendChild(checkbox);
            domainItem.appendChild(labelEl);
            containerElement.appendChild(domainItem);
        });

        // Show "View Archived" link only when archived docs exist
        this.apiClient.data.getData('domaindocs', { archived: true }).then(data => {
            const archivedCount = data.domaindocs.length;
            if (archivedCount === 0) return;

            const archivedLink = document.createElement('a');
            archivedLink.href = '/domaindocs?archived=true';
            archivedLink.textContent = `View Archived (${archivedCount})`;
            archivedLink.style.cssText = `
                display: block;
                text-align: center;
                padding: 8px;
                margin-top: 8px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                color: var(--text-inline-muted);
                font-size: 12px;
                text-decoration: none;
            `;
            archivedLink.addEventListener('mouseenter', () => { archivedLink.style.color = '#ffa500'; });
            archivedLink.addEventListener('mouseleave', () => { archivedLink.style.color = '#888'; });
            containerElement.appendChild(archivedLink);
        }).catch(error => {
            console.error('Failed to count archived domains; the "View Archived" link is not shown:', error);
        });
    }
}

/**
 * Initialize the chat page's domain indicator and popover.
 * Called from core.js after apiClient is ready.
 */
function initDomainKnowledge(apiClient) {
    const domainManager = new DomainKnowledgeManager(apiClient);
    window.domainManager = domainManager;

    const domainBtn = document.querySelector('[data-indicator="domain_btn"]');
    const domainPopover = document.getElementById('domain-popover');
    const domainList = document.getElementById('domain-list');
    const domainPopoverClose = domainPopover?.querySelector('.popover-close');

    // Fetch domains on init so the indicator shows whether any domain is enabled
    domainManager.fetchDomains().catch(error => {
        console.error('Failed to fetch initial domain state; domain indicator may be stale:', error);
    });

    if (domainBtn && domainPopover) {
        domainBtn.addEventListener('click', async () => {
            domainPopover.classList.toggle('active');
            if (domainPopover.classList.contains('active')) {
                try {
                    await domainManager.fetchDomains();
                    domainManager.renderDomainPopover(domainList);
                } catch (error) {
                    console.error('Failed to load domains:', error);
                    domainList.innerHTML = '<div class="domain-item error">Failed to load domains</div>';
                }
            }
        });

        if (domainPopoverClose) {
            domainPopoverClose.addEventListener('click', () => {
                domainPopover.classList.remove('active');
            });
        }

        // Close popover on click outside
        document.addEventListener('click', (e) => {
            if (domainPopover.classList.contains('active') &&
                !domainPopover.contains(e.target) &&
                !domainBtn.contains(e.target)) {
                domainPopover.classList.remove('active');
            }
        });
    }
}

window.DomainKnowledgeManager = DomainKnowledgeManager;
window.initDomainKnowledge = initDomainKnowledge;
