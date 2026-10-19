/**
 * TOOL-CONFIG.JS - Tool Configuration Management
 *
 * PURPOSE:
 * Manages user-specific tool configurations through the /actions/tools API.
 * Auto-generates forms from JSON Schema for configurable tools.
 *
 * RESPONSIBILITIES:
 * - Listing all configurable tools
 * - Fetching/updating tool configurations
 * - Generating forms from JSON Schema
 * - Rendering the Tools settings page
 *
 * API INTEGRATION (all through MiraAPIClient._httpRequest):
 * - GET /v1/api/actions/tools - List configurable tools
 * - GET /v1/api/actions/tools/{tool} - Get tool config
 * - GET /v1/api/actions/tools/{tool}/schema - Get JSON Schema
 * - PUT /v1/api/actions/tools/{tool} - Update config
 * - POST /v1/api/actions/tools/{tool}/validate - Validate config without saving
 * - DELETE /v1/api/actions/tools/{tool} - Reset to defaults
 *
 * MECHANISMS:
 * - email_tool's four folder fields are deliberately absent from the generated
 *   form: they exist only in the validate response's `discovered` mapping, so
 *   the first Save on that tool runs validation, renders the folder dropdowns
 *   and returns without writing — the user reviews the mapping and saves again.
 *   A save that skipped that step would persist folder guesses nothing checked.
 * - extractFormValues throws naming the field when a number or JSON value will
 *   not convert, and the dialog shows the message in its status slot; a silent
 *   coercion here would save a config the tool then fails on.
 * - _escapeAttribute is required for every value placed inside a quoted
 *   attribute: schema defaults and tool descriptions are server-side data, and
 *   an unescaped quote ends the attribute and injects markup.
 */

class ToolConfigManager {
    constructor(apiClient) {
        this.apiClient = apiClient;
        this.tools = [];
    }

    _toolPath(toolName) {
        return `/v1/api/actions/tools/${encodeURIComponent(toolName)}`;
    }

    /**
     * Fetch list of all configurable tools
     */
    async fetchTools() {
        const data = await this.apiClient._httpRequest('/v1/api/actions/tools');
        this.tools = data.tools;
        return this.tools;
    }

    /**
     * Fetch current config for a specific tool
     */
    async getToolConfig(toolName) {
        return this.apiClient._httpRequest(this._toolPath(toolName));
    }

    /**
     * Fetch JSON Schema for a tool's config
     */
    async getToolSchema(toolName) {
        const data = await this.apiClient._httpRequest(`${this._toolPath(toolName)}/schema`);
        return data.schema;
    }

    /**
     * Update config for a tool
     */
    async updateToolConfig(toolName, config) {
        return this.apiClient._httpRequest(this._toolPath(toolName), {
            method: 'PUT',
            body: JSON.stringify({ config })
        });
    }

    /**
     * Validate config for a tool without saving. The response carries
     * live_validation (whether the tool's own connection test ran),
     * message, and discovered (whatever the live test returned).
     */
    async validateToolConfig(toolName, config) {
        return this.apiClient._httpRequest(`${this._toolPath(toolName)}/validate`, {
            method: 'POST',
            body: JSON.stringify({ config })
        });
    }

    /**
     * Reset tool config to defaults
     */
    async resetToolConfig(toolName) {
        return this.apiClient._httpRequest(this._toolPath(toolName), {
            method: 'DELETE'
        });
    }

    /**
     * Generate form HTML from JSON Schema
     * @param {string} toolName - Tool name for special handling
     */
    generateFormFromSchema(schema, currentConfig, toolName) {
        if (!schema.properties) {
            return '<p>No configuration options available.</p>';
        }

        // email_tool folder fields are chosen in the folder discovery section instead.
        const hiddenFields = toolName === 'email_tool'
            ? ['inbox_folder', 'sent_folder', 'drafts_folder', 'trash_folder']
            : [];

        let formHtml = '<div class="tool-config-form">';
        const properties = schema.properties;
        const required = schema.required || [];

        for (const [fieldName, fieldSchema] of Object.entries(properties)) {
            if (hiddenFields.includes(fieldName)) continue;

            const value = currentConfig[fieldName];
            const description = fieldSchema.description || '';
            const isRequired = required.includes(fieldName);
            const fieldType = fieldSchema.type;

            const fieldId = `field-${this._escapeAttribute(fieldName)}`;
            const fieldNameAttr = this._escapeAttribute(fieldName);
            const fieldLabel = this._escapeHtml(this._formatFieldName(fieldName));

            formHtml += `<div class="form-group">`;
            formHtml += `<label for="${fieldId}">${fieldLabel}${isRequired ? ' *' : ''}</label>`;

            if (fieldType === 'boolean') {
                const checked = value ? 'checked' : '';
                formHtml += `<input type="checkbox" id="${fieldId}" name="${fieldNameAttr}" ${checked}>`;
            } else if (fieldType === 'integer' || fieldType === 'number') {
                const min = fieldSchema.minimum !== undefined ? `min="${this._escapeAttribute(fieldSchema.minimum)}"` : '';
                const max = fieldSchema.maximum !== undefined ? `max="${this._escapeAttribute(fieldSchema.maximum)}"` : '';
                formHtml += `<input type="number" id="${fieldId}" name="${fieldNameAttr}" value="${this._escapeAttribute(value ?? '')}" ${min} ${max}>`;
            } else if (fieldSchema.enum) {
                formHtml += `<select id="${fieldId}" name="${fieldNameAttr}">`;
                for (const option of fieldSchema.enum) {
                    const selected = value === option ? 'selected' : '';
                    formHtml += `<option value="${this._escapeAttribute(option)}" ${selected}>${this._escapeHtml(option)}</option>`;
                }
                formHtml += '</select>';
            } else if (fieldType === 'object') {
                const jsonValue = JSON.stringify(value || {}, null, 2);
                formHtml += `<textarea id="${fieldId}" name="${fieldNameAttr}" rows="4">${this._escapeHtml(jsonValue)}</textarea>`;
            } else {
                // String type - check for password/secret fields
                const inputType = (fieldName.toLowerCase().includes('password') || fieldName.toLowerCase().includes('secret'))
                    ? 'password' : 'text';
                formHtml += `<input type="${inputType}" id="${fieldId}" name="${fieldNameAttr}" value="${this._escapeAttribute(value ?? '')}">`;
            }

            if (description) {
                formHtml += `<span class="field-description">${this._escapeHtml(description)}</span>`;
            }

            formHtml += '</div>';
        }

        // Folder discovery section for email_tool (its validate_config lists IMAP folders)
        if (toolName === 'email_tool') {
            formHtml += `
                <div class="folder-discovery-section">
                    <div class="folder-header">
                        <h4>Email Folders</h4>
                        <button type="button" class="btn-test-connection">Test Connection & Discover Folders</button>
                    </div>
                    <div id="discovered-folders" class="discovered-folders">
                        <p class="folder-hint">Click "Test Connection" to discover available folders</p>
                    </div>
                </div>
            `;
        }

        formHtml += '</div>';
        return formHtml;
    }

    /**
     * Render the validate endpoint's discovered folders as selection dropdowns.
     * `discovered` is email_tool.validate_config's return:
     * {folders: [{name, flags}], discovered_folders: {inbox_folder, sent_folder, drafts_folder, trash_folder}}.
     */
    renderDiscoveredFolders(discovered, container) {
        const folders = discovered.folders;
        const mapping = discovered.discovered_folders;

        const folderSelect = (name, label, currentValue, allowEmpty) => {
            const emptyOption = allowEmpty ? '<option value="">(not set)</option>' : '';
            const options = folders.map(f => {
                const selected = f.name === currentValue ? 'selected' : '';
                return `<option value="${this._escapeAttribute(f.name)}" ${selected}>${this._escapeHtml(f.name)}</option>`;
            }).join('');

            return `
                <div class="folder-select-group">
                    <label for="folder-${this._escapeAttribute(name)}">${this._escapeHtml(label)}</label>
                    <select id="folder-${this._escapeAttribute(name)}" name="${this._escapeAttribute(name)}" class="folder-select">
                        ${emptyOption}
                        ${options}
                    </select>
                    ${!currentValue && !allowEmpty ? '<span class="folder-warning">⚠ Not detected</span>' : ''}
                </div>
            `;
        };

        let html = '<div class="folder-config">';
        html += `<p class="folder-success">Found ${folders.length} folders.</p>`;
        html += '<div class="folder-selects">';
        html += folderSelect('inbox_folder', 'Inbox', mapping.inbox_folder, false);
        html += folderSelect('sent_folder', 'Sent', mapping.sent_folder, true);
        html += folderSelect('drafts_folder', 'Drafts', mapping.drafts_folder, true);
        html += folderSelect('trash_folder', 'Trash', mapping.trash_folder, true);
        html += '</div>';
        html += '</div>';

        container.innerHTML = html;
    }

    /**
     * Extract form values from the generated form. Throws on a value that
     * cannot be converted to its schema type, naming the field.
     */
    extractFormValues(formContainer, schema) {
        const config = {};
        const properties = schema.properties || {};

        for (const [fieldName, fieldSchema] of Object.entries(properties)) {
            const input = formContainer.querySelector(`[name="${fieldName}"]`);
            if (!input) continue;

            const fieldType = fieldSchema.type;
            const label = this._formatFieldName(fieldName);

            if (fieldType === 'boolean') {
                config[fieldName] = input.checked;
            } else if (fieldType === 'integer' || fieldType === 'number') {
                const parsed = fieldType === 'integer' ? parseInt(input.value, 10) : parseFloat(input.value);
                if (Number.isNaN(parsed)) {
                    throw new Error(`${label} must be a number`);
                }
                config[fieldName] = parsed;
            } else if (fieldType === 'object') {
                try {
                    config[fieldName] = JSON.parse(input.value);
                } catch (e) {
                    throw new Error(`${label} must be valid JSON: ${e.message}`);
                }
            } else {
                config[fieldName] = input.value;
            }
        }

        return config;
    }

    /**
     * Render the tools list page
     */
    async renderToolsPage(container) {
        container.innerHTML = '<p class="loading">Loading tools...</p>';

        try {
            await this.fetchTools();
        } catch (error) {
            console.error('[ToolConfig] Failed to fetch tools:', error);
            container.innerHTML = `<p style="color: #f66;">Failed to load tools: ${this._escapeHtml(error.message)}</p>`;
            return;
        }

        if (this.tools.length === 0) {
            container.innerHTML = '<p>No configurable tools found.</p>';
            return;
        }

        let html = '<div class="tools-grid">';
        for (const tool of this.tools) {
            const statusClass = tool.has_user_config ? 'configured' : 'not-configured';
            const statusText = tool.has_user_config ? 'Configured' : 'Not configured';
            html += `
                <div class="tool-card" data-tool="${this._escapeAttribute(tool.name)}">
                    <div class="tool-header">
                        <h3>${this._escapeHtml(this._formatFieldName(tool.name))}</h3>
                        <span class="tool-status ${statusClass}">${statusText}</span>
                    </div>
                    <p class="tool-config-class">${this._escapeHtml(tool.config_class)}</p>
                    <button class="configure-btn" data-tool="${this._escapeAttribute(tool.name)}">Configure</button>
                </div>
            `;
        }
        html += '</div>';

        container.innerHTML = html;

        container.querySelectorAll('.configure-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.showConfigDialog(btn.dataset.tool, container);
            });
        });
    }

    /**
     * Show configuration dialog for a tool
     */
    async showConfigDialog(toolName, parentContainer) {
        // email_tool folder mapping from the last live validation; saving waits for it.
        let discoveredFolders = null;

        let schema;
        let configData;
        try {
            [schema, configData] = await Promise.all([
                this.getToolSchema(toolName),
                this.getToolConfig(toolName)
            ]);
        } catch (error) {
            console.error(`[ToolConfig] Failed to load ${toolName}:`, error);
            alert(`Failed to load tool configuration: ${error.message}`);
            return;
        }

        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.style.display = 'flex';
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog--form">
                <div class="modal-header">
                    <h2>Configure ${this._escapeHtml(this._formatFieldName(toolName))}</h2>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div id="tool-config-status"></div>
                    ${this.generateFormFromSchema(schema, configData.config, toolName)}
                </div>
                <div class="modal-actions">
                    <button class="btn btn-ghost btn-reset">Reset to Defaults</button>
                    <button class="btn btn-ghost btn-cancel">Cancel</button>
                    <button class="btn btn-primary btn-save">Save Configuration</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        const statusEl = modal.querySelector('#tool-config-status');

        const showStatus = (message, type) => {
            statusEl.textContent = message;
            statusEl.className = `status-${type}`;
            statusEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };

        // Validate, then show exactly what the server reports: its message, and
        // the discovered folders when the tool's live check ran and returned them.
        const runValidation = async (config) => {
            const result = await this.validateToolConfig(toolName, config);
            showStatus(result.message, result.live_validation ? 'success' : 'info');
            const foldersContainer = modal.querySelector('#discovered-folders');
            if (result.live_validation && result.discovered.folders) {
                discoveredFolders = result.discovered.discovered_folders;
                this.renderDiscoveredFolders(result.discovered, foldersContainer);
            } else {
                foldersContainer.innerHTML = `<p class="folder-error">${this._escapeHtml(result.message)}</p>`;
            }
            return result;
        };

        // email_tool: test connection and discover folders
        const testBtn = modal.querySelector('.btn-test-connection');
        if (testBtn) {
            testBtn.addEventListener('click', async () => {
                const foldersContainer = modal.querySelector('#discovered-folders');
                try {
                    testBtn.disabled = true;
                    testBtn.textContent = 'Testing...';
                    foldersContainer.innerHTML = '<p class="folder-hint">Connecting to server...</p>';

                    const config = this.extractFormValues(modal.querySelector('.tool-config-form'), schema);
                    await runValidation(config);
                } catch (error) {
                    foldersContainer.innerHTML = `<p class="folder-error">Connection failed: ${this._escapeHtml(error.message)}</p>`;
                    showStatus(error.message, 'error');
                } finally {
                    testBtn.disabled = false;
                    testBtn.textContent = 'Test Connection & Discover Folders';
                }
            });
        }

        modal.querySelector('.modal-close').addEventListener('click', () => modal.remove());
        modal.querySelector('.btn-cancel').addEventListener('click', () => modal.remove());

        modal.querySelector('.btn-save').addEventListener('click', async () => {
            try {
                const config = this.extractFormValues(modal.querySelector('.tool-config-form'), schema);

                if (toolName === 'email_tool') {
                    // Discover folders first so the user reviews the mapping before it is saved.
                    if (!discoveredFolders) {
                        showStatus('Testing connection and discovering folders...', 'info');
                        try {
                            await runValidation(config);
                        } catch (validationError) {
                            showStatus(`Connection failed: ${validationError.message}`, 'error');
                            return;
                        }
                        if (discoveredFolders) {
                            showStatus('Folders discovered. Review selections and click Save again.', 'info');
                        }
                        return;
                    }

                    modal.querySelectorAll('.folder-select').forEach(select => {
                        if (select.value) {
                            config[select.name] = select.value;
                        }
                    });
                }

                showStatus('Saving...', 'info');

                const result = await this.updateToolConfig(toolName, config);
                showStatus(result.message, 'success');

                setTimeout(() => {
                    modal.remove();
                    this.renderToolsPage(parentContainer);
                }, 1000);
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            }
        });

        modal.querySelector('.btn-reset').addEventListener('click', async () => {
            if (!confirm('Reset all settings to defaults?')) return;

            try {
                showStatus('Resetting...', 'info');

                const result = await this.resetToolConfig(toolName);
                showStatus(result.message, 'success');

                setTimeout(() => {
                    modal.remove();
                    this.renderToolsPage(parentContainer);
                }, 1000);
            } catch (error) {
                showStatus(`Error: ${error.message}`, 'error');
            }
        });

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    /**
     * Format field name for display (snake_case to Title Case)
     */
    _formatFieldName(name) {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Escape HTML for safe rendering
     */
    _escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Escape values before placing them inside quoted HTML attributes.
     */
    _escapeAttribute(value) {
        return this._escapeHtml(String(value))
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }
}

window.ToolConfigManager = ToolConfigManager;
