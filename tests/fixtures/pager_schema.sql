-- Pager Tool Database Schema
-- This schema defines the tables required for the pager_tool functionality

-- Pager devices table
CREATE TABLE IF NOT EXISTS pager_devices (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,  -- ISO format UTC timestamp
    last_active TEXT NOT NULL,  -- ISO format UTC timestamp
    active INTEGER DEFAULT 1,
    device_secret TEXT NOT NULL,
    device_fingerprint TEXT NOT NULL
);

-- Index for user queries
CREATE INDEX IF NOT EXISTS idx_pager_devices_user_id ON pager_devices(user_id);
CREATE INDEX IF NOT EXISTS idx_pager_devices_active ON pager_devices(active);

-- Pager messages table
CREATE TABLE IF NOT EXISTS pager_messages (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    sender_id TEXT NOT NULL,
    recipient_id TEXT NOT NULL,
    content TEXT NOT NULL,
    original_content TEXT,  -- Original before AI distillation
    ai_distilled INTEGER DEFAULT 0,
    priority INTEGER DEFAULT 0,  -- 0=normal, 1=high, 2=urgent
    location TEXT,  -- JSON string with location data
    sent_at TEXT NOT NULL,  -- ISO format UTC timestamp
    expires_at TEXT NOT NULL,  -- ISO format UTC timestamp
    read_at TEXT,  -- ISO format UTC timestamp when read
    delivered INTEGER DEFAULT 1,
    read INTEGER DEFAULT 0,
    sender_fingerprint TEXT NOT NULL,
    message_signature TEXT NOT NULL,
    FOREIGN KEY (sender_id) REFERENCES pager_devices(id),
    FOREIGN KEY (recipient_id) REFERENCES pager_devices(id)
);

-- Indexes for message queries
CREATE INDEX IF NOT EXISTS idx_pager_messages_user_id ON pager_messages(user_id);
CREATE INDEX IF NOT EXISTS idx_pager_messages_recipient ON pager_messages(recipient_id);
CREATE INDEX IF NOT EXISTS idx_pager_messages_sender ON pager_messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_pager_messages_expires ON pager_messages(expires_at);
CREATE INDEX IF NOT EXISTS idx_pager_messages_read ON pager_messages(read);

-- Pager trust relationships table
CREATE TABLE IF NOT EXISTS pager_trust (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    trusting_device_id TEXT NOT NULL,
    trusted_device_id TEXT NOT NULL,
    trusted_fingerprint TEXT NOT NULL,
    trusted_name TEXT,
    first_seen TEXT NOT NULL,  -- ISO format UTC timestamp
    last_verified TEXT NOT NULL,  -- ISO format UTC timestamp
    trust_status TEXT DEFAULT 'trusted',  -- 'trusted', 'revoked', 'conflicted'
    FOREIGN KEY (trusting_device_id) REFERENCES pager_devices(id),
    FOREIGN KEY (trusted_device_id) REFERENCES pager_devices(id),
    UNIQUE(user_id, trusting_device_id, trusted_device_id)
);

-- Indexes for trust queries
CREATE INDEX IF NOT EXISTS idx_pager_trust_user_id ON pager_trust(user_id);
CREATE INDEX IF NOT EXISTS idx_pager_trust_trusting ON pager_trust(trusting_device_id);
CREATE INDEX IF NOT EXISTS idx_pager_trust_status ON pager_trust(trust_status);