-- store: postgres
-- notice: Adds the table MIRA uses to know which users have scheduled briefs. Nothing in your data changes.
CREATE TABLE IF NOT EXISTS heartbeat_armed_users (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    earliest_wake_at TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_heartbeat_armed_wake ON heartbeat_armed_users(earliest_wake_at);
