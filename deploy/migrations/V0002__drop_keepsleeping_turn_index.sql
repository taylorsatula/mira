-- store: postgres
-- notice: Removes a database index this version no longer uses. Nothing in your data changes.
DROP INDEX IF EXISTS idx_messages_keepsleeping_turn_id;
