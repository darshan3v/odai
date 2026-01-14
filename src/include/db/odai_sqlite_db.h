#pragma once

#include <string>
#include <memory>
#include <vector>

#include <SQLiteCpp/SQLiteCpp.h>
#include <nlohmann/json.hpp>

#include "types/odai_types.h"
#include "db/odai_db.h"

using namespace std;
using namespace nlohmann;

/// SQLite implementation of ODAIDb interface for managing RAG (Retrieval-Augmented Generation) chat sessions and messages.
/// Provides functionality for initializing SQLite database with vector extensions, managing chat sessions,
/// and storing chat messages with metadata.
class ODAISqliteDb : public ODAIDb
{

private:
    string dbPath;
    unique_ptr<SQLite::Database> db = nullptr;

    /// Registers the sqlite-vec extension and opens the database connection.
    /// The extension is registered before creating the database object to enable vector operations.
    /// @return true if extension registered, false on error
    bool register_vec_extension();

public:
    /// Constructs a new ODAISqliteDb instance with the specified database configuration.
    /// The database is not opened until initialize_db() is called.
    /// @param dbConfig Database configuration object.
    ODAISqliteDb(const DBConfig &dbConfig);

    /// Destructor that cleans up database resources.
    /// ToDo : Not yet implemented
    ~ODAISqliteDb() override;

    /// Initializes the database object (if db doesn't exist then create and initializes with schema)
    /// Registers the sqlite-vec extension, opens the connection, and initializes schema if needed.
    /// @return true if initialization succeeded, false otherwise
    bool initialize_db() override;

    /// Checks if a chat session with the given chat_id exists in the database.
    /// @param chat_id The chat identifier to check
    /// @return true if chat_id exists, false if not found or on error
    bool chat_id_exists(const ChatId &chat_id) override;

    /// Insert a new chat in DB with the specified configuration.
    /// Stores the chat configuration as JSONB in the database and inserts the system prompt as the initial message.
    /// @param chat_id Unique identifier for the chat session
    /// @param chat_config Configuration object containing model settings, system prompt, and RAG options
    /// @return true if creating chat and inserting system prompt succeeds, false on error
    bool create_chat(const ChatId &chat_id, const ChatConfig &chat_config) override;

    /// Retrieves the chat configuration for the specified chat session.
    /// The configuration is parsed from JSONB stored in the database and converted to a ChatConfig object.
    /// @param chat_id The chat identifier to retrieve configuration for
    /// @param chat_config Output parameter that will be populated with the chat configuration (modified in place)
    /// @return true if configuration retrieved successfully, false if chat_id doesn't exist or on error
    bool get_chat_config(const ChatId &chat_id, ChatConfig &chat_config) override;

    /// Retrieves all chat messages for the specified chat session.
    /// Messages are returned in chronological order based on sequence_index.
    /// @param chat_id The chat identifier to retrieve messages for
    /// @param messages Output parameter that will be populated with the chat messages (cleared and populated)
    /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error
    bool get_chat_history(const ChatId &chat_id, vector<ChatMessage> &messages) override;

    /// Inserts multiple chat messages into the database.
    /// Each message is assigned a sequence index automatically based on existing messages for the chat.
    /// @param chat_id Unique identifier for the chat session
    /// @param messages Vector of ChatMessage objects to insert
    /// @return true if all messages inserted successfully, false on error
    bool insert_chat_messages(const ChatId &chat_id, const vector<ChatMessage> &messages) override;

    /// Starts a transaction.
    /// Supports nested calls by flattening: real SQL transaction starts only on the first call.
    /// @return true if transaction started successfully (or was already active).
    bool begin_transaction() override;

    /// Commits a transaction.
    /// Supports nested calls: real SQL commit happens only when the outermost transaction commits.
    /// @return true if commit call was successful (or decremented depth).
    bool commit_transaction() override;

    /// Rolls back the entire transaction.
    /// This aborts the current transaction completely regardless of nesting depth.
    bool rollback_transaction() override;

    /// Closes the database connection and releases resources.
    void close() override;

private:
    int m_transaction_depth = 0;
    std::unique_ptr<SQLite::Transaction> m_transaction = nullptr;

    inline static string DB_SCHEMA = R"(
        
CREATE TABLE chats (
    chat_id        TEXT PRIMARY KEY,
    title          TEXT DEFAULT NULL,
    chat_config    BLOB NOT NULL,        -- JSON
    created_at     INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE chat_messages (
    message_id          INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    chat_id             TEXT NOT NULL,
    role                TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content             TEXT NOT NULL,
    sequence_index      INTEGER NOT NULL,
    message_metadata    BLOB,                -- JSON, let's store context / citation here, so that we can show it when displaying chat_history
    created_at          INTEGER NOT NULL DEFAULT (unixepoch()),
    
    FOREIGN KEY (chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE,
    UNIQUE(chat_id, sequence_index)
    );
    
CREATE INDEX idx_chat_messages_chat_id_seq 
ON chat_messages(chat_id, sequence_index);

-- Documents: The source of truth (File, Chat Thread, etc.)
CREATE TABLE document (
    id TEXT NOT NULL PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    scope_id TEXT NOT NULL,     -- Partition key (e.g., 'user_1', 'workspace_A', 'chat_x')
    source_uri TEXT NOT NULL,   -- File path or any ID that app can use to identify the document (e.g., chat_k)
    metadata TEXT,              -- JSON blob for flexibility
    created_at INTEGER NOT NULL
);

-- Chunks: The unique content blobs.
-- Deduplicated! If two docs have the exact same paragraph, we store it once.

-- ToDo 1. We need to delete orphan chunks , that is if no docs refer to a chunk clean them up, we can run a cron job for it
-- ToDo 2. Expose an resolver API or guideline framework, that allows the app to navigate to source
CREATE TABLE chunk (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    content_text TEXT NOT NULL,     -- The chunk content
    content_ref TEXT,               -- Optional app reference to map this chunk back to source (e.g. msg12_16, means msg 12 to 16, or any format)
    metadata TEXT,                  -- JSON blob for flexibility
    content_hash INTEGER NOT NULL UNIQUE -- Fast integer hash for deduplication checks
    );
    
-- Provenance: The Many-to-Many link.
-- Maps which Documents contain which Chunks.
CREATE TABLE doc_chunk_ref (
    doc_id TEXT NOT NULL,
    chunk_id INTEGER NOT NULL,
    sequence_index INTEGER NOT NULL, -- Order of chunk in the doc
    PRIMARY KEY (doc_id, chunk_id),
    FOREIGN KEY (doc_id) REFERENCES document(id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunk(id) ON DELETE CASCADE
    );

    -- Vector Store: The 'sqlite-vec' Virtual Table.
-- We use scope_id as a PARTITION KEY for fast filtering.
CREATE VIRTUAL TABLE vec_items USING vec0(
    chunk_id INTEGER PRIMARY KEY, -- Maps 1:1 to chunk.id
    embedding FLOAT[384],         -- Dimension depends on your model (384 is common for mobile/all-minilm)
    scope_id TEXT PARTITION KEY
);

)";
};