#pragma once

#ifdef ODAI_ENABLE_SQLITE_DB
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <SQLiteCpp/SQLiteCpp.h>
#include <nlohmann/json.hpp>

#include "db/odai_db.h"
#include "types/odai_types.h"

/// SQLite implementation of ODAIDb interface for managing RAG
/// (Retrieval-Augmented Generation) chat sessions and messages. Provides
/// functionality for initializing SQLite database with vector extensions,
/// managing chat sessions, and storing chat messages with metadata.
class OdaiSqliteDb : public IOdaiDb
{
private:
  std::unique_ptr<SQLite::Database> m_db = nullptr;

  uint16_t m_transactionDepth = 0;
  std::unique_ptr<SQLite::Transaction> m_transaction = nullptr;

  /// Registers the sqlite-vec extension and opens the database connection.
  /// The extension is registered before creating the database object to enable
  /// vector operations.
  /// @return true if extension registered, false on error
  static bool register_vec_extension();

  /// Internal helper to handle media item caching logic for both file paths and memory buffers.
  /// This will store the given media item in the cache directory and insert the mapping in db and update the item_out
  /// with the cached file path details.
  /// @note Here we assume the item being passed is not yet present in media store path
  /// @param item The media item to store
  /// @param checksum The pre-computed checksum of the media item to use for file_name and db entry
  /// @return stored item details on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<InputItem> store_media_item_impl(const InputItem& item, const std::string& checksum);

public:
  /// Constructs a new ODAISqliteDb instance with the specified database
  /// configuration. The database is not opened until initialize_db() is called.
  /// @param dbConfig Database configuration object.
  OdaiSqliteDb(const DBConfig& db_config);

  /// Destructor that cleans up database resources.
  /// ToDo : Not yet implemented
  ~OdaiSqliteDb() override;

  /// Initializes the database object (if db doesn't exist then create and
  /// initializes with schema) Registers the sqlite-vec extension, opens the
  /// connection, and initializes schema if needed.
  /// @return empty expected if initialization succeeded, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> initialize_db() override;

  /// Starts a transaction.
  /// Supports nested calls by flattening: real SQL transaction starts only on
  /// the first call.
  /// @return empty expected if the transaction state is valid, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> begin_transaction() override;

  /// Commits a transaction.
  /// Supports nested calls: real SQL commit happens only when the outermost
  /// transaction commits.
  /// @return empty expected if the transaction state is valid, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> commit_transaction() override;

  /// Rolls back the entire transaction.
  /// This aborts the current transaction completely regardless of nesting depth.
  /// @return empty expected if rollback completed, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> rollback_transaction() override;

  /// Registers a new model in the database.
  /// Stores the model name, details JSON, checksums JSON, and type in the models table.
  /// @param name The unique name to assign to the model
  /// @param model_file_details The generic model file details
  /// @param checksums The computed checksums for the files
  /// @return empty expected if registration succeeded, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<void> register_model_files(const ModelName& name, const ModelFiles& model_file_details,
                                        const std::string& checksums) override;

  /// Retrieves the generic details for a registered model.
  /// @param name The name of the model to look up
  /// @return model file details on success, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<ModelFiles> get_model_files(const ModelName& name) override;

  /// Retrieves the stored checksums for a registered model.
  /// @param name The name of the model to look up
  /// @return model checksums JSON on success, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<std::string> get_model_checksums(const ModelName& name) override;

  /// Updates the details for an existing model record.
  /// Note: This method expects `new_details` and `new_checksums` to contain the
  /// complete and comprehensive details mapping (replacing any existing record entirely).
  /// @param name The name of the model to update
  /// @param new_model_file_details The complete new model registration details to store
  /// @param new_checksums The complete new computed checksums
  /// @return empty expected if update succeeded, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<void> update_model_files(const ModelName& name, const ModelFiles& new_model_file_details,
                                      const std::string& new_checksums) override;

  /// @brief stores a media item in media store path and store the mapping in database.
  /// @note we don't store the media if its already present (based on checksum) in media store path, instead we just
  /// return the existing mapping.
  /// for in memory text (that is media type text and inputitemtype memory buffer, we directly set item_out and return
  /// success we don't store)
  /// @param item The media item to store
  /// @return stored item details on success, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<InputItem> store_media_item(const InputItem& item) override;

  /// Creates a new semantic space configuration in the database.
  /// @param config The semantic space configuration to store.
  /// @return empty expected if created successfully, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> create_semantic_space(const SemanticSpaceConfig& config) override;

  /// Retrieves the configuration for a semantic space.
  /// @param name The name of the semantic space to retrieve.
  /// @return semantic space configuration on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<SemanticSpaceConfig> get_semantic_space_config(const SemanticSpaceName& name) override;

  /// Lists all available semantic spaces from the database.
  /// @return semantic space configurations on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<std::vector<SemanticSpaceConfig>> list_semantic_spaces() override;

  /// Deletes a semantic space configuration from the database.
  /// @param name The name of the semantic space to delete.
  /// @return empty expected if deleted successfully, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> delete_semantic_space(const SemanticSpaceName& name) override;

  /// Checks if a chat session with the given chat_id exists in the database.
  /// @param chat_id The chat identifier to check
  /// @return true/false on success, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<bool> chat_id_exists(const ChatId& chat_id) override;

  /// Insert a new chat in DB with the specified configuration.
  /// Stores the chat configuration as JSONB in the database and inserts the
  /// system prompt as the initial message.
  /// @param chat_id Unique identifier for the chat session
  /// @param chat_config Configuration object containing model settings, system
  /// prompt, and RAG options
  /// @return empty expected if chat creation and system prompt insertion succeed, or an unexpected OdaiResultEnum
  /// indicating the error
  OdaiResult<void> create_chat(const ChatId& chat_id, const ChatConfig& chat_config) override;

  /// Retrieves the chat configuration for the specified chat session.
  /// The configuration is parsed from JSONB stored in the database and
  /// converted to a ChatConfig object.
  /// @param chat_id The chat identifier to retrieve configuration for
  /// @return chat configuration on success, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<ChatConfig> get_chat_config(const ChatId& chat_id) override;

  /// @brief Retrieves all chat messages for the specified chat session.
  /// @note Messages are returned in chronological order based on sequence_index.
  /// @note Any multimodal inputs (audio/image) that were previously cached
  /// using store_media_items() will be returned as InputItems with type FILE_PATH.
  /// Their m_data payload will contain the local absolute path string pointing
  /// to the cached file on disk, rather than the raw binary data.
  /// @param chat_id The chat identifier to retrieve messages for.
  /// @return chat messages on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<std::vector<ChatMessage>> get_chat_history(const ChatId& chat_id) override;

  /// @brief Inserts multiple chat messages into the database.
  /// This function attaches messages to an existing chat session. Each message
  /// is assigned an auto-incrementing sequence index for timeline preservation.
  /// @note All multimodal inputs (audio/image) inside message contents
  /// should be already cached in media store path and the mapping should be present in db.
  /// therefore, the msgContentItems should have item from store_media_items which will be basically of type file path
  /// except for text items which would be memory buffer
  /// @param chat_id Unique identifier for the chat session.
  /// @param messages Vector of ChatMessage objects to insert.
  /// @return empty expected if all messages were inserted successfully, or an unexpected OdaiResultEnum indicating the
  /// error.
  OdaiResult<void> insert_chat_messages(const ChatId& chat_id, const std::vector<ChatMessage>& messages) override;

  /// Closes the database connection and releases resources.
  void close() override;

private:
  inline static std::string db_schema = R"(
        
CREATE TABLE chats (
    chat_id        TEXT PRIMARY KEY,
    title          TEXT DEFAULT NULL,
    chat_config    BLOB NOT NULL,        -- JSON
    created_at     INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE media_cache (
    id             TEXT PRIMARY KEY,
    hash_xxhash    TEXT UNIQUE NOT NULL,
    mime_type      TEXT NOT NULL,
    absolute_path  TEXT NOT NULL, -- absolute_path of cached file not the original
    file_name      TEXT NOT NULL, -- file_name of cached file not the original
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

CREATE TABLE semantic_spaces (
    name TEXT NOT NULL PRIMARY KEY,
    config BLOB NOT NULL,       -- JSON stored SemanticSpaceConfig
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE models (
    name TEXT NOT NULL PRIMARY KEY,
    file_details BLOB NOT NULL,
    checksums BLOB NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('LLM', 'EMBEDDING')),
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

    -- Vector Store: The 'sqlite-vec' Virtual Table.
-- We use scope_id as a PARTITION KEY for fast filtering.
-- CREATE VIRTUAL TABLE vec_items USING vec0(
--    chunk_id INTEGER PRIMARY KEY, -- Maps 1:1 to chunk.id
--    embedding FLOAT[384],         -- Dimension depends on your model (384 is common for mobile/all-minilm)
--    scope_id TEXT PARTITION KEY
--);

)";
};

#endif // ODAI_ENABLE_SQLITE_DB
