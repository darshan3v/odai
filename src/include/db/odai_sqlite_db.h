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

using namespace std;
using namespace nlohmann;

/// SQLite implementation of ODAIDb interface for managing RAG
/// (Retrieval-Augmented Generation) chat sessions and messages. Provides
/// functionality for initializing SQLite database with vector extensions,
/// managing chat sessions, and storing chat messages with metadata.
class OdaiSqliteDb : public IOdaiDb
{

private:
  string m_dbPath;
  string m_cacheDirPath;
  unique_ptr<SQLite::Database> m_db = nullptr;

  uint16_t m_transactionDepth = 0;
  std::unique_ptr<SQLite::Transaction> m_transaction = nullptr;

  /// Registers the sqlite-vec extension and opens the database connection.
  /// The extension is registered before creating the database object to enable
  /// vector operations.
  /// @return true if extension registered, false on error
  static bool register_vec_extension();

  /// Processes and caches media items (such as audio or image files and buffers)
  /// into the media cache directory to prevent storing large blobs directly in the database.
  ///
  /// This method calculates the xxhash checksum of the media item. If the item is not
  /// found in the cache, it is written to the file system. In both cases, the properties
  /// of the InputItem are modified in-place: its type is updated to FILE_PATH,
  /// and its data is replaced with the absolute path to the cached file.
  ///
  /// @param item The InputItem to process and mutate in place.
  /// @return true if processing and caching succeeded, false upon an error.
  bool process_and_cache_media_item(InputItem& item);

public:
  /// Constructs a new ODAISqliteDb instance with the specified database
  /// configuration. The database is not opened until initialize_db() is called.
  /// @param dbConfig Database configuration object.
  /// @param cache_dir_path Path to the cache directory.
  OdaiSqliteDb(const DBConfig& db_config, const string& cache_dir_path);

  /// Destructor that cleans up database resources.
  /// ToDo : Not yet implemented
  ~OdaiSqliteDb() override;

  /// Initializes the database object (if db doesn't exist then create and
  /// initializes with schema) Registers the sqlite-vec extension, opens the
  /// connection, and initializes schema if needed.
  /// @return true if initialization succeeded, false otherwise
  bool initialize_db() override;

  /// Starts a transaction.
  /// Supports nested calls by flattening: real SQL transaction starts only on
  /// the first call.
  /// @return true if transaction started successfully (or was already active).
  bool begin_transaction() override;

  /// Commits a transaction.
  /// Supports nested calls: real SQL commit happens only when the outermost
  /// transaction commits.
  /// @return true if commit call was successful (or decremented depth).
  bool commit_transaction() override;

  /// Rolls back the entire transaction.
  /// This aborts the current transaction completely regardless of nesting
  /// depth.
  bool rollback_transaction() override;

  /// Registers a new model in the database.
  /// Stores the model name, details JSON, checksums JSON, and type in the models table.
  /// @param name The unique name to assign to the model
  /// @param details The generic model registration details
  /// @param checksums The computed checksums for the files
  /// @return true if registration succeeded, false on error
  bool register_model_files(const ModelName& name, const ModelFiles& details, const string& checksums) override;

  /// Retrieves the generic details for a registered model.
  /// @param name The name of the model to look up
  /// @param details Output parameter to store the model details
  /// @return true if model found, false if not found or on error
  bool get_model_files(const ModelName& name, ModelFiles& details) override;

  /// Retrieves the stored checksums for a registered model.
  /// @param name The name of the model to look up
  /// @param checksums Output parameter to store the model checksums
  /// @return true if model found, false if not found or on error
  bool get_model_checksums(const ModelName& name, string& checksums) override;

  /// Updates the details for an existing model record.
  /// Note: This method expects `new_details` and `new_checksums` to contain the
  /// complete and comprehensive details mapping (replacing any existing record entirely).
  /// @param name The name of the model to update
  /// @param new_details The complete new registration details to store
  /// @param new_checksums The complete new computed checksums
  /// @return true if update succeeded, false on error
  bool update_model_files(const ModelName& name, const ModelFiles& new_details, const string& new_checksums) override;

  /// Creates a new semantic space configuration in the database.
  /// @param config The semantic space configuration to store.
  /// @return true if created successfully, false on error.
  bool create_semantic_space(const SemanticSpaceConfig& config) override;

  /// Retrieves the configuration for a semantic space.
  /// @param name The name of the semantic space to retrieve.
  /// @param config Output parameter to store the configuration.
  /// @return true if found, false on error or if not found.
  bool get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config) override;

  /// Lists all available semantic spaces from the database.
  /// @param spaces Vector to be populated with the list of semantic spaces.
  /// @return true if listed successfully, false on error.
  bool list_semantic_spaces(vector<SemanticSpaceConfig>& spaces) override;

  /// Deletes a semantic space configuration from the database.
  /// @param name The name of the semantic space to delete.
  /// @return true if deleted successfully, false on error.
  bool delete_semantic_space(const SemanticSpaceName& name) override;

  /// Checks if a chat session with the given chat_id exists in the database.
  /// @param chat_id The chat identifier to check
  /// @return true if chat_id exists, false if not found or on error
  bool chat_id_exists(const ChatId& chat_id) override;

  /// Insert a new chat in DB with the specified configuration.
  /// Stores the chat configuration as JSONB in the database and inserts the
  /// system prompt as the initial message.
  /// @param chat_id Unique identifier for the chat session
  /// @param chat_config Configuration object containing model settings, system
  /// prompt, and RAG options
  /// @return true if creating chat and inserting system prompt succeeds, false
  /// on error
  bool create_chat(const ChatId& chat_id, const ChatConfig& chat_config) override;

  /// Retrieves the chat configuration for the specified chat session.
  /// The configuration is parsed from JSONB stored in the database and
  /// converted to a ChatConfig object.
  /// @param chat_id The chat identifier to retrieve configuration for
  /// @param chat_config Output parameter that will be populated with the chat
  /// configuration (modified in place)
  /// @return true if configuration retrieved successfully, false if chat_id
  /// doesn't exist or on error
  bool get_chat_config(const ChatId& chat_id, ChatConfig& chat_config) override;

  /// Retrieves all chat messages for the specified chat session.
  ///
  /// Messages are returned in chronological order based on sequence_index.
  /// Note: Any multimodal inputs (audio/image) that were previously cached
  /// during insertion will be returned as InputItems with type FILE_PATH.
  /// Their m_data payload will contain the local absolute path string pointing
  /// to the cached file on disk, rather than the raw binary data.
  ///
  /// @param chat_id The chat identifier to retrieve messages for.
  /// @param messages Output parameter that will be populated with the chat messages (cleared and populated).
  /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error.
  bool get_chat_history(const ChatId& chat_id, vector<ChatMessage>& messages) override;

  /// Inserts multiple chat messages into the database.
  ///
  /// This function attaches messages to an existing chat session. Each message
  /// is assigned an auto-incrementing sequence index for timeline preservation.
  /// Furthermore, all multimodal inputs (audio/image) inside message contents
  /// are automatically processed, deduplicated (via xxhash checksum comparison),
  /// and stored directly onto the filesystem (media cache) rather than stuffing
  /// large blobs into the chat_messages table. The database stores the file
  /// paths instead.
  ///
  /// IMPORTANT: Media within `messages` must be strictly provided as true format
  /// sources (FILE_PATH or MEMORY_BUFFER). Pre-processed or decoded audio data
  /// (`PROCESSED_DATA`) will be rejected since deduplication routines operate on
  /// raw compressed binary payloads.
  ///
  /// @param chat_id Unique identifier for the chat session.
  /// @param messages Vector of ChatMessage objects to insert. Its multimodal items will be modified after caching.
  /// @return true if all messages were inserted and cached successfully, false on error.
  bool insert_chat_messages(const ChatId& chat_id, const vector<ChatMessage>& messages) override;

  /// Closes the database connection and releases resources.
  void close() override;

private:
  inline static string db_schema = R"(
        
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
    details BLOB NOT NULL,
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