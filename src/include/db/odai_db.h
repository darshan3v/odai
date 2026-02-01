#pragma once

#include <string>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "types/odai_types.h"

using namespace std;
using namespace nlohmann;

/// Abstract interface for database backends managing RAG (Retrieval-Augmented Generation) chat sessions and messages.
/// Provides functionality for managing chat sessions, storing chat messages with metadata, and more.
/// Implementations can use different database backends (e.g., SQLite, PostgreSQL, etc.).
class ODAIDb
{
public:

    /// Virtual destructor
    virtual ~ODAIDb() = default;

    /// Initializes the database backend.
    /// Should be called before any other operations.
    /// @return true if initialization succeeded, false otherwise
    virtual bool initialize_db() = 0;

    /// Starts a transaction.
    /// Supports nested calls by flattening: real transaction starts only on the first call.
    /// @return true if transaction started successfully (or was already active).
    virtual bool begin_transaction() = 0;

    /// Commits a transaction.
    /// Supports nested calls: real commit happens only when the outermost transaction commits.
    /// @return true if commit call was successful (or decremented depth).
    virtual bool commit_transaction() = 0;

    /// Rolls back the entire transaction.
    /// This aborts the current transaction completely regardless of nesting depth.
    virtual bool rollback_transaction() = 0;

    /// Registers a new model in the database.
    /// Stores the model name, path, checksum and type in the models table.
    /// @param name The unique name to assign to the model
    /// @param path The full file system path to the model file
    /// @param type The type of the model (LLM or EMBEDDING)
    /// @param checksum The checksum of the model file for integrity verification
    /// @return true if registration succeeded, false on error
    virtual bool register_model(const ModelName& name, const ModelPath& path, ModelType type, const string& checksum) = 0;

    /// Retrieves the file system path for a registered model.
    /// @param name The name of the model to look up
    /// @param path Output parameter to store the model path
    /// @return true if model found, false if not found or on error
    virtual bool get_model_path(const ModelName& name, ModelPath& path) = 0;

    /// Retrieves the stored checksum for a registered model.
    /// @param name The name of the model to look up
    /// @param checksum Output parameter to store the model checksum
    /// @return true if model found, false if not found or on error
    virtual bool get_model_checksum(const ModelName& name, string& checksum) = 0;

    /// Updates the path for an existing model record.
    /// @param name The name of the model to update
    /// @param new_path The new file system path to store
    /// @return true if update succeeded, false on error
    virtual bool update_model_path(const ModelName& name, const ModelPath& new_path) = 0;

    /// Creates a new semantic space.
    /// @param config The configuration for the semantic space.
    /// @return true if created successfully, false on error.
    virtual bool create_semantic_space(const SemanticSpaceConfig &config) = 0;

    /// Retrieves the configuration for a semantic space.
    /// @param name The name of the semantic space to retrieve.
    /// @param config Output parameter to store the configuration.
    /// @return true if found, false on error or if not found.
    virtual bool get_semantic_space_config(const SemanticSpaceName &name, SemanticSpaceConfig &config) = 0;

    /// Lists all available semantic spaces.
    /// @param spaces Output parameter to store list of space configs.
    /// @return true if successful, false on error.
    virtual bool list_semantic_spaces(vector<SemanticSpaceConfig> &spaces) = 0;

    /// Deletes a semantic space.
    /// @param name The name of the semantic space to delete.
    /// @return true if deleted successfully, false on error.
    virtual bool delete_semantic_space(const SemanticSpaceName &name) = 0;

    /// Checks if a chat session with the given chat_id exists in the database.
    /// @param chat_id The chat identifier to check
    /// @return true if chat_id exists, false if not found or on error
    virtual bool chat_id_exists(const ChatId &chat_id) = 0;

    /// Insert a new chat in DB with the specified configuration.
    /// Stores the chat configuration and inserts the system prompt as the initial message.
    /// @param chat_id Unique identifier for the chat session
    /// @param chat_config Configuration object containing model settings, system prompt, and RAG options
    /// @return true if creating chat and inserting system prompt succeeds, false on error
    virtual bool create_chat(const ChatId &chat_id, const ChatConfig &chat_config) = 0;

    /// Retrieves the chat configuration for the specified chat session.
    /// @param chat_id The chat identifier to retrieve configuration for
    /// @param chat_config Output parameter that will be populated with the chat configuration
    /// @return true if configuration retrieved successfully, false if chat_id doesn't exist or on error
    virtual bool get_chat_config(const ChatId &chat_id, ChatConfig &chat_config) = 0;

    /// Retrieves all chat messages for the specified chat session.
    /// Messages are returned in chronological order
    /// @param chat_id The chat identifier to retrieve messages for
    /// @param messages Output parameter that will be populated with the chat messages (cleared and populated)
    /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error
    virtual bool get_chat_history(const ChatId &chat_id, vector<ChatMessage> &messages) = 0;

    /// Inserts multiple chat messages into the database.
    /// Each message is assigned a sequence index automatically based on existing messages for the chat.
    /// @note This operation is wrapped in a transaction. If any insertion fails, the entire operation is rolled back.
    /// @param chat_id Unique identifier for the chat session
    /// @param messages Vector of ChatMessage objects to insert
    /// @return true if all messages inserted successfully, false on error
    virtual bool insert_chat_messages(const ChatId &chat_id, const vector<ChatMessage> &messages) = 0;

    /// Closes the database connection and releases resources.
    virtual void close() = 0;
};