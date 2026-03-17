#pragma once

#include "types/odai_types.h"
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Forward declarations
struct ModelFiles;
struct InputItem;
struct SemanticSpaceConfig;
struct ChatConfig;
struct ChatMessage;

// Typedefs from odai_types.h (since we can't include it fully)
typedef std::string ChatId;
typedef std::string DocumentId;
typedef std::string ScopeId;
typedef std::string SemanticSpaceName;
typedef std::string ModelName;

/// Abstract interface for database backends managing RAG (Retrieval-Augmented Generation) chat sessions and messages.
/// Provides functionality for managing chat sessions, storing chat messages with metadata, and more.
/// Implementations can use different database backends (e.g., SQLite, PostgreSQL, etc.).
class IOdaiDb
{
protected:
  DBConfig m_dbConfig;

public:
  IOdaiDb(const DBConfig& db_config)
  {
    if (!db_config.is_sane())
    {
      throw std::invalid_argument("Invalid DBConfig provided");
    }

    m_dbConfig = db_config;
  }

  IOdaiDb(const IOdaiDb&) = delete;
  IOdaiDb& operator=(const IOdaiDb&) = delete;
  IOdaiDb(IOdaiDb&&) = delete;
  IOdaiDb& operator=(IOdaiDb&&) = delete;

  /// Virtual destructor
  virtual ~IOdaiDb() = default;

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
  /// Stores the model name, model_files JSON, checksums JSON, and type in the models table.
  /// @param name The unique name to assign to the model
  /// @param model_file_details The generic model file details
  /// @param checksums_json The computed checksums for the files
  /// @return true if registration succeeded, false on error
  virtual bool register_model_files(const ModelName& name, const ModelFiles& model_file_details,
                                    const std::string& checksums_json) = 0;

  /// Retrieves the generic details for a registered model.
  /// @param name The name of the model to look up
  /// @param model_file_details Output parameter to store the model file details
  /// @return true if model found, false if not found or on error
  virtual bool get_model_files(const ModelName& name, ModelFiles& model_file_details) = 0;

  /// Retrieves the stored checksums for a registered model.
  /// @param name The name of the model to look up
  /// @param checksums Output parameter to store the model checksums
  /// @return true if model found, false if not found or on error
  virtual bool get_model_checksums(const ModelName& name, std::string& checksums) = 0;

  /// Updates the details for an existing model record.
  /// @note The database layer expects `new_file_details` and `new_checksums` to contain the complete, comprehensive set
  /// of all details (both existing and newly added).This will overwrite and replace the previously stored details
  /// entirely.
  /// @param name The name of the model to update
  /// @param new_model_file_details The complete new registration details to store
  /// @param new_checksums The complete new computed checksums
  /// @return true if update succeeded, false on error
  virtual bool update_model_files(const ModelName& name, const ModelFiles& new_model_file_details,
                                  const std::string& new_checksums) = 0;

  /// @brief stores a media item whererver it seems fit and store the mapping in database.
  /// for in memory text (that is media type text and inputitemtype memory buffer, we should set item_out and directly
  /// return success and not store anything)
  /// @param item The media item to store
  /// @param item_out Output parameter to receive the stored item details. currently we expect the item_out to have item
  /// type as file path but this can be extended in future if needed.
  /// @return true if storage succeeded, false on error
  virtual bool store_media_item(const InputItem& item, InputItem& item_out) = 0;

  /// Creates a new semantic space.
  /// @param config The configuration for the semantic space.
  /// @return true if created successfully, false on error.
  virtual bool create_semantic_space(const SemanticSpaceConfig& config) = 0;

  /// Retrieves the configuration for a semantic space.
  /// @param name The name of the semantic space to retrieve.
  /// @param config Output parameter to store the configuration.
  /// @return true if found, false on error or if not found.
  virtual bool get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config) = 0;

  /// Lists all available semantic spaces.
  /// @param spaces Output parameter to store list of space configs.
  /// @return true if successful, false on error.
  virtual bool list_semantic_spaces(std::vector<SemanticSpaceConfig>& spaces) = 0;

  /// Deletes a semantic space.
  /// @param name The name of the semantic space to delete.
  /// @return true if deleted successfully, false on error.
  virtual bool delete_semantic_space(const SemanticSpaceName& name) = 0;

  /// Checks if a chat session with the given chat_id exists in the database.
  /// @param chat_id The chat identifier to check
  /// @return true if chat_id exists, false if not found or on error
  virtual bool chat_id_exists(const ChatId& chat_id) = 0;

  /// Insert a new chat in DB with the specified configuration.
  /// Stores the chat configuration and inserts the system prompt as the initial message.
  /// @param chat_id Unique identifier for the chat session
  /// @param chat_config Configuration object containing model settings, system prompt, and RAG options
  /// @return true if creating chat and inserting system prompt succeeds, false on error
  virtual bool create_chat(const ChatId& chat_id, const ChatConfig& chat_config) = 0;

  /// Retrieves the chat configuration for the specified chat session.
  /// @param chat_id The chat identifier to retrieve configuration for
  /// @param chat_config Output parameter that will be populated with the chat configuration
  /// @return true if configuration retrieved successfully, false if chat_id doesn't exist or on error
  virtual bool get_chat_config(const ChatId& chat_id, ChatConfig& chat_config) = 0;

  /// Retrieves all chat messages for the specified chat session.
  ///
  /// Messages are returned in chronological order.
  /// @note for multimodal inputs (audio/image): The implementation may return cached files as InputItems with type
  /// FILE_PATH, and the m_data payload containing the local absolute path string pointing to the cached file on disk,
  /// rather than raw binary data.
  /// @param chat_id The chat identifier to retrieve messages for.
  /// @param messages Output parameter that will be populated with the chat messages (cleared and populated).
  /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error.
  virtual bool get_chat_history(const ChatId& chat_id, std::vector<ChatMessage>& messages) = 0;

  /// Inserts multiple chat messages into the database.
  ///
  /// Each message is assigned a sequence index automatically based on existing messages for the chat.
  /// @note we expect that messages contentItems to have input item of type File Path, except for text which would be
  /// memory buffer caller can store_media_items() to store the media items in and get the new item struct with the new
  /// file path from stored location.
  /// @note This operation is wrapped in a transaction. If any insertion fails, the entire operation is rolled back.
  /// @param chat_id Unique identifier for the chat session.
  /// @param messages Vector of ChatMessage objects to insert. The objects' contents might be modified (e.g. data
  /// replaced by file paths).
  /// @return true if all messages inserted successfully, false on error.
  virtual bool insert_chat_messages(const ChatId& chat_id, const std::vector<ChatMessage>& messages) = 0;

  /// Closes the database connection and releases resources.
  virtual void close() = 0;
};