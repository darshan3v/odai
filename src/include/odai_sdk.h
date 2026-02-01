#pragma once

#include "types/odai_types.h"
#include "types/odai_export.h"
#include <string>
#include <vector>
#include <memory>

#include "odai_logger.h"
#include "db/odai_db.h"
#include "backendEngine/odai_backend_engine.h"
#include "ragEngine/odai_rag_engine.h"

using namespace std;

/// C++ Entry point for ODAI SDK
class ODAI_API ODAISdk {
public:
    /// Get the singleton instance of the SDK
    static ODAISdk& get_instance();

    /// Prevent copying and assignment
    ODAISdk(const ODAISdk&) = delete;
    ODAISdk& operator=(const ODAISdk&) = delete;

    /// Sets a custom logging callback function for receiving log messages.
    /// @param callback Function to call for each log message, or nullptr to disable custom logging
    /// @param user_data User-provided data pointer that will be passed to the callback function
    void set_logger(odai_log_callback_fn callback, void *user_data);

    /// Sets the minimum log level for messages to be processed.
    /// Only messages at or below this level will be logged or passed to the callback.
    /// @param log_level Minimum log level (OdaiLogLevel)
    void set_log_level(OdaiLogLevel log_level);

    /// Initializes the SDK with database and backend engine configurations.
    /// Must be called before using any other SDK functions.
    /// @param dbConfig Configuration structure containing the database type and path
    /// @param backendConfig Configuration structure specifying which backend engine to use
    /// @return true if initialization succeeded, false otherwise
    bool initialize_sdk(const DBConfig& dbConfig, const BackendEngineConfig& backendConfig);

    /// Registers a new model.
    /// @param name The unique name of the model.
    /// @param path The file path to the model.
    /// @param type The type of the model.
    /// @return true if registered successfully, false otherwise.
    bool register_model(const ModelName& name, const ModelPath& path, ModelType type);

    /// Updates the path for a model.
    /// @param name The name of the model.
    /// @param path The new path.
    /// @return true if updated successfully, false otherwise.
    bool update_model_path(const ModelName& name, const ModelPath& path);

    /// Creates a new semantic space.
    /// @param config The configuration for the semantic space.
    /// @return true if created successfully, false on error.
    bool create_semantic_space(const SemanticSpaceConfig& config);

    /// Retrieves the configuration for a semantic space.
    /// @param name The name of the semantic space to retrieve.
    /// @param config Output parameter to store the configuration.
    /// @return true if found, false on error or if not found.
    bool get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config);

    /// Lists all available semantic spaces.
    /// @param spaces Output parameter to store list of space names.
    /// @return true if successful, false on error.
    bool list_semantic_spaces(vector<SemanticSpaceConfig>& spaces);

    /// Deletes a semantic space.
    /// @param name The name of the semantic space to delete.
    /// @return true if deleted successfully, false on error.
    bool delete_semantic_space(const string& name);

    /// Adds a document to the RAG knowledge base for retrieval during generation.
    /// @param content The text content of the document to add
    /// @param documentId Unique identifier for this document
    /// @param semanticSpaceName Name of the semantic space to use
    /// @param scopeId Scope identifier to group documents
    /// @return true if document was added successfully, false otherwise
    bool add_document(const string& content, const DocumentId& documentId, const SemanticSpaceName& semanticSpaceName, const ScopeId& scopeId);

    /// Generates a streaming response for the given query.
    /// Its like a Completion API, and won't use RAG
    /// @param llmModelConfig The Language Model and its config to be used for response generation
    /// @param query The input query/prompt
    /// @param callback Function called for each generated token
    /// @param userData User-provided data pointer passed to the callback function
    /// @return Total number of tokens generated, or -1 on error
    int32_t generate_streaming_response(const LLMModelConfig& llmModelConfig, const string& query, 
                                     odai_stream_resp_callback_fn callback, void *userData);

    /// Creates a new chat session with the specified configuration.
    /// @param chatIdIn Input chat ID (empty to auto-generate)
    /// @param chatConfig Configuration structure defining chat behavior
    /// @param chatIdOut Output parameter receiving the chat ID (generated or passed through)
    /// @return true if chat session was created successfully, false otherwise
    bool create_chat(const ChatId& chatIdIn, const ChatConfig& chatConfig, ChatId& chatIdOut);

    /// Loads an existing chat by its ID and loads the chat KV cache into memory, along with the Language model
    /// Its only purpose is to pre-load a existing chat
    /// @param chatId The unique identifier of the chat session to load
    /// @return true if chat session was loaded successfully, false if not found or on error
    bool load_chat(const ChatId& chatId);

    /// Retrieves all chat messages for the specified chat session.
    /// @param chatId The chat identifier to retrieve messages for
    /// @param messages Output parameter: vector of ChatMessage
    /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error
    bool get_chat_history(const ChatId& chatId, vector<ChatMessage>& messages);

    /// Generates a streaming chat response for the given query in the specified chat session.
    /// It will load  languagde model mentioned in chat config and load the chat history into context and then input the query and generate response
    /// @param chatId The unique identifier of the chat session
    /// @param query The input query/message
    /// @param semanticSpaceName Name of the semantic space to use (ignored if RAG is disabled)
    /// @param scopeId Scope identifier to filter documents during RAG retrieval (ignored if RAG is disabled)
    /// @param callback Function called for each generated token
    /// @param userData User-provided data pointer passed to the callback function
    /// @return true if response was generated successfully, false on error
    bool generate_streaming_chat_response(const ChatId& chatId, const string& query, const SemanticSpaceName& semanticSpaceName, const ScopeId& scopeId,
                                      odai_stream_resp_callback_fn callback, void *userData);

    /// Unloads the chat session from memory, freeing up resources.
    /// @param chatId The unique identifier of the chat session to unload
    /// @return true if chat session was unloaded successfully, false on error
    bool unload_chat(const ChatId& chatId);

private:
    ODAISdk();
    ~ODAISdk();

    bool m_sdkInitialized = false;

    std::unique_ptr<ODAILogger> m_logger;
    std::unique_ptr<ODAIDb> m_db;
    std::unique_ptr<ODAIBackendEngine> m_backendEngine;
    std::unique_ptr<ODAIRagEngine> m_ragEngine;

public:
    ODAILogger* get_logger() { return m_logger.get(); }
    ODAIBackendEngine* get_backend_engine() { return m_backendEngine.get(); }
    ODAIDb* get_db() { return m_db.get(); }
};

#define ODAI_LOG(level, fmt, ...) \
    if (auto logger = ODAISdk::get_instance().get_logger()) \
        logger->log(level, "[{}:{}] " fmt, __func__, __LINE__, ##__VA_ARGS__)
