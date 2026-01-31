#pragma once

// This is a C Stable Header File

#include <cstdint>
#include <cstddef>
#include "types/odai_ctypes.h"
#include "types/odai_export.h"

// In your header

#ifdef __cplusplus
extern "C"
{
#endif

  /// Callback function type for logging messages.
  /// Called whenever a log message is generated at or below the current log level.
  /// @param level The log level of the message (OdaiLogLevel)
  /// @param message The log message string
  /// @param user_data User-provided data pointer passed when setting the logger
  typedef void (*odai_log_callback_fn)(OdaiLogLevel level, const char *message, void *user_data);

  /// Sets a custom logging callback function for receiving log messages.
  /// @param callback Function to call for each log message, or nullptr to disable custom logging
  /// @param user_data User-provided data pointer that will be passed to the callback function
  ODAI_API void odai_set_logger(odai_log_callback_fn callback, void *user_data);

  /// Sets the minimum log level for messages to be processed.
  /// Only messages at or below this level will be logged or passed to the callback.
  /// @param log_level Minimum log level (OdaiLogLevel) - use ODAI_LOG_ERROR, ODAI_LOG_WARN, ODAI_LOG_INFO, or ODAI_LOG_DEBUG
  ODAI_API void odai_set_log_level(OdaiLogLevel log_level);

  /// Initializes the SDK with database and backend engine configurations.
  /// Must be called before using any other SDK functions. Creates or opens the database at the specified path.
  /// @param dbConfig Configuration structure containing the database type and path
  /// @param backendEngineConfig Configuration structure specifying which backend engine to use
  /// @return true if initialization succeeded, false otherwise
  ODAI_API bool odai_initialize_sdk(const c_DBConfig *dbConfig, const c_BackendEngineConfig *backendEngineConfig);

  /// Adds a document to the RAG knowledge base for retrieval during generation.
  /// The document content is embedded and stored in the database for later retrieval.
  /// ToDo: Implementation not yet defined.
  /// @param content The text content of the document to add
  /// @param document_id Unique identifier for this document (used for updates/deletion)
  /// @param semantic_space_name Name of the semantic space to use
  /// @param scope_id Scope identifier to group documents (used for filtering during retrieval)
  /// @return true if document was added successfully, false otherwise
  ODAI_API bool odai_add_document(const char *content, const c_DocumentId document_id, const c_SemanticSpaceName semantic_space_name, const c_ScopeId scope_id);

  /// Generates a streaming response for the given query.
  /// Its like a Completion API, and won't use RAG
  /// This is a synchronous function that calls the callback for chunks of responses.
  /// @param llm_model_config The Language Model and its config to be used for response generation
  /// @param query The input query/prompt to generate a response for
  /// @param callback Function called for each generated chunk of response
  /// @param user_data User-provided data pointer passed to the callback function
  /// @return Total number of tokens generated, or -1 on error. Returns -1 if callback returns false to cancel streaming.
  ODAI_API int32_t odai_generate_streaming_response(const c_LLMModelConfig* llm_model_config, const char *query,
                                                    odai_stream_resp_callback_fn callback, void *user_data);

  /// Creates a new chat session with the specified configuration.
  /// If chat_id_in is nullptr, a unique chat ID will be generated and returned in chat_id_out.
  /// If chat_id_in is provided, it will be used as the chat ID (must be unique).
  /// @param chat_id_in Input chat ID (nullptr to auto-generate, or a unique string to use a specific ID)
  /// @param chat_config Configuration structure defining chat behavior (persistence, RAG usage, system prompt, model config, etc..)
  /// @param chat_id_out Buffer to receive the chat ID (must be pre-allocated)
  /// @param chat_id_out_len Pointer to the size of chat_id_out buffer (input), updated with actual length written (output)
  /// @return true if chat session was created successfully, false otherwise
  ODAI_API bool odai_create_chat(const c_ChatId chat_id_in, const c_ChatConfig *chat_config, c_ChatId chat_id_out, size_t *chat_id_out_len);


   /// Loads an existing chat by its ID and loads the chat KV cache into memory, along with the Language model
  /// Its only purpose is to pre-load a existing chat
  /// @param chat_id The unique identifier of the chat session to load
  /// @return true if chat session was loaded successfully, false if chat_id not found or on error
  ODAI_API bool odai_load_chat(const c_ChatId chat_id);

  /// Retrieves all chat messages for the specified chat session.
  /// Messages are returned in chronological order.
  /// The caller is responsible for freeing the allocated memory using odai_free_chat_messages().
  /// @param chat_id The chat identifier to retrieve messages for
  /// @param messages_out Output parameter: pointer to array of i_ChatMessage (allocated by this function)
  /// @param messages_count Output parameter: number of messages retrieved
  /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error
  ODAI_API bool odai_get_chat_history(const c_ChatId chat_id, c_ChatMessage **messages_out, size_t *messages_count);

  /// Frees memory allocated for an array of i_ChatMessage structures.
  /// Should be called after odai_get_chat_history to free allocated strings.
  /// @param messages Array of i_ChatMessage structures to free
  /// @param count Number of messages in the array
  ODAI_API void odai_free_chat_messages(c_ChatMessage *messages, size_t count);

  /// Generates a streaming chat response for the given query in the specified chat session.
  /// It will load  languagde model mentioned in chat config and load the chat history into context and then input the query and generate response
  /// If RAG is enabled for the chat, retrieves relevant context from the knowledge base.
  /// @param chat_id The unique identifier of the chat session
  /// @param query The input query/message to generate a response for
  /// @param semantic_space_name Name of the semantic space to use (ignored if RAG is disabled)
  /// @param scope_id Scope identifier to filter documents during RAG retrieval (ignored if RAG is disabled)
  /// @param callback Function called for each generated token
  /// @param user_data User-provided data pointer passed to the callback function
  /// @return true if response was generated successfully, false on error or if callback returns false to cancel streaming
  ODAI_API bool odai_generate_streaming_chat_response(const c_ChatId chat_id, const char *query, const c_SemanticSpaceName semantic_space_name, const c_ScopeId scope_id,
                                                      odai_stream_resp_callback_fn callback, void *user_data);

  /// Unloads the chat session from memory, freeing up resources (e.g., KV cache).
  /// @param chat_id The unique identifier of the chat session to unload
  /// @return true if chat session was unloaded successfully, false on error
  ODAI_API bool odai_unload_chat(const c_ChatId chat_id);

  /// Creates a new semantic space configuration.
  /// @param config The semantic space configuration to create.
  /// @return true if created successfully, false on error.
  ODAI_API bool odai_create_semantic_space(const c_SemanticSpaceConfig *config);

  /// Retrieves the configuration for a semantic space.
  /// @param name The name of the semantic space to retrieve.
  /// @param config_out Output parameter: pointer to array of the retrieved configuration (allocated by this function)
  /// @return true if found, false on error.
  ODAI_API bool odai_get_semantic_space(const c_SemanticSpaceName name, c_SemanticSpaceConfig *config_out);

  /// Frees members allocated by odai_get_semantic_space
  /// @param config pointer to c_SemanticSpaceConfig struct to free .
  ODAI_API void odai_free_semantic_space_config(c_SemanticSpaceConfig *config);

  /// Lists all available semantic spaces.
  /// Caller is responsible for freeing the allocated array using odai_free_semantic_spaces_list.
  /// @param spaces_out Output parameter: pointer to array of c_SemanticSpaceConfig (allocated by this function).
  /// @param spaces_count Output parameter: number of spaces returned.
  /// @return true if successful, false on error.
  ODAI_API bool odai_list_semantic_spaces(c_SemanticSpaceConfig **spaces_out, size_t *spaces_count);

  /// Frees memory allocated by odai_list_semantic_spaces.
  /// @param spaces Array of c_SemanticSpaceConfig to free.
  /// @param count Number of items in the array.
  ODAI_API void odai_free_semantic_spaces_list(c_SemanticSpaceConfig *spaces, size_t count);

  /// Deletes a semantic space configuration.
  /// @param name The name of the semantic space to delete.
  /// @return true if deleted successfully, false on error.
  ODAI_API bool odai_delete_semantic_space(const c_SemanticSpaceName name);


#ifdef __cplusplus
}
#endif