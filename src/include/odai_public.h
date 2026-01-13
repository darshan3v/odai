#pragma once

// This is a C Stable Header File

#include <cstdint>
#include <cstddef>
#include "types/odai_ctypes.h"

// In your header
#ifdef BUILDING_ODAI_SHARED
#define ODAI_API __attribute__((visibility("default")))
#else
#define ODAI_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

  /// Log level type for controlling logging verbosity.
  /// Use the constants ODAI_LOG_ERROR, ODAI_LOG_WARN, ODAI_LOG_INFO, ODAI_LOG_DEBUG.
  typedef uint8_t OdaiLogLevel;

  /// Backend engine type identifier.
  /// Use the constants like LLAMA_BACKEND_ENGINE to specify which backend to use.
  typedef uint8_t BackendEngineType;

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

  /// Initializes the RAG (Retrieval-Augmented Generation) engine with RAG configurations.
  /// Must be called before using RAG-related functions like odai_add_document or odai_generate_response.
  /// @param config Configuration structure containing embedding and language model paths and RAG related config
  /// @return true if RAG engine initialized successfully, false otherwise
  ODAI_API bool odai_initialize_rag_engine(const c_RagConfig *config);

  /// Adds a document to the RAG knowledge base for retrieval during generation.
  /// The document content is embedded and stored in the database for later retrieval.
  /// ToDo: Implementation not yet defined.
  /// @param content The text content of the document to add
  /// @param document_id Unique identifier for this document (used for updates/deletion)
  /// @param scope_id Scope identifier to group documents (used for filtering during retrieval)
  /// @return true if document was added successfully, false otherwise
  ODAI_API bool odai_add_document(const char *content, const c_DocumentId document_id, const c_ScopeId scope_id);

  /// Generates a streaming response for the given query using RAG.
  /// This is a synchronous function that calls the callback for each chunk.
  /// Retrieves relevant context from the knowledge base and streams the generated response token by token.
  /// @param query The input query/prompt to generate a response for
  /// @param scope_id Scope identifier to filter documents during retrieval
  /// @param callback Function called for each generated token
  /// @param user_data User-provided data pointer passed to the callback function
  /// @return Total number of tokens generated, or -1 on error. Returns -1 if callback returns false to cancel streaming.
  ODAI_API int32_t odai_generate_streaming_response(const char *query, const c_ScopeId scope_id,
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

  /// Loads an existing chat by its ID and loads the chat KV cache into memory.
  /// Must be called before generating responses for a specific chat session.
  /// This function loads the chat's key-value cache from persistent storage into memory for efficient context continuation.
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
  /// Uses the chat history and configuration from the loaded chat session.
  /// If RAG is enabled for the chat, retrieves relevant context from the knowledge base.
  /// ToDo: Implementation not yet defined.
  /// @param chat_id The unique identifier of the chat session
  /// @param query The input query/message to generate a response for
  /// @param scope_id Scope identifier to filter documents during RAG retrieval (ignored if RAG is disabled)
  /// @param callback Function called for each generated token
  /// @param user_data User-provided data pointer passed to the callback function
  /// @return true if response was generated successfully, false on error or if callback returns false to cancel streaming
  ODAI_API bool odai_generate_streaming_chat_response(const c_ChatId chat_id, const char *query, const c_ScopeId scope_id,
                                                      odai_stream_resp_callback_fn callback, void *user_data);

#ifdef __cplusplus
}
#endif