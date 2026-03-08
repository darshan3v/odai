#pragma once

// This is a C Stable Header File

#include "types/odai_ctypes.h"
#include <cstddef>
#include <cstdint>

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
  typedef void (*OdaiLogCallbackFn)(OdaiLogLevel level, const char* message, void* user_data);

  /// Sets a custom logging callback function for receiving log messages.
  /// @param callback Function to call for each log message, or nullptr to disable custom logging
  /// @param user_data User-provided data pointer that will be passed to the callback function
  void odai_set_logger(OdaiLogCallbackFn callback, void* user_data);

  /// Sets the minimum log level for messages to be processed.
  /// Only messages at or below this level will be logged or passed to the callback.
  /// @param log_level Minimum log level (OdaiLogLevel) - use ODAI_LOG_ERROR, ODAI_LOG_WARN, ODAI_LOG_INFO, or
  /// ODAI_LOG_DEBUG
  void odai_set_log_level(OdaiLogLevel log_level);

  /// Initializes the SDK with database and backend engine configurations.
  /// Must be called before using any other SDK functions. Creates or opens the database at the specified path.
  /// @param db_config Configuration structure containing the database type and path
  /// @param backend_engine_config Configuration structure specifying which backend engine to use
  /// @return true if initialization succeeded, false otherwise
  bool odai_initialize_sdk(const c_DbConfig* db_config, const c_BackendEngineConfig* backend_engine_config);

  /// Registers a new model in the system with the given name and generic files map.
  /// The engine validates the files and a checksum is computed to ensure integrity.
  /// @param model_name The unique name to assign to the model.
  /// @param files The Model Files struct containing paths
  /// @return true if registration succeeded, false if name exists or properties are invalid.
  bool odai_register_model_files(c_ModelName model_name, const c_ModelFiles* files);

  /// Updates the registration files for an existing model.
  /// Validates the files and potentially checks checksums based on the flag.
  /// Note: At this SDK layer, only the newly added or explicitly modified files should
  /// be provided in `files`. Existing registrations for other properties will remain untouched.
  /// @param model_name The name of the model to update.
  /// @param files The Model Files struct containing newly added or updated paths.
  /// @param flag Flag indicating how to handle checksum changes.
  /// @return true if update succeeded, false if validation fails or model not found.
  bool odai_update_model_files(c_ModelName model_name, const c_ModelFiles* files, c_UpdateModelFlag flag);

  /// Creates a new semantic space configuration.
  /// @param config The semantic space configuration to create.
  /// @return true if created successfully, false on error.
  bool odai_create_semantic_space(const c_SemanticSpaceConfig* config);

  /// Retrieves the configuration for a semantic space.
  /// @param name The name of the semantic space to retrieve.
  /// @param config_out Output parameter: pointer to array of the retrieved configuration (allocated by this function)
  /// @return true if found, false on error.
  bool odai_get_semantic_space(c_SemanticSpaceName semantic_space_name, c_SemanticSpaceConfig* config_out);

  /// Frees members allocated by odai_get_semantic_space
  /// @param config pointer to c_SemanticSpaceConfig struct to free .
  void odai_free_semantic_space_config(c_SemanticSpaceConfig* config);

  /// Lists all available semantic spaces.
  /// Caller is responsible for freeing the allocated array using odai_free_semantic_spaces_list.
  /// @param spaces_out Output parameter: pointer to array of c_SemanticSpaceConfig (allocated by this function).
  /// @param spaces_count Output parameter: number of spaces returned.
  /// @return true if successful, false on error.
  bool odai_list_semantic_spaces(c_SemanticSpaceConfig** spaces_out, uint16_t* spaces_count);

  /// Frees memory allocated by odai_list_semantic_spaces.
  /// @param spaces Array of c_SemanticSpaceConfig to free.
  /// @param count Number of items in the array.
  void odai_free_semantic_spaces_list(c_SemanticSpaceConfig* spaces, uint16_t count);

  /// Deletes a semantic space configuration.
  /// @param name The name of the semantic space to delete.
  /// @return true if deleted successfully, false on error.
  bool odai_delete_semantic_space(c_SemanticSpaceName name);

  /// Adds a document to the RAG knowledge base for retrieval during generation.
  /// The document content is embedded and stored in the database for later retrieval.
  /// ToDo: Implementation not yet defined.
  /// @param content The text content of the document to add
  /// @param document_id Unique identifier for this document (used for updates/deletion)
  /// @param semantic_space_name Name of the semantic space to use
  /// @param scope_id Scope identifier to group documents (used for filtering during retrieval)
  /// @return true if document was added successfully, false otherwise
  bool odai_add_document(const char* content, c_DocumentId document_id, c_SemanticSpaceName semantic_space_name,
                         c_ScopeId scope_id);

  /// Generates a streaming response for a single query using the specified LLM Model.
  /// @param llm_model_config Configuration of the LLM model to use
  /// @param c_prompt_items Array of input items forming the query (text, images, etc.)
  /// @param prompt_items_count Number of input items in the array
  /// @param c_sampler_config Sampling parameters for generation (temperature, top_p, etc.)
  /// @param c_callback Function to be called for each generated chunk
  /// @param c_user_data User-provided data pointer passed to the callback function
  /// @return Total number of tokens generated, or -1 on error. Returns -1 if callback returns false to cancel
  /// streaming.
  int32_t odai_generate_streaming_response(const c_LlmModelConfig* llm_model_config, const c_InputItem* c_prompt_items,
                                           uint16_t prompt_items_count, const c_SamplerConfig* c_sampler_config,
                                           OdaiStreamRespCallbackFn c_callback, void* c_user_data);

  /// Creates a new chat session with the specified configuration.
  /// If chat_id_in is nullptr, a unique chat ID will be generated and returned in chat_id_out.
  /// If chat_id_in is provided, it will be used as the chat ID (must be unique).
  /// @param c_chat_id_in Input chat ID (nullptr to auto-generate, or a unique string to use a specific ID)
  /// @param c_chat_config Configuration structure defining chat behavior (persistence, RAG usage, system prompt, model
  /// config, etc..)
  /// @param c_chat_id_out Output parameter: pointer to the chat ID string (allocated by this function, caller must free
  /// using odai_free_chat_id)
  /// @return true if chat session was created successfully, false otherwise
  bool odai_create_chat(c_ChatId c_chat_id_in, const c_ChatConfig* c_chat_config, c_ChatId* c_chat_id_out);

  /// Frees the chat ID string allocated by odai_create_chat.
  /// @param chat_id The chat ID string to free
  void odai_free_chat_id(c_ChatId chat_id);

  /// Retrieves all chat messages for the specified chat session.
  /// Messages are returned in chronological order.
  /// The caller is responsible for freeing the allocated memory using odai_free_chat_messages().
  /// @param c_chat_id The chat identifier to retrieve messages for
  /// @param c_messages_out Output parameter: pointer to array of i_ChatMessage (allocated by this function)
  /// @param messages_count Output parameter: number of messages retrieved
  /// @return true if messages retrieved successfully, false if chat_id doesn't exist or on error
  bool odai_get_chat_history(c_ChatId c_chat_id, c_ChatMessage** c_messages_out, uint16_t* messages_count);

  /// Frees memory allocated for an array of i_ChatMessage structures.
  /// Should be called after odai_get_chat_history to free allocated strings.
  /// @param messages Array of i_ChatMessage structures to free
  /// @param count Number of messages in the array
  void odai_free_chat_messages(c_ChatMessage* messages, uint16_t count);

  /// Generates a streaming response for an existing chat session.
  /// @param c_chat_id The unique identifier of the chat session
  /// @param c_prompt_items Array of input items forming the query (text, images, etc.)
  /// @param prompt_items_count Number of input items in the array
  /// @param c_generator_config Configuration governing both RAG (if used) and generation sampling
  /// @param callback Function to be called for each generated text chunk
  /// @param user_data Opaque pointer passed back to the callback
  /// @return true if generation starts successfully, false if chat not found or on error
  bool odai_generate_streaming_chat_response(c_ChatId c_chat_id, const c_InputItem* c_prompt_items,
                                             uint16_t prompt_items_count, const c_GeneratorConfig* c_generator_config,
                                             OdaiStreamRespCallbackFn callback, void* user_data);

#ifdef __cplusplus
}
#endif