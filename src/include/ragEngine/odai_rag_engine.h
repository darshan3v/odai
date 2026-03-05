#pragma once

#include "audioEngine/odai_audio_decoder.h"
#include "backendEngine/odai_backend_engine.h"
#include "db/odai_db.h"
#include "types/odai_types.h"
#include <memory>
#include <unordered_map>

/// RAG (Retrieval-Augmented Generation) engine that combines embedding and
/// language models for context-aware text generation. Manages the
/// initialization of models and generation of streaming responses using
/// retrieved context.
class OdaiRagEngine
{
public:
  OdaiRagEngine(const DBConfig& db_config, const BackendEngineConfig& backend_config, const SdkConfig& sdk_config);

  /// Registers a new model in the system with the given name and paths.
  /// The backend engine validates the paths and computes checksums.
  /// @param name The unique name to assign to the model.
  /// @param details The Model Registration Details struct containing paths.
  /// @return true if registration succeeded, false if name exists or details are invalid.
  bool register_model_files(const ModelName& name, const ModelFiles& details);

  /// Updates the registration details for an existing model.
  /// Validates the details and potentially checks checksums based on the flag.
  /// Note: At the Engine/SDK layer, this method expects `details` to contain only
  /// the newly added or updated files. It will be merged with existing details.
  /// @param name The name of the model to update.
  /// @param details The Model Registration Details struct containing newly updated files.
  /// @param flag Flag indicating how to handle checksum changes.
  /// @return true if update succeeded, false if validation fails or model not found.
  bool update_model_files(const ModelName& name, const ModelFiles& details, UpdateModelFlag flag);

  /// Generates a streaming response for the given query.
  /// The response is streamed incrementally via the callback function.
  /// @param llm_model_config The Language Model and its config to be used for
  /// response generation
  /// @param query The input query/prompt to generate a response for
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text. Can
  /// return false to cancel streaming.
  /// @param user_data User-provided data passed to the callback function
  /// @return Total number of tokens generated (excluding EOG token), or -1 on
  /// error
  int32_t generate_streaming_response(const LLMModelConfig& llm_model_config, const vector<InputItem>& prompt,
                                      const SamplerConfig& sampler_config, OdaiStreamRespCallbackFn callback,
                                      void* user_data);

  /// Loads a chat from the database to backend engine using the provided chat
  /// ID. Retrieves the chat configuration and loads the appropriate language
  /// model. If the chat uses RAG, the embedding model will be loaded.
  /// @param chat_id Unique identifier for the chat session to load
  /// @return true if chat session loaded successfully, false if chat_id not
  /// found or model loading failed
  bool load_chat_session(const ChatId& chat_id);

  /// Generates a streaming response for the given query for the given chat.
  /// Uses the previously loaded chat if cached, else will load chat and then
  /// generates a response. If RAG is enabled for the chat, retrieves relevant
  /// context from the knowledge base.
  /// @param chat_id Unique identifier for the chat session
  /// @param query The input query/message to generate a response for
  /// @param generator_config (Sampler, RAG settings, etc.)
  /// @param scope_id Scope identifier to filter documents during RAG retrieval
  /// (ignored if RAG is disabled)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback function
  /// @return Total number of tokens generated (excluding EOG token), or -1 on
  /// error
  int32_t generate_streaming_chat_response(const ChatId& chat_id, const vector<InputItem>& prompt,
                                           const GeneratorConfig& generator_config, OdaiStreamRespCallbackFn callback,
                                           void* user_data);

  /// Unloads the chat session from memory, freeing up resources.
  /// @param chat_id Unique identifier for the chat session
  /// @return true if unloaded successfully, false on error
  bool unload_chat_session(const ChatId& chat_id);

  /// Creates a new semantic space for vector embeddings in the database.
  /// @param config The configuration for the semantic space to be created
  /// @return true if semantic space created successfully, false on error or if name already exists
  bool create_semantic_space(const SemanticSpaceConfig& config);

  /// Retrieves the configuration of an existing semantic space by name.
  /// @param name The name of the semantic space to retrieve
  /// @param config Output parameter populated with the semantic space configuration
  /// @return true if successful, false on error or if name not found
  bool get_semantic_space_config(const SemanticSpaceName& name, SemanticSpaceConfig& config);

  /// Retrieves a list of all existing semantic spaces in the database.
  /// @param spaces Output parameter populated with configurations of all semantic spaces
  /// @return true if successful, false on error
  bool list_semantic_spaces(vector<SemanticSpaceConfig>& spaces);

  /// Deletes an existing semantic space and its associated data from the database.
  /// @param name The name of the semantic space to delete
  /// @return true if deleted successfully, false on error or if name not found
  bool delete_semantic_space(const SemanticSpaceName& name);

  /// Creates a new chat session in the database with the provided identifier and configuration.
  /// @param chat_id Unique identifier for the new chat session
  /// @param chat_config Configuration parameters for the chat session
  /// @return true if chat created successfully, false on error or if chat_id already exists
  bool create_chat(const ChatId& chat_id, const ChatConfig& chat_config);

  /// Retrieves the message history for a given chat session from the database.
  /// @param chat_id Unique identifier for the chat session
  /// @param messages Output parameter populated with the chronological sequence of chat messages
  /// @return true if successful, false on error or if chat_id not found
  bool get_chat_history(const ChatId& chat_id, vector<ChatMessage>& messages);

  /// Checks if a chat session with the specified identifier already exists in the database.
  /// @param chat_id Unique identifier for the chat session to check
  /// @return true if the chat_id exists, false otherwise
  bool chat_id_exists(const ChatId& chat_id);

private:
  /// Resolves the file system path for a given model name using cache or
  /// database.
  /// @param modelName The name of the model.
  /// @param details Output parameter for the resolved paths.
  /// @return true if found, false otherwise.
  bool resolve_model_files(const ModelName& model_name, ModelFiles& details);

  /// Helper to ensure chat session is loaded into memory (backend engine
  /// context). If not loaded, it retrieves history from DB and loads it.
  /// @param chat_id Unique identifier for the chat session
  /// @param chat_config Configuration for the chat
  /// @return true if session is loaded (or was already loaded), false on error
  bool ensure_chat_session_loaded(const ChatId& chat_id, const ChatConfig& chat_config);

  /// Helper to process multimodal inputs and decode/load files into memory buffer
  /// before sending them to the inference engine. This mutates the input items,
  /// converting formats like AUDIO_FILE into an internal processed data format.
  /// Thus, the returned / mutated prompt contains decoded/processed data meant exclusively
  /// for the backend engine and should NOT be persisted to the database.
  /// @param prompt Items to process.
  /// @param llm_model_config The target LlmModelConfig used.
  bool process_multimodal_inputs(vector<InputItem>& prompt_out, const LLMModelConfig& llm_model_config);

  std::unique_ptr<IOdaiDb> m_db;
  std::unique_ptr<IOdaiBackendEngine> m_backendEngine;
  std::unique_ptr<IOdaiAudioDecoder> m_audioDecoder;

  unordered_map<string, ModelFiles> m_modelDetailsCache;
};