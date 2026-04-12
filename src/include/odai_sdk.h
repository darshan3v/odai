#pragma once

#include <memory>
#include <string>
#include <vector>

#include "odai_logger.h"
#include "types/odai_result.h"
#include "types/odai_types.h"

// Forward declarations to reduce header dependency surface
class IOdaiDb;
class OdaiRagEngine;
class IOdaiAudioDecoder;
class IOdaiImageDecoder;
struct DBConfig;
struct BackendEngineConfig;

/// C++ Entry point for ODAI SDK
class OdaiSdk
{
public:
  /// Get the singleton instance of the SDK
  static OdaiSdk& get_instance();

  /// Prevent copying and assignment
  OdaiSdk(const OdaiSdk&) = delete;
  OdaiSdk& operator=(const OdaiSdk&) = delete;

  /// Sets a custom logging callback function for receiving log messages.
  /// @param callback Function to call for each log message, or nullptr to
  /// disable custom logging
  /// @param user_data User-provided data pointer that will be passed to the
  /// callback function
  void set_logger(OdaiLogCallbackFn callback, void* user_data);

  /// Sets the minimum log level for messages to be processed.
  /// Only messages at or below this level will be logged or passed to the
  /// callback.
  /// @param log_level Minimum log level (OdaiLogLevel)
  void set_log_level(OdaiLogLevel log_level);

  /// Initializes the SDK with database and backend engine configurations.
  /// Must be called before using any other SDK functions.
  /// @param dbConfig Configuration structure containing the database type and
  /// path
  /// @param backendConfig Configuration structure specifying which backend
  /// engine to use
  /// @return empty expected if initialization succeeded, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> initialize_sdk(const DBConfig& db_config, const BackendEngineConfig& backend_config);

  /// Shuts down the SDK and releases owned engines before process teardown.
  /// This call is idempotent and should be used by consumers when GPU-backed resources are active.
  /// @return empty expected if shutdown succeeded, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> shutdown();

  /// Registers a new model with generic files.
  /// @param name The unique name of the model.
  /// @param files The Model Files struct containing paths.
  /// @return empty expected if registered successfully, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> register_model_files(const ModelName& name, const ModelFiles& files);

  /// Updates the registration files for a model.
  /// Only the files explicitly provided in `files` will be updated and validated.
  /// Existing registrations for other files belonging to the model remain untouched.
  /// @param name The name of the model.
  /// @param files The Model Files struct containing paths to update.
  /// @param flag Flag indicating how to handle checksum changes.
  /// @return empty expected if updated successfully, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> update_model_files(const ModelName& name, const ModelFiles& files, UpdateModelFlag flag);

  /// Creates a new semantic space.
  /// @param config The configuration for the semantic space.
  /// @return empty expected if created successfully, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> create_semantic_space(const SemanticSpaceConfig& config);

  /// Retrieves the configuration for a semantic space.
  /// @param name The name of the semantic space to retrieve.
  /// @return semantic space configuration on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<SemanticSpaceConfig> get_semantic_space_config(const SemanticSpaceName& name);

  /// Lists all available semantic spaces.
  /// @return semantic space configurations on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<std::vector<SemanticSpaceConfig>> list_semantic_spaces();

  /// Deletes a semantic space.
  /// @param name The name of the semantic space to delete.
  /// @return empty expected if deleted successfully, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> delete_semantic_space(const SemanticSpaceName& name);

  /// Adds a document to the RAG knowledge base for retrieval during generation.
  /// @param content The text content of the document to add
  /// @param documentId Unique identifier for this document
  /// @param semanticSpaceName Name of the semantic space to use
  /// @param scopeId Scope identifier to group documents
  /// @return empty expected if the document was added successfully, or an unexpected OdaiResultEnum indicating the
  /// error.
  OdaiResult<void> add_document(const std::string& content, const DocumentId& document_id,
                                const SemanticSpaceName& semantic_space_name, const ScopeId& scope_id) const;

  /// Generates a streaming response for the given query.
  /// Its like a Completion API, and won't use RAG
  /// @param llmModelConfig The Language Model and its config to be used for
  /// response generation
  /// @param prompt The input query/prompt
  /// @param samplerConfig Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each generated token
  /// @param userData User-provided data pointer passed to the callback function
  /// @return streaming stats on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<StreamingStats> generate_streaming_response(const LLMModelConfig& llm_model_config,
                                                         const std::vector<InputItem>& prompt,
                                                         const SamplerConfig& sampler_config,
                                                         OdaiStreamRespCallbackFn callback, void* user_data);

  /// Creates a new chat session with the specified configuration.
  /// @param chatIdIn Input chat ID (empty to auto-generate)
  /// @param chatConfig Configuration structure defining chat behavior
  /// @return final chat ID on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<ChatId> create_chat(const ChatId& chat_id_in, const ChatConfig& chat_config);

  /// Retrieves all chat messages for the specified chat session.
  /// @param chatId The chat identifier to retrieve messages for
  /// @return chronological chat messages on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<std::vector<ChatMessage>> get_chat_history(const ChatId& chat_id);

  /// Generates a streaming chat response for the given query in the specified
  /// chat session. It will load  languagde model mentioned in chat config and
  /// load the chat history into context and then input the query and generate
  /// response
  /// @param chatId The unique identifier of the chat session
  /// @param prompt The input query/message
  /// @param generatorConfig Configuration for the generator (Sampler, RAG
  /// settings, etc.)
  /// @param callback Function called for each generated token
  /// @param userData User-provided data pointer passed to the callback function
  /// @return streaming stats on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<StreamingStats> generate_streaming_chat_response(const ChatId& chat_id,
                                                              const std::vector<InputItem>& prompt,
                                                              const GeneratorConfig& generator_config,
                                                              OdaiStreamRespCallbackFn callback, void* user_data);

private:
  OdaiSdk();
  ~OdaiSdk();

  bool m_sdkInitialized = false;
  std::unique_ptr<OdaiLogger> m_logger;
  std::unique_ptr<OdaiRagEngine> m_ragEngine;

public:
  OdaiLogger* get_logger() { return m_logger.get(); }

  /// @return returns a new AudioDecoder instance that the library was built with, if it was built with none then
  /// returns nullptr
  static std::unique_ptr<IOdaiAudioDecoder> get_new_odai_audio_decoder_instance();

  /// @return returns a new ImageDecoder instance that the library was built with, if it was built with none then
  /// returns nullptr
  static std::unique_ptr<IOdaiImageDecoder> get_new_odai_image_decoder_instance();
};
