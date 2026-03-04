#pragma once

#include "types/odai_types.h"
#include <vector>

/// Abstract base class for backend engines that handle model loading and text generation.
class IOdaiBackendEngine
{
public:
  IOdaiBackendEngine() = default;

  IOdaiBackendEngine(const IOdaiBackendEngine&) = delete;
  IOdaiBackendEngine& operator=(const IOdaiBackendEngine&) = delete;
  IOdaiBackendEngine(IOdaiBackendEngine&&) = delete;
  IOdaiBackendEngine& operator=(IOdaiBackendEngine&&) = delete;

  /// Initializes the backend engine. Must be called before loading models.
  /// @return true if initialization succeeded, false otherwise
  virtual bool initialize_engine() = 0;

  /// Validates the given model registration paths for this specific backend engine.
  /// @param files The generic model paths to validate.
  /// @return true if the files contain required paths and are valid.
  virtual bool validate_model_files(const ModelFiles& files) const = 0;

  /// Loads an embedding model from the specified configuration.
  /// If a model is already loaded, it will be freed and replaced with the new one.
  /// @param files The Model Files struct.
  /// @param config Configuration containing parameters.
  /// @return true if model loaded successfully, false otherwise
  virtual bool load_embedding_model(const ModelFiles& files, const EmbeddingModelConfig& config) = 0;

  /// Loads a language model from the specified configuration.
  /// If a model is already loaded, it will be freed and replaced with the new one.
  /// @param files The Model Files struct.
  /// @param config Configuration containing parameters.
  /// @return true if model loaded successfully, false otherwise
  virtual bool load_language_model(const ModelFiles& files, const LLMModelConfig& config) = 0;

  /// Generates a streaming response for the given prompt using the loaded language model.
  /// The response is streamed incrementally via the callback function.
  /// @param prompt The input prompt to generate a response for
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return Total number of tokens generated (excluding EOG token), or -1 on error
  virtual int32_t generate_streaming_response(const vector<InputItem>& prompt, const SamplerConfig& sampler_config,
                                              OdaiStreamRespCallbackFn callback, void* user_data) = 0;

  /// Loads the provided sequence of chat messages into the model's context for the specified chat session.
  /// This will compute the KV cache (key-value memory for transformer inference) and keep it in memory,
  /// so future generations for the same chat can use the existing context efficiently.
  /// If the chat context is already loaded and cached for the given chat_id, this function will return immediately.
  /// @param chat_id Unique identifier for the chat session to load context for
  /// @param messages Vector of chat messages (in order) to load into the context
  /// @return true if the context was successfully loaded or already cached, false if there was an error
  virtual bool load_chat_messages_into_context(const ChatId& chat_id, const vector<ChatMessage>& messages) = 0;

  /// Generates a streaming chat response for the given query in the given chat session.
  /// This function expects the chat context is already loaded into the model's context using
  /// load_chat_messages_into_context.
  /// @param chat_id Unique identifier for the chat session whose cached context will be used
  /// @param query The input query/message to generate a response for
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return Total number of tokens generated (excluding EOG token), or -1 on error
  virtual int32_t generate_streaming_chat_response(const ChatId& chat_id, const vector<InputItem>& prompt,
                                                   const SamplerConfig& sampler_config,
                                                   OdaiStreamRespCallbackFn callback, void* user_data) = 0;

  /// Checks if the context for a specific chat session is currently loaded in memory.
  /// @param chat_id Unique identifier for the chat session
  /// @return true if context is loaded, false otherwise
  virtual bool is_chat_context_loaded(const ChatId& chat_id) = 0;

  /// Unloads the context for a specific chat session from memory, freeing resources.
  /// @param chat_id Unique identifier for the chat session
  /// @return true if unloaded successfully (or was not loaded), false on error
  virtual bool unload_chat_context(const ChatId& chat_id) = 0;

  virtual ~IOdaiBackendEngine() = default;
};