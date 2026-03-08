#pragma once

#include "types/odai_types.h"
#include <vector>

/// Abstract base class for backend engines that handle model loading and text generation.
class IOdaiBackendEngine
{
protected:
  BackendEngineConfig m_backendEngineconfig{};

public:
  IOdaiBackendEngine(const BackendEngineConfig& config)
  {
    if (!config.is_sane())
    {
      throw std::invalid_argument("Invalid BackendEngineConfig provided");
    }

    m_backendEngineconfig = config;
  };

  IOdaiBackendEngine(const IOdaiBackendEngine&) = delete;
  IOdaiBackendEngine& operator=(const IOdaiBackendEngine&) = delete;
  IOdaiBackendEngine(IOdaiBackendEngine&&) = delete;
  IOdaiBackendEngine& operator=(IOdaiBackendEngine&&) = delete;

  /// Initializes the backend engine. Must be called at the very first before calling any other function in this layer
  /// @return true if initialization succeeded, false otherwise
  virtual bool initialize_engine() = 0;

  /// Returns the required audio specification for a specific language model.
  /// @param config The LLM model configuration to check requirements for.
  /// @param files The associated model files containing text model and multimodal projectors.
  /// @return The required audio spec if the model supports processing audio, std::nullopt otherwise.
  virtual std::optional<OdaiAudioTargetSpec> get_required_audio_spec(const LLMModelConfig& config,
                                                                     const ModelFiles& files) = 0;

  /// Validates the given model registration paths for this specific backend engine.
  /// @note Implementations should document the expected model files in this function's documentation.
  /// @param files The generic model paths to validate.
  /// @return true if the files contain required paths and are valid.
  virtual bool validate_model_files(const ModelFiles& files) = 0;

  /// Generates a streaming response for the given prompt
  /// The response is streamed incrementally via the callback function.
  /// Note: The engine expects the input items in the prompt to be of type PROCESSED_DATA
  /// (e.g., pre-decoded audio or raw buffers) rather than file paths.
  /// @param prompt The input prompt to generate a response for
  /// @param llm_model_config The LLM model configuration to use for generation
  /// @param model_files The model files to use for generation
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return Total number of tokens generated (excluding EOG token), or -1 on error
  virtual int32_t generate_streaming_response(const vector<InputItem>& prompt, const LLMModelConfig& llm_model_config,
                                              const ModelFiles& model_files, const SamplerConfig& sampler_config,
                                              OdaiStreamRespCallbackFn callback, void* user_data) = 0;

  /// Generates a streaming chat response for the given query and given chat history.
  /// @note The engine expects the input items in the prompt to be of type PROCESSED_DATA
  /// (e.g., pre-decoded audio buffers) rather than file paths.
  /// @param prompt The input query/message to generate a response for
  /// @param chat_history chat_history of the chat
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return Total number of tokens generated (excluding EOG token), or -1 on error
  virtual int32_t generate_streaming_chat_response(const vector<InputItem>& prompt,
                                                   const vector<ChatMessage>& chat_history,
                                                   const SamplerConfig& sampler_config,
                                                   OdaiStreamRespCallbackFn callback, void* user_data) = 0;

  virtual ~IOdaiBackendEngine() = default;
};