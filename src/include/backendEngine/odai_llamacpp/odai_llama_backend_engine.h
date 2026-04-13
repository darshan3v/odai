#pragma once

#ifdef ODAI_ENABLE_LLAMA_BACKEND

#include "backendEngine/odai_backend_engine.h"

#include "ggml-backend.h"
#include "types/odai_common_types.h"
#include "types/odai_result.h"
#include "types/odai_types.h"
#include <llama.h>
#include <memory>
#include <mtmd.h>
#include <vector>

struct LlamaModelDeleter
{
  void operator()(llama_model* ptr) const
  {
    if (ptr != nullptr)
    {
      llama_model_free(ptr);
    }
  }
  LlamaModelDeleter() = default; // explicitly default constructor
};

struct LlamaContextDeleter
{
  void operator()(llama_context* ptr) const
  {
    if (ptr != nullptr)
    {
      llama_free(ptr);
    }
  }
  LlamaContextDeleter() = default; // explicitly default constructor
};

struct LlamaSamplerDeleter
{
  void operator()(llama_sampler* ptr) const
  {
    if (ptr != nullptr)
    {
      llama_sampler_free(ptr);
    }
  }
  LlamaSamplerDeleter() = default; // explicitly default constructor
};

struct LlamaBatchDeleter
{
  void operator()(llama_batch* ptr) const
  {
    if (ptr != nullptr)
    {
      llama_batch_free(*ptr);
    }
  }
  LlamaBatchDeleter() = default; // explicitly default constructor
};

struct MtmdContextDeleter
{
  void operator()(mtmd_context* ptr) const
  {
    if (ptr != nullptr)
    {
      mtmd_free(ptr);
    }
  }
  MtmdContextDeleter() = default; // explicitly default constructor
};

/// Llama.cpp-based implementation of the backend engine for model loading and
/// text generation. Currenntly For LLM's supports Decoder-only LLMs.
class OdaiLlamaEngine : public IOdaiBackendEngine
{
public:
  OdaiLlamaEngine(const BackendEngineConfig& backend_engine_config);

  /// Initializes the llama backend engine, configures logging, and discovers hardware context.
  /// It selects devices according to the AUTO selection logic or explicit configurations.
  /// @return empty expected if initialization succeeded, or an unexpected OdaiResultEnum indicating the error
  OdaiResult<void> initialize_engine() override;

  /// Returns the primary execution path candidates (e.g., discrete or integrated GPUs)
  /// selected based on platform and user preference.
  OdaiResult<std::vector<BackendDevice>> get_candidate_devices() override;

  /// Validates the model files specifically for the llama.cpp backend engine.
  /// Ensures that the engine type is LLAMA_BACKEND_ENGINE.
  /// For LLM models, expects at most 2 entries: a mandatory 'base_model_path' and an optional 'mmproj_model_path', both
  /// pointing to valid existing files. For EMBEDDING models, expects exactly 1 entry: a mandatory 'base_model_path'
  /// pointing to a valid existing file.
  /// @param files The model files object to validate
  /// @return true/false on success, or an unexpected OdaiResultEnum indicating an operational failure
  OdaiResult<bool> validate_model_files(const ModelFiles& files) override;

  /// Generates a streaming response for the given prompt
  /// @note engine expects the input media items to be of type File Path, and text as Memory Buffer
  /// @param prompt The input prompt to generate a response for
  /// @param llm_model_config The LLM model configuration to use for generation
  /// @param model_files The model files to use for generation
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return streaming stats on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<StreamingStats> generate_streaming_response(const std::vector<InputItem>& prompt,
                                                         const LLMModelConfig& llm_model_config,
                                                         const ModelFiles& model_files,
                                                         const SamplerConfig& sampler_config,
                                                         OdaiStreamRespCallbackFn callback, void* user_data) override;

  /// Generates a streaming chat response for the given query and chat history.
  /// @note The engine expects the input media items in the prompt to be of type File Path, and text as Memory Buffer
  /// @param prompt The input query/message to generate a response for
  /// @param chat_history chat_history of the chat
  /// @param llm_model_config The LLM model configuration to use for generation
  /// @param model_files The model files to use for generation
  /// @param sampler_config Configuration for the sampler (top_k, top_p, etc.)
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return streaming stats on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<StreamingStats>
  generate_streaming_chat_response(const std::vector<InputItem>& prompt, const std::vector<ChatMessage>& chat_history,
                                   const LLMModelConfig& llm_model_config, const ModelFiles& model_files,
                                   const SamplerConfig& sampler_config, OdaiStreamRespCallbackFn callback,
                                   void* user_data) override;

  /// Destructor that frees the llama backend resources.
  /// @note llama.cpp backends loaded via ggml_backend_load_all() are NOT intended to be unloaded
  /// manually during application lifecycle. Unloading graphics/compute DLLs mid-execution is
  /// inherently risky and can lead to driver-level instability.
  ~OdaiLlamaEngine() override;

private:
  struct CandidateDeviceRecord
  {
    BackendDevice m_info{};
    ggml_backend_dev_t m_handle = nullptr;
  };

  struct DeviceInventory
  {
    std::vector<CandidateDeviceRecord> m_candidates;
    BackendDeviceType m_requestedType = BackendDeviceType::AUTO;
    bool m_hasAccelerationCandidate = false;
  };

  enum class PlacementMode
  {
    CPU_ONLY,
    ACCELERATED_FULL,
    ACCELERATED_PARTIAL
  };

  struct LlmLoadPlan
  {
    PlacementMode m_mode = PlacementMode::CPU_ONLY;
    std::vector<size_t> m_selectedCandidateIndices;
    bool m_shouldUseMlock = false;
    bool m_shouldUseMmap = true;
    int32_t m_nGpuLayers = 0;
    llama_split_mode m_splitMode = LLAMA_SPLIT_MODE_NONE;
    bool m_allowCpuRetry = false;
    std::string m_reason;
  };

  struct GpuSelectionResult
  {
    std::vector<size_t> m_candidateIndices;
    bool m_hasFullFit = false;
  };

  struct PreparedLlmModelParams
  {
    llama_model_params m_params = llama_model_default_params();
    std::vector<ggml_backend_dev_t> m_selectedDevices;
  };

  struct LoadedLanguageModelState
  {
    std::unique_ptr<llama_model, LlamaModelDeleter> m_model = nullptr;
    // managed automatically by llama.cpp
    // so no need of unique_ptr
    const llama_vocab* m_vocab = nullptr;
    std::unique_ptr<mtmd_context, MtmdContextDeleter> m_mtmdContext = nullptr;
    LLMModelConfig m_config{};
    ModelFiles m_files{};

    void clear() { *this = {}; }

    bool is_loaded() const { return (m_model != nullptr) && (m_vocab != nullptr); }

    bool matches(const ModelFiles& files, const LLMModelConfig& config) const
    {
      return is_loaded() && (m_files == files) && (m_config == config);
    }
  };

  bool m_isInitialized{false};

  DeviceInventory m_deviceInventory{};

  EmbeddingModelConfig m_embeddingModelConfig{};

  ModelFiles m_embeddingModelFiles{};

  std::unique_ptr<llama_model, LlamaModelDeleter> m_embeddingModel = nullptr;
  LoadedLanguageModelState m_loadedLlmState{};

  /// Registers ggml backends, then discovers candidate devices according to ODAI's runtime policy.
  /// @param preferred_type The desired device preference (AUTO, GPU, IGPU, CPU)
  /// @return ODAI_SUCCESS on success, or ODAI_INTERNAL_ERROR if strict hardware requirements are not met.
  OdaiResult<void> discover_candidate_devices(BackendDeviceType preferred_type);

  /// Registers all ggml backend families found in the runtime backend directory.
  /// ggml internally scores backend variants and loads the best match for each family.
  /// @return true if the registration step completed, false otherwise.
  static bool register_available_backends();

  /// Helper to collect devices for a registered backend family and accepted device types.
  /// If found, those devices are added to the inventory.
  /// @param backend_name The GGML backend name (e.g., "cuda", "vulkan", "cpu")
  /// @param accepted_types The device types to accept for this probe
  /// @return true if the backend provides at least one matching device.
  bool append_backend_candidate_devices(const std::string& backend_name,
                                        const std::vector<BackendDeviceType>& accepted_types);

  /// Plans base LLM placement from the discovered inventory and current platform policy.
  /// The returned plan explains why ODAI chose CPU-only, full acceleration, or best-effort partial offload.
  /// @param model_file_size_bytes Size of the GGUF file on disk.
  /// @return Placement plan for the next base language model load.
  LlmLoadPlan plan_llm_load(uint64_t model_file_size_bytes) const;

  /// Selects discrete GPUs for desktop llama.cpp offload.
  /// If the model estimate fits, returns the minimum subset whose combined free VRAM is sufficient.
  /// Otherwise returns all discovered dGPUs so llama.cpp can still offload layers best-effort.
  /// @param estimated_model_bytes Estimated VRAM needed for the base model load.
  /// @return Selected dGPU indices plus whether the returned subset satisfies the estimate.
  GpuSelectionResult select_minimum_gpu_subset(uint64_t estimated_model_bytes) const;

  /// Queries the current free memory reported by a ggml backend device.
  /// @param backend_handle The ggml device handle to query.
  /// @return Fresh free memory in bytes, or std::nullopt if the handle is invalid.
  static std::optional<uint64_t> get_device_free_memory_bytes(ggml_backend_dev_t backend_handle);

  /// Produces a CPU-only load plan with explicit llama.cpp placement fields set.
  /// @param model_file_size_bytes Size of the GGUF file on disk.
  /// @param reason Human-readable explanation for choosing CPU-only placement.
  /// @return CPU-only load plan.
  LlmLoadPlan make_cpu_only_llm_plan(uint64_t model_file_size_bytes, std::string reason) const;

  /// Materializes llama.cpp model params and the temporary NULL-terminated device buffer for one load plan.
  /// @param llm_load_plan Placement plan to convert into llama.cpp model params.
  /// @return Materialized params and the temporary device-handle buffer needed by llama_model_load_from_file().
  OdaiResult<PreparedLlmModelParams> build_llm_model_params(const LlmLoadPlan& llm_load_plan) const;

  /// Attempts to load the base LLM and optional multimodal projector for a single plan into transaction-local state.
  /// @param base_model_path Path to the base GGUF model.
  /// @param mmproj_model_path Optional multimodal projector path.
  /// @param llm_load_plan Placement plan to execute.
  /// @return Transaction-local loaded state populated on success.
  OdaiResult<LoadedLanguageModelState>
  try_load_language_model_for_plan(const std::string& base_model_path,
                                   const std::optional<std::string>& mmproj_model_path,
                                   const LlmLoadPlan& llm_load_plan) const;

  /// Validates if a specific entry key exists in the model files and points to a valid file on the filesystem.
  /// @param entries The map of model file entries
  /// @param key The entry key to validate (e.g., "base_model_path")
  /// @param is_optional If true, the entry is not required. If present and not empty, it must be a valid file.
  /// @return true if the entry is valid or successfully omitted, false otherwise
  static bool validate_model_file_entry(const std::unordered_map<std::string, std::string>& entries,
                                        const std::string& key, bool is_optional);

  /// Checks if the given input data is valid / supported.
  /// @param input_items The input items to validate against the model capabilities.
  /// @param model_files The model files containing mmproj path.
  /// @return empty result when the data is valid, or an unexpected OdaiResultEnum on failure.
  OdaiResult<void> does_model_support_input_data(const std::vector<InputItem>& input_items,
                                                 const LLMModelConfig& llm_model_config, const ModelFiles& model_files);

  /// Returns the required audio specification for the given GGUF model.
  /// @param config The LLM model configuration to check requirements for.
  /// @param files The associated model files containing text model and multimodal projectors.
  OdaiResult<OdaiAudioTargetSpec> get_required_audio_spec(const LLMModelConfig& config, const ModelFiles& files) const;

  /// Loads an embedding model from the specified configuration.
  /// If the same model is already loaded, only updates the configuration.
  /// @param files The generic model files containing paths.
  /// @param config Configuration containing parameters.
  /// @return empty result on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<void> load_embedding_model(const ModelFiles& files, const EmbeddingModelConfig& config);

  /// Loads a language model from the specified configuration.
  /// If the same model is already loaded, only updates the configuration.
  /// @param files The generic model files containing paths.
  /// @param config Configuration containing parameters.
  /// @return empty result on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<void> load_language_model(const ModelFiles& files, const LLMModelConfig& config);

  /// Creates a new llama context for the specified model type.
  /// @param model_type Type of model (LLM or EMBEDDING) to create context for
  /// @return Unique pointer to the context, or nullptr on error
  std::unique_ptr<llama_context, LlamaContextDeleter> get_new_llama_context(ModelType model_type);

  /// Creates a new sampler chain for language model generation.
  /// The sampler uses top-k, top-p, and greedy sampling strategies from the
  /// config.
  /// @param config Configuration for the sampler
  /// @return Unique pointer to the sampler, or nullptr on error
  static std::unique_ptr<llama_sampler, LlamaSamplerDeleter> get_new_llm_llama_sampler(const SamplerConfig& config);

  /// Tokenizes an input string into llama tokens using the appropriate model.
  /// @param input The string to tokenize
  /// @param is_first Whether this is the first prompt
  /// @param model_type Type of model (LLM or EMBEDDING) to use for tokenization
  /// @return Tokens on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<std::vector<llama_token>> tokenize(const std::string& input, bool is_first, ModelType model_type) const;

  /// Adds tokens to a llama batch for processing. we set the logits request for
  /// the last token.
  /// @param tokens Vector of tokens to add
  /// @param batch The batch to add tokens to (modified in place)
  /// @param start_pos Starting position for tokens (incremented by number of
  /// tokens added) (0-indexed)
  /// @param seq_id Sequence ID to assign to the tokens
  /// @param set_logit_request_for_last_token Whether to request logits for the
  /// last token added
  static void add_tokens_to_batch(const std::vector<llama_token>& tokens, llama_batch& batch, uint32_t& start_pos,
                                  llama_seq_id seq_id, bool set_logit_request_for_last_token);

  /// Converts a vector of tokens back into a string.
  /// @param tokens Vector of tokens to detokenize
  /// @return Detokenized string on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<std::string> detokenize(const std::vector<llama_token>& tokens) const;

  /// Loads the given prompt into the provided llama context.
  /// If the context already has some other context (Eg: some message hisotry)
  /// then we don't clear it. we just append
  /// @param model_context Language Model context to load the prompt into
  /// @param prompt The prompt string to load
  /// @param request_logits_for_last_token Whether to request logits for the
  /// last token in the prompt, (do it if you are generating next token after
  /// this)
  /// @return Next context position on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<uint32_t> load_into_context(llama_context& model_context, const std::string& prompt,
                                         bool request_logits_for_last_token);

  /// Loads the given tokens into the provided llama context.
  /// If the context already has some other context (Eg: some message hisotry)
  /// then we don't clear it. we just append
  /// @param model_context Language Model context to load the tokens into
  /// @param tokens Vector of tokens to load
  /// @param request_logits_for_last_token Whether to request logits for the
  /// last token, (do it if you are generating next token after this)
  /// @return Next context position on success, or an unexpected OdaiResultEnum on failure.
  static OdaiResult<uint32_t> load_into_context(llama_context& model_context, const std::vector<llama_token>& tokens,
                                                bool request_logits_for_last_token);

  /// Loads the given prompt string and its accompanying mtmd bitmaps into the provided llama context.
  /// Handles chunking logic with mtmd internally if there are multimodal bitmaps.
  /// @param model_context Language Model context
  /// @param prompt The prompt string to load
  /// @param bitmaps The vector of extracted mtmd bitmaps
  /// @param request_logits_for_last_token Whether to request logits for the last token
  /// @return Next context position on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<uint32_t> load_into_context(llama_context& model_context, const std::string& prompt,
                                         const std::vector<mtmd::bitmap>& bitmaps, bool request_logits_for_last_token);

  /// Helper function that performs the common logic for loading tokens into
  /// context.
  /// @param model_context Language Model context to load tokens into
  /// @param tokens Vector of tokens to load
  /// @param request_logits_for_last_token Whether to request logits for the
  /// last token
  /// @return Next context position on success, or an unexpected OdaiResultEnum on failure.
  static OdaiResult<uint32_t> load_tokens_into_context_impl(llama_context& model_context,
                                                            const std::vector<llama_token>& tokens,
                                                            bool request_logits_for_last_token);

  /// Loads the provided sequence of chat messages into the model's context
  /// @param messages Vector of chat messages (in order) to load into the
  /// context
  /// @return Loaded llama context on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<std::unique_ptr<llama_context, LlamaContextDeleter>>
  load_chat_messages_into_context(const std::vector<ChatMessage>& messages);

  /// Generates the next token using the provided llama context and sampler.
  /// @param model_context Language Model context (has KV cache of old tokens
  /// and other stuff) to use for generation
  /// @param sampler sampler to use for token sampling
  /// @param append_to_context Whether to append the generated token to the
  /// context's memory
  /// @return Generated token on success, or an unexpected OdaiResultEnum on failure.
  static OdaiResult<llama_token> generate_next_token(llama_context& model_context, llama_sampler& sampler,
                                                     bool append_to_context);

  /// Processes buffered tokens into a UTF-8 safe string for streaming.
  /// Detokenizes tokens, appends to output buffer, then splits at a safe UTF-8
  /// boundary. Prevents sending incomplete multi-byte characters to the client.
  /// @param buffered_tokens Tokens to process (cleared after processing)
  /// @param output_buffer Accumulated output buffer (unsafe tail remains after
  /// return)
  /// @return Safe UTF-8 string on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<std::string> flush_utf8_safe_output(std::vector<llama_token>& buffered_tokens, std::string& output_buffer);

  /// Formats a vector of chat messages into a single prompt string using the
  /// model's chat template. Uses llama_chat_apply_template to apply the model's
  /// configured chat template format.
  /// @param messages Vector of chat messages to format
  /// @param add_generation_prompt Whether to add generation prompt (set to true
  /// when expecting model response)
  /// @return Formatted prompt string on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<std::string>
  format_chat_messages_to_prompt(const std::vector<std::pair<std::string, std::string>>& messages,
                                 bool add_generation_prompt) const;

  /// Processes input items to extract formatted text and multimodal bitmaps.
  /// @todo Replace current mtmd image decoding logic with our custom internal image decoder
  /// @param items The input items to process
  /// @return Formatted text and extracted mtmd bitmaps on success, or an unexpected OdaiResultEnum on failure.
  OdaiResult<std::pair<std::string, std::vector<mtmd::bitmap>>>
  process_input_items(const std::vector<InputItem>& items);

  /// Core implementation of streaming response generation that handles token
  /// generation and buffering. Takes an already-initialized context and sampler
  /// to perform the streaming.
  /// Handles both text-only and multimodal input: if media items are present,
  /// uses the mtmd pipeline (tokenize → encode chunks → decode embeddings).
  /// @param model_context Llama context with KV cache and token state
  /// @param sampler Sampler chain for token sampling
  /// @param input_items The input items (text, audio, image) to process
  /// @param callback Function called for each chunk of generated text
  /// @param user_data User-provided data passed to the callback
  /// @return streaming stats on success, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<StreamingStats> generate_streaming_response_impl(llama_context& model_context, llama_sampler& sampler,
                                                              const std::string& prompt,
                                                              const std::vector<mtmd::bitmap>& bitmaps,
                                                              OdaiStreamRespCallbackFn callback, void* user_data);
};

#endif // ODAI_ENABLE_LLAMA_BACKEND
