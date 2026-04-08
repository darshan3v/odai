#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include "audioEngine/odai_audio_decoder.h"
#include "backendEngine/odai_llamacpp/odai_llama_backend_engine.h"
#include "backendEngine/odai_llamacpp/odai_llama_type_conversions.h"
#include "imageEngine/odai_image_decoder.h"

#include "odai_sdk.h"

#include "types/odai_common_types.h"
#include "types/odai_type_conversions.h"
#include "types/odai_types.h"
#include "utils/string_utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
#include <utility>
#include <vector>

/// Redirects llama.cpp log messages to the Odai logging system.
/// Maps GGML log levels to Odai log levels and filters out debug messages.
/// @param level The GGML log level
/// @param text The log message text
/// @param user_data User data (unused)
static void llama_log_redirect(ggml_log_level level, const char* text, void* user_data)
{
  // Map llama levels to Odai levels
  OdaiLogLevel our_level = ODAI_LOG_INFO;

  if (level == GGML_LOG_LEVEL_ERROR)
  {
    our_level = ODAI_LOG_ERROR;
  }
  else if (level == GGML_LOG_LEVEL_WARN)
  {
    our_level = ODAI_LOG_WARN;
  }
  else if (level == GGML_LOG_LEVEL_INFO)
  {
    our_level = ODAI_LOG_INFO;
  }
  else
  {
    return; // Ignore debug/spam from llama if you want
  }

  ODAI_LOG(our_level, "[llama.cpp] {}", text);
}

OdaiLlamaEngine::OdaiLlamaEngine(const BackendEngineConfig& backend_engine_config)
    : IOdaiBackendEngine(backend_engine_config)
{
  if (backend_engine_config.m_engineType != LLAMA_BACKEND_ENGINE)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid BackendEngineConfig provided to OdaiLlamaEngine constructor");
    throw std::invalid_argument("Invalid BackendEngineConfig provided to OdaiLlamaEngine constructor");
  }
}

bool OdaiLlamaEngine::load_backend(const std::string& name)
{
  if (ggml_backend_reg_by_name(name.c_str()) != nullptr)
  {
    return true;
  }

  std::string filename;
#ifdef _WIN32
  filename = "ggml-" + name + ".dll";
#elif defined(__APPLE__)
  filename = "libggml-" + name + ".dylib";
#else
  filename = "libggml-" + name + ".so";
#endif

  ODAI_LOG(ODAI_LOG_INFO, "Attempting to load backend: {}", filename);
  ggml_backend_reg_t reg = ggml_backend_load(filename.c_str());
  if (reg != nullptr)
  {
    ODAI_LOG(ODAI_LOG_INFO, "Successfully loaded backend: {}", name);
    return true;
  }

  ODAI_LOG(ODAI_LOG_WARN, "Failed to load backend: {}", filename);
  return false;
}

OdaiResult<void> OdaiLlamaEngine::load_and_setup_candidate_devices(BackendDeviceType preferred_type)
{
  m_candidateDevices.clear();

  // 1. Always ensure CPU backend library is loaded (it's the universal fallback)
  if (!OdaiLlamaEngine::load_backend("cpu"))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Critical failure: CPU backend library not found");
    return tl::make_unexpected(OdaiResultEnum::INTERNAL_ERROR);
  }

  // 2. If CPU only requested, just select CPU device and return
  if (preferred_type == BackendDeviceType::CPU)
  {
    ODAI_LOG(ODAI_LOG_INFO, "CPU only mode requested; selecting CPU as primary execution path");
    this->try_load_and_add_candidate_devices("cpu", BackendDeviceType::CPU);
    return {};
  }

  // 3. Define prioritized GPU search plan based on platform
#ifdef __APPLE__
  const std::vector<std::string> gpu_backends = {"metal"};
#elif defined(__ANDROID__)
  const std::vector<std::string> gpu_backends = {"vulkan"};
#else
  const std::vector<std::string> gpu_backends = {"cuda", "hip", "sycl", "vulkan"};
#endif

  bool graphics_hardware_found = false;

  // 4. Try to satisfy Discrete GPU or AUTO preference
  if (preferred_type == BackendDeviceType::GPU || preferred_type == BackendDeviceType::AUTO)
  {
    for (const auto& backend_name : gpu_backends)
    {
      if (this->try_load_and_add_candidate_devices(backend_name, BackendDeviceType::GPU))
      {
        graphics_hardware_found = true;
        break; // Stop at the first backend that provides the requested hardware type
      }
    }
  }

  // 5. Try to satisfy Integrated GPU if GPU failed or explicitly requested
  if (!graphics_hardware_found &&
      (preferred_type == BackendDeviceType::IGPU || preferred_type == BackendDeviceType::AUTO))
  {
#ifdef __APPLE__
    // Apple Silicon treats the integrated GPU as the primary (and only) Metal device
    graphics_hardware_found = this->try_load_and_add_candidate_devices("metal", BackendDeviceType::IGPU);
#else
    graphics_hardware_found = this->try_load_and_add_candidate_devices("vulkan", BackendDeviceType::IGPU);
#endif
  }

  // 6. Handle AUTO fallback to CPU
  if (!graphics_hardware_found && preferred_type == BackendDeviceType::AUTO)
  {
    ODAI_LOG(ODAI_LOG_INFO, "No GPU acceleration found; primary execution will use CPU");
    this->try_load_and_add_candidate_devices("cpu", BackendDeviceType::CPU);
    graphics_hardware_found = true;
  }

  // 7. Strict validation for explicit GPU/IGPU requests
  if (!graphics_hardware_found &&
      (preferred_type == BackendDeviceType::GPU || preferred_type == BackendDeviceType::IGPU))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Requested {} device not found in discovered hardware",
             preferred_type == BackendDeviceType::GPU ? "GPU" : "IGPU");
    return tl::make_unexpected(OdaiResultEnum::NOT_FOUND);
  }

  if (graphics_hardware_found && preferred_type != BackendDeviceType::AUTO && preferred_type != BackendDeviceType::CPU)
  {
    ODAI_LOG(ODAI_LOG_INFO, "Successfully configured execution context with hardware acceleration");
  }

  return {};
}

bool OdaiLlamaEngine::try_load_and_add_candidate_devices(const std::string& backend_name, BackendDeviceType target_type)
{
  if (!OdaiLlamaEngine::load_backend(backend_name))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load backend library: {}", backend_name);
    return false;
  }

  bool found = false;
  size_t count = ggml_backend_dev_count();
  for (size_t i = 0; i < count; ++i)
  {
    ggml_backend_dev_t dev = ggml_backend_dev_get(i);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (std::string(ggml_backend_reg_name(reg)) == backend_name)
    {
      BackendDeviceType dev_type = to_odai_backend_device_type(ggml_backend_dev_type(dev));
      if (dev_type == target_type)
      {
        BackendDevice odai_dev;
        odai_dev.m_name = ggml_backend_dev_name(dev);
        odai_dev.m_description = ggml_backend_dev_description(dev);

        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        odai_dev.m_totalRam = total_mem;
        odai_dev.m_type = dev_type;

        m_candidateDevices.push_back(odai_dev);

        ODAI_LOG(ODAI_LOG_INFO, "Discovered Candidate: {} ({}) | Backend: {} | Type: {}", odai_dev.m_name,
                 odai_dev.m_description, backend_name, static_cast<uint8_t>(odai_dev.m_type));
        found = true;
      }
    }
  }
  return found;
}

OdaiResult<void> OdaiLlamaEngine::initialize_engine()
{
  llama_backend_init();

  OdaiResult<void> config_res = load_and_setup_candidate_devices(m_backendEngineconfig.m_preferredDeviceType);
  if (!config_res)
  {
    return config_res;
  }

  ODAI_LOG(ODAI_LOG_INFO, "Initialized llama backend and execution context");

  llama_log_set(llama_log_redirect, nullptr);
  mtmd_helper_log_set(llama_log_redirect, nullptr);

  this->m_isInitialized = true;
  return {};
}

OdaiResult<std::vector<BackendDevice>> OdaiLlamaEngine::get_candidate_devices()
{
  if (!this->m_isInitialized)
  {
    return tl::make_unexpected(OdaiResultEnum::NOT_INITIALIZED);
  }
  return m_candidateDevices;
}

bool OdaiLlamaEngine::does_model_support_input_data(const std::vector<InputItem>& items,
                                                    const LLMModelConfig& llm_model_config, const ModelFiles& files)
{
  std::unique_ptr<mtmd_context, MtmdContextDeleter> temp_ctx = nullptr;

  if (!this->load_language_model(files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load language model");
    return false;
  }

  bool supports_image = this->m_mtmdContext != nullptr ? mtmd_support_vision(this->m_mtmdContext.get()) : false;
  bool supports_audio = this->m_mtmdContext != nullptr ? mtmd_support_audio(this->m_mtmdContext.get()) : false;

  for (const InputItem& item : items)
  {
    MediaType media_type = item.get_media_type();

    if (media_type == MediaType::INVALID)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for prompt");
      return false;
    }

    if (media_type == MediaType::IMAGE && !supports_image)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model does not support image input");
      return false;
    }

    if (media_type == MediaType::AUDIO && !supports_audio)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Model does not support audio input");
      return false;
    }

    if (media_type == MediaType::TEXT && item.m_type != InputItemType::MEMORY_BUFFER)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Text input items must be provided as MEMORY_BUFFER");
      return false;
    }

    if ((media_type == MediaType::IMAGE || media_type == MediaType::AUDIO) && item.m_type != InputItemType::FILE_PATH)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Image / Audio input items must be provided as File Path");
      return false;
    }
  }

  return true;
}

std::unique_ptr<llama_context, LlamaContextDeleter> OdaiLlamaEngine::get_new_llama_context(ModelType model_type)
{
  std::unique_ptr<llama_context, LlamaContextDeleter> context = nullptr;
  llama_context_params context_params = llama_context_default_params();
  llama_model* model = nullptr;

  context_params.n_threads = 4;

  if (model_type == ModelType::LLM)
  {
    model = this->m_llmModel.get();
    context_params.n_ctx = 2048;
    context_params.embeddings = false;
  }
  else if (model_type == ModelType::EMBEDDING)
  {
    model = this->m_embeddingModel.get();
    context_params.n_ctx = 512;
    context_params.embeddings = true;
  }
  else
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid Model Type passed");
    return nullptr;
  }

  if (model == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Model not loaded yet hence can't create context");
    return nullptr;
  }

  context.reset(llama_init_from_model(model, context_params));
  return context;
}

std::unique_ptr<llama_sampler, LlamaSamplerDeleter>
OdaiLlamaEngine::get_new_llm_llama_sampler(const SamplerConfig& config)
{
  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler = nullptr;

  llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();

  sampler.reset(llama_sampler_chain_init(sampler_params));

  if (sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create llama sampler");
    return nullptr;
  }

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(config.m_topK));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(config.m_topP, 1));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

  return sampler;
}

bool OdaiLlamaEngine::validate_model_file_entry(const std::unordered_map<std::string, std::string>& entries,
                                                const std::string& key, bool is_optional)
{
  if (!entries.contains(key))
  {
    if (is_optional)
    {
      return true;
    }
    ODAI_LOG(ODAI_LOG_ERROR, "Missing '{}' in model registration entries", key);
    return false;
  }

  const std::string& path = entries.at(key);
  if (is_optional && path.empty())
  {
    return true; // Optional file path can be empty
  }

  if (path.empty() || !std::filesystem::exists(path))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid or missing file at '{}': {}", key, path);
    return false;
  }

  return true;
}

bool OdaiLlamaEngine::validate_model_files(const ModelFiles& files)
{

  if (files.m_engineType != LLAMA_BACKEND_ENGINE)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid Engine Type passed");
    return false;
  }

  if (files.m_modelType == ModelType::LLM)
  {
    if (files.m_entries.size() > 2)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid number of entries passed");
      return false;
    }

    if (!OdaiLlamaEngine::validate_model_file_entry(files.m_entries, "base_model_path", false))
    {
      return false;
    }

    if (!OdaiLlamaEngine::validate_model_file_entry(files.m_entries, "mmproj_model_path", true))
    {
      return false;
    }
  }
  else if (files.m_modelType == ModelType::EMBEDDING)
  {
    if (files.m_entries.size() != 1)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Invalid number of entries passed");
      return false;
    }

    if (!OdaiLlamaEngine::validate_model_file_entry(files.m_entries, "base_model_path", false))
    {
      return false;
    }
  }

  return true;
}

std::optional<OdaiAudioTargetSpec> OdaiLlamaEngine::get_required_audio_spec(const LLMModelConfig& config,
                                                                            const ModelFiles& model_files)
{

  if (this->m_mtmdContext == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "MtmD context not initialized");
    return std::nullopt;
  }

  if (!mtmd_support_audio(this->m_mtmdContext.get()))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Loaded model does not support audio");
    return std::nullopt;
  }

  int sample_rate = mtmd_get_audio_sample_rate(this->m_mtmdContext.get());

  if (sample_rate <= 0)
  {
    return std::nullopt;
  }

  return OdaiAudioTargetSpec{static_cast<uint32_t>(sample_rate), 1};
}

bool OdaiLlamaEngine::load_embedding_model(const ModelFiles& files, const EmbeddingModelConfig& config)
{

  if (files.m_modelType != ModelType::EMBEDDING)
  {
    return false;
  }

  std::string path = files.m_entries.at("base_model_path");

  if (this->m_embeddingModelFiles.m_entries.at("base_model_path") == path)
  {
    ODAI_LOG(ODAI_LOG_INFO, "embedding model {} is already loaded", path);
    // update config though, some other params might have changed
    this->m_embeddingModelConfig = config;
    this->m_embeddingModelFiles = files;
    return true;
  }

  llama_model_params embedding_model_params = llama_model_default_params();
  embedding_model_params.n_gpu_layers = 0; // Load entire model on CPU
  embedding_model_params.use_mlock = false;

  this->m_embeddingModel.reset(llama_model_load_from_file(path.c_str(), embedding_model_params));

  if (this->m_embeddingModel == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load embedding model");
    return false;
  }

  this->m_embeddingModelConfig = config;
  this->m_embeddingModelFiles = files;

  ODAI_LOG(ODAI_LOG_INFO, "successfully loaded embedding model {}", path);
  return true;
}

bool OdaiLlamaEngine::load_language_model(const ModelFiles& files, const LLMModelConfig& config)
{
  if (files.m_modelType != ModelType::LLM)
  {
    return false;
  }

  if ((files == this->m_llmModelFiles) && (config == this->m_llmModelConfig))
  {
    ODAI_LOG(ODAI_LOG_INFO, "Model is already loaded with same config");
    return true;
  }

  std::string base_path = files.m_entries.at("base_model_path");

  ODAI_LOG(ODAI_LOG_INFO, "Cleared all chat contexts and mtmd context as new model is being loaded");
  this->m_mtmdContext.reset();

  llama_model_params llm_model_params = llama_model_default_params();
  llm_model_params.n_gpu_layers = 0; // Load entire model on CPU
  llm_model_params.use_mlock = false;

  this->m_llmModel.reset(llama_model_load_from_file(base_path.c_str(), llm_model_params));

  if (this->m_llmModel == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load language model");
    return false;
  }

  this->m_llmVocab = llama_model_get_vocab(this->m_llmModel.get());

  if (this->m_llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load vocabulary");
    return false;
  }
  if (files.m_entries.contains("mmproj_model_path"))
  {
    const std::string& mmproj_path = files.m_entries.at("mmproj_model_path");
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = false;
    mparams.print_timings = false;

    // For vision models requiring min/max token settings
    mparams.image_min_tokens = 0; // default
    mparams.image_max_tokens = 0; // default

    mtmd_context* temp_ctx = mtmd_init_from_file(mmproj_path.c_str(), this->m_llmModel.get(), mparams);
    if (temp_ctx == nullptr)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to initialize mtmd context from: {}", mmproj_path);
      return false;
    }
    this->m_mtmdContext = std::unique_ptr<mtmd_context, MtmdContextDeleter>(temp_ctx);
    ODAI_LOG(ODAI_LOG_INFO, "Loaded multimodal projector from {}", mmproj_path);
  }

  this->m_llmModelConfig = config;
  this->m_llmModelFiles = files;

  ODAI_LOG(ODAI_LOG_INFO, "successfully loaded language model {}", base_path);
  return true;
}

std::vector<llama_token> OdaiLlamaEngine::tokenize(const std::string& input, bool is_first, ModelType model_type)
{

  const llama_vocab* vocab = nullptr;

  if (model_type == EMBEDDING)
  {
    // ToDo
  }
  else if (model_type == LLM)
  {
    vocab = this->m_llmVocab;
  }
  else
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid Tokenization purpose passed");
    return {};
  }

  if (vocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no vocab present for tokenization");
    return {};
  }

  // 1. Ask llama.cpp how much space we need
  int32_t n_tokens = -llama_tokenize(vocab, input.c_str(), input.length(), NULL, 0, is_first, true);

  if (n_tokens < 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize give input");
    return {};
  }

  std::vector<llama_token> tokens(n_tokens);

  // 2. Perform the tokenization
  llama_tokenize(vocab, input.c_str(), input.length(), tokens.data(), tokens.size(), is_first, true);

  ODAI_LOG(ODAI_LOG_DEBUG, "Input Tokenized successfully , total input tokens - {}", tokens.size());
  return tokens;
}

std::string OdaiLlamaEngine::detokenize(const std::vector<llama_token>& tokens)
{
  if (this->m_llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no LLM model loaded yet, so can't detokenize");
    return "";
  }

  std::string result;

  for (llama_token token : tokens)
  {
    char buf[128];
    int32_t n = llama_token_to_piece(this->m_llmVocab, token, buf, sizeof(buf), 0, false);

    if (n < 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to detokenize give input");
      return "";
    }

    result += std::string(buf, n);
  }

  ODAI_LOG(ODAI_LOG_DEBUG, "Input DeTokenized successfully");
  return result;
}

std::string OdaiLlamaEngine::flush_utf8_safe_output(std::vector<llama_token>& buffered_tokens,
                                                    std::string& output_buffer)
{
  output_buffer += this->detokenize(buffered_tokens);

  size_t safe_len = get_safe_utf8_length(output_buffer);
  std::string safe_output_buffer = output_buffer.substr(0, safe_len);

  // updating output buffer so that it only contains remaining unsafe part
  output_buffer = output_buffer.substr(safe_len);
  // clearing buffered tokens so that its fresh next time
  buffered_tokens.clear();

  return safe_output_buffer;
}

void OdaiLlamaEngine::add_tokens_to_batch(const std::vector<llama_token>& tokens, llama_batch& batch,
                                          uint32_t& start_pos, const llama_seq_id seq_id,
                                          const bool set_logit_request_for_last_token)
{
  for (llama_token token : tokens)
  {
    int32_t i = batch.n_tokens;
    batch.token[i] = token;
    batch.pos[i] = start_pos;
    batch.n_seq_id[i] = 1;       // single sequence
    batch.seq_id[i][0] = seq_id; // only one sequence that is sequence seq_id
    batch.logits[i] = 0;         // no logits output for input tokens
    start_pos++;
    batch.n_tokens++;
  }

  batch.logits[batch.n_tokens - 1] = set_logit_request_for_last_token ? 1 : 0; // request logits for the last token
}

bool OdaiLlamaEngine::load_tokens_into_context_impl(llama_context& model_context,
                                                    const std::vector<llama_token>& tokens, uint32_t& next_pos,
                                                    const bool request_logits_for_last_token)
{
  if (tokens.empty())
  {
    ODAI_LOG(ODAI_LOG_WARN, "empty token sequence passed");
    return true;
  }

  const uint32_t n_ctx = llama_n_ctx(&model_context);
  uint32_t n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) + 1;

  if (n_ctx_used + tokens.size() > n_ctx)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "token sequence length {} exceeds model context window (used {}/{}).", tokens.size(),
             n_ctx_used, n_ctx);
    return false;
  }

  next_pos = n_ctx_used;

  std::unique_ptr<llama_batch, LlamaBatchDeleter> batch = nullptr;

  batch.reset(new llama_batch(llama_batch_init(tokens.size(), 0, 1)));

  OdaiLlamaEngine::add_tokens_to_batch(tokens, *batch, next_pos, 0, request_logits_for_last_token);

  if (llama_decode(&model_context, *batch) != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama_decode failed");
    return false;
  }

  return true;
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const std::string& prompt, uint32_t& next_pos,
                                        const bool request_logits_for_last_token)
{
  const bool is_first = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) == -1;

  std::vector<llama_token> prompt_tokens = this->tokenize(prompt, is_first, ModelType::LLM);

  if (prompt_tokens.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize prompt");
    return false;
  }

  return OdaiLlamaEngine::load_tokens_into_context_impl(model_context, prompt_tokens, next_pos,
                                                        request_logits_for_last_token);
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const std::vector<llama_token>& tokens,
                                        uint32_t& next_pos, const bool request_logits_for_last_token)
{
  return OdaiLlamaEngine::load_tokens_into_context_impl(model_context, tokens, next_pos, request_logits_for_last_token);
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const std::string& prompt,
                                        const std::vector<mtmd::bitmap>& bitmaps, uint32_t& next_pos,
                                        bool request_logits_for_last_token)
{
  if (bitmaps.empty())
  {
    return this->load_into_context(model_context, prompt, next_pos, request_logits_for_last_token);
  }

  mtmd_input_text text;
  text.text = prompt.c_str();
  text.add_special = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) == -1;
  text.parse_special = true;

  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  std::vector<const mtmd_bitmap*> bitmaps_c_ptr(bitmaps.size());
  for (size_t i = 0; i < bitmaps.size(); ++i)
  {
    bitmaps_c_ptr[i] = bitmaps[i].ptr.get();
  }

  int32_t res =
      mtmd_tokenize(this->m_mtmdContext.get(), chunks.ptr.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
  if (res != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize prompt with mtmd, res = {}", res);
    return false;
  }

  llama_pos n_past = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) + 1;
  n_past = std::max(n_past, 0);

  uint32_t n_batch = 512;
  llama_pos new_n_past = 0;

  if (mtmd_helper_eval_chunks(this->m_mtmdContext.get(), &model_context, chunks.ptr.get(), n_past, 0, n_batch,
                              request_logits_for_last_token, &new_n_past) != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to evaluate chunks using mtmd in context");
    return false;
  }

  next_pos = static_cast<uint32_t>(new_n_past);
  return true;
}

bool OdaiLlamaEngine::generate_next_token(llama_context& model_context, llama_sampler& sampler, llama_token& out_token,
                                          const bool append_to_context)
{

  llama_token generated_token = llama_sampler_sample(&sampler, &model_context, -1);

  if (generated_token == LLAMA_TOKEN_NULL)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama_sampler_sample failed");
    return false;
  }

  llama_sampler_accept(&sampler, generated_token);

  out_token = generated_token;

  uint32_t next_pos = 0;

  if (append_to_context)
  {
    std::vector<llama_token> token_vec = {generated_token};
    if (!load_into_context(model_context, token_vec, next_pos, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to append generated token to context");
      return false;
    }
  }

  return true;
}

int32_t OdaiLlamaEngine::generate_streaming_response_impl(llama_context& model_context, llama_sampler& sampler,
                                                          const std::string& prompt,
                                                          const std::vector<mtmd::bitmap>& bitmaps,
                                                          OdaiStreamRespCallbackFn callback, void* user_data)
{
  uint32_t next_pos = 0;

  if (!this->load_into_context(model_context, prompt, bitmaps, next_pos, true))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load prompt into context");
    return -1;
  }

  llama_token generated_token = 0;
  std::vector<llama_token> buffered_tokens;
  int32_t total_tokens = 0;
  std::string output_buffer;

  while (true)
  {
    if (!OdaiLlamaEngine::generate_next_token(model_context, sampler, generated_token, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to generate next token");
      return -1;
    }

    // Check if model is done
    if (llama_vocab_is_eog(this->m_llmVocab, generated_token))
    {
      // flush left over tokens
      if (!buffered_tokens.empty())
      {
        std::string safe_output_buffer = this->flush_utf8_safe_output(buffered_tokens, output_buffer);
        if (!callback(safe_output_buffer.c_str(), user_data))
        {
          return total_tokens;
        }
      }
      break;
    }

    buffered_tokens.push_back(generated_token);
    total_tokens++;

    if ((buffered_tokens.size() % 20) == 0)
    {
      std::string safe_output_buffer = this->flush_utf8_safe_output(buffered_tokens, output_buffer);
      if (!callback(safe_output_buffer.c_str(), user_data))
      {
        return total_tokens;
      }
    }
  }

  return total_tokens;
}

int32_t OdaiLlamaEngine::generate_streaming_response(const std::vector<InputItem>& prompt,
                                                     const LLMModelConfig& llm_model_config,
                                                     const ModelFiles& model_files, const SamplerConfig& sampler_config,
                                                     OdaiStreamRespCallbackFn callback, void* user_data)
{
  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet hence can't generate response");
    return -1;
  }

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "empty callback is passed so can't stream response");
    return -1;
  }

  if (!validate_model_files(model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid model files passed");
    return -1;
  }

  if (!does_model_support_input_data(prompt, llm_model_config, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid input data");
    return -1;
  }

  if (!this->load_language_model(model_files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load given language model");
    return -1;
  }

  this->m_llmVocab = llama_model_get_vocab(this->m_llmModel.get());

  if (this->m_llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load vocabulary");
    return -1;
  }

  std::unique_ptr<llama_context, LlamaContextDeleter> llm_llama_context = this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return -1;
  }

  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> llm_llama_sampler =
      OdaiLlamaEngine::get_new_llm_llama_sampler(sampler_config);

  if (llm_llama_sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create sampler");
    return -1;
  }

  std::optional<std::pair<std::string, std::vector<mtmd::bitmap>>> process_result = this->process_input_items(prompt);

  if (!process_result.has_value())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to process input items");
    return -1;
  }

  std::string text_prompt = process_result->first;
  std::vector<mtmd::bitmap> bitmaps = std::move(process_result->second);

  // Pass bitmaps vector explicitly. Note: we need to change generate_streaming_response_impl signature.
  return generate_streaming_response_impl(*llm_llama_context, *llm_llama_sampler, text_prompt, bitmaps, callback,
                                          user_data);
}

std::optional<std::pair<std::string, std::vector<mtmd::bitmap>>>
OdaiLlamaEngine::process_input_items(const std::vector<InputItem>& items)
{
  std::string text_content;
  std::vector<mtmd::bitmap> bitmaps;

  // Cache the audio spec so we don't query it per audio item
  std::optional<OdaiAudioTargetSpec> spec_opt = get_required_audio_spec(this->m_llmModelConfig, this->m_llmModelFiles);

  for (const InputItem& item : items)
  {
    MediaType media_type = item.get_media_type();
    if (media_type == MediaType::TEXT)
    {
      std::string part_prompt = byte_vector_to_string(item.m_data);
      text_content += part_prompt;
    }
    else if (media_type == MediaType::IMAGE)
    {
      text_content += mtmd_default_marker();
      std::string file_path = byte_vector_to_string(item.m_data);

      std::unique_ptr<IOdaiImageDecoder> image_decoder = OdaiSdk::get_new_odai_image_decoder_instance();
      if (!image_decoder)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "No image decoder available to decode image input");
        return std::nullopt;
      }

      // Request original dimensions but exactly 3 channels (RGB) for mtmd
      OdaiImageTargetSpec target_spec;
      target_spec.m_maxWidth = 0;
      target_spec.m_maxHeight = 0;
      target_spec.m_channels = 3;

      OdaiDecodedImage decoded_image;
      if (!image_decoder->decode_to_spec(item, target_spec, decoded_image))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "image decoding failed for path {}", file_path);
        return std::nullopt;
      }

      mtmd_bitmap* bmp = mtmd_bitmap_init(decoded_image.m_width, decoded_image.m_height, decoded_image.m_pixels.data());
      if (bmp == nullptr)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to create mtmd_bitmap from decoded image");
        return std::nullopt;
      }
      bitmaps.emplace_back(bmp);
    }
    else if (media_type == MediaType::AUDIO)
    {
      text_content += mtmd_default_marker();
      std::unique_ptr<IOdaiAudioDecoder> audio_decoder = OdaiSdk::get_new_odai_audio_decoder_instance();
      if (!audio_decoder)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "No audio decoder available to decode audio input");
        return std::nullopt;
      }

      if (!spec_opt.has_value())
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Failed to get target audio spec");
        return std::nullopt;
      }
      OdaiDecodedAudio decoded_audio;
      if (!audio_decoder->decode_to_spec(item, spec_opt.value(), decoded_audio))
      {
        ODAI_LOG(ODAI_LOG_ERROR, "audio decoding failed");
        return std::nullopt;
      }
      mtmd_bitmap* bmp = mtmd_bitmap_init_from_audio(decoded_audio.m_samples.size(), decoded_audio.m_samples.data());
      if (bmp == nullptr)
      {
        ODAI_LOG(ODAI_LOG_ERROR, "failed to create mtmd_bitmap from audio");
        return std::nullopt;
      }
      bitmaps.emplace_back(bmp);
    }
    else
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for prompt");
      return std::nullopt;
    }
  }

  return std::make_pair(text_content, std::move(bitmaps));
}

std::string
OdaiLlamaEngine::format_chat_messages_to_prompt(const std::vector<std::pair<std::string, std::string>>& messages,
                                                bool add_generation_prompt)
{
  if (this->m_llmModel == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no model loaded yet");
    return "";
  }

  // Get the chat template from the model
  const char* tmpl = llama_model_chat_template(this->m_llmModel.get(), nullptr);

  if (tmpl == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat template from model");
    return "";
  }

  ODAI_LOG(ODAI_LOG_TRACE, "Got chat template from model: {}", tmpl);

  std::vector<llama_chat_message> llama_messages;
  llama_messages.reserve(messages.size());
  for (size_t i = 0; i < messages.size(); ++i)
  {
    llama_messages.push_back({messages[i].first.c_str(), messages[i].second.c_str()});
  }

  if (messages.size() == 1 && messages[0].first == "system")
  {
    ODAI_LOG(ODAI_LOG_WARN, "Since only system message is present, appending empty user "
                            "message to avoid chat template issues where they expect at least "
                            "one user message if system message is present");
    llama_messages.push_back({"user", ""});
  }

  std::vector<char> formatted_buffer(2048);

  // Estimate needed buffer size first
  int32_t needed_size =
      llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt,
                                formatted_buffer.data(), formatted_buffer.size());

  if (needed_size <= 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to calculate required template buffer size");
    return "";
  }

  if (needed_size > static_cast<int32_t>(formatted_buffer.size()))
  {
    // Allocate buffer and apply template
    formatted_buffer.resize(needed_size);
    int32_t actual_size =
        llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt,
                                  formatted_buffer.data(), formatted_buffer.size());
    if (actual_size <= 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to apply chat template");
      return "";
    }
  }

  ODAI_LOG(ODAI_LOG_DEBUG, "Formatted prompt: {}", formatted_buffer.data());
  return std::string(formatted_buffer.data());
}

std::unique_ptr<llama_context, LlamaContextDeleter>
OdaiLlamaEngine::load_chat_messages_into_context(const std::vector<ChatMessage>& messages)
{

  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet");
    return nullptr;
  }

  std::unique_ptr<llama_context, LlamaContextDeleter> llm_llama_context = this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return nullptr;
  }

  std::vector<mtmd::bitmap> bitmaps;
  std::vector<std::pair<std::string, std::string>> extracted_messages;

  for (const ChatMessage& msg : messages)
  {
    auto process_result = this->process_input_items(msg.m_contentItems);
    if (!process_result.has_value())
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to process chat message content items");
      return nullptr;
    }

    extracted_messages.push_back({msg.m_role, process_result->first});

    // Move the extracted bitmaps into our main bitmaps vector
    for (auto& bmp : process_result->second)
    {
      bitmaps.emplace_back(std::move(bmp));
    }
  }

  // Format the chat messages into a prompt string (without generation prompt)
  std::string formatted_prompt = this->format_chat_messages_to_prompt(extracted_messages, false);

  if (formatted_prompt.empty() && !extracted_messages.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to format chat messages into prompt");
    return nullptr;
  }

  // Load the formatted prompt into the context to build KV cache
  uint32_t next_pos = 0;

  if (!this->load_into_context(*llm_llama_context, formatted_prompt, bitmaps, next_pos, false))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load formatted prompt into context");
    return nullptr;
  }

  ODAI_LOG(ODAI_LOG_INFO, "Successfully loaded chat context");
  return llm_llama_context;
}

int32_t OdaiLlamaEngine::generate_streaming_chat_response(const std::vector<InputItem>& prompt,
                                                          const std::vector<ChatMessage>& chat_history,
                                                          const LLMModelConfig& llm_model_config,
                                                          const ModelFiles& model_files,
                                                          const SamplerConfig& sampler_config,
                                                          OdaiStreamRespCallbackFn callback, void* user_data)
{

  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet hence can't generate response");
    return -1;
  }

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "empty callback is passed so can't stream response");
    return -1;
  }

  if (!validate_model_files(model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid model files passed");
    return -1;
  }

  if (!does_model_support_input_data(prompt, llm_model_config, model_files))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Invalid input data");
    return -1;
  }

  if (!this->load_language_model(model_files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load given language model");
    return -1;
  }

  std::unique_ptr<llama_context, LlamaContextDeleter> chat_context =
      this->load_chat_messages_into_context(chat_history);

  if (!chat_context)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load chat history into cache");
    return -1;
  }

  std::unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler =
      OdaiLlamaEngine::get_new_llm_llama_sampler(sampler_config);

  if (sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create new sampler");
    return -1;
  }

  auto process_result = this->process_input_items(prompt);
  if (!process_result.has_value())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to process chat input items");
    return -1;
  }

  std::vector<std::pair<std::string, std::string>> extracted_msgs;
  extracted_msgs.push_back({"user", process_result->first});
  std::vector<mtmd::bitmap> bitmaps = std::move(process_result->second);

  std::string formatted_prompt = this->format_chat_messages_to_prompt(extracted_msgs, true);

  // Use the cached context and sampler to generate streaming response
  return this->generate_streaming_response_impl(*chat_context, *sampler, formatted_prompt, bitmaps, callback,
                                                user_data);
}

OdaiLlamaEngine::~OdaiLlamaEngine()
{
  llama_backend_free();
}