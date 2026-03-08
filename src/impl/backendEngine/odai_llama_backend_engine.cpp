#include "llama.h"

#include "backendEngine/odai_llama_backend_engine.h"
#include "odai_sdk.h"
#include "types/odai_common_types.h"
#include "types/odai_type_conversions.h"
#include "types/odai_types.h"
#include "utils/string_utils.h"
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>
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

bool OdaiLlamaEngine::initialize_engine()
{

  llama_backend_init();

  ODAI_LOG(ODAI_LOG_INFO, "Initialized llama backend");

  llama_log_set(llama_log_redirect, nullptr);

  this->m_isInitialized = true;

  return true;
}

unique_ptr<mtmd_context, MtmdContextDeleter> OdaiLlamaEngine::load_mmproj_for_info(const string& mmproj_path,
                                                                                   llama_model* model)
{

  if (model == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "null model passed");
    return nullptr;
  }

  mtmd_context_params mparams = mtmd_context_params_default();
  mparams.use_gpu = false;
  mparams.print_timings = false;
  mparams.warmup = false;

  mtmd_context* temp_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, mparams);
  if (temp_ctx == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to initialize mtmd context from: {}", mmproj_path);
    return nullptr;
  }
  return unique_ptr<mtmd_context, MtmdContextDeleter>(temp_ctx);
}

bool OdaiLlamaEngine::does_model_support_input_data(const vector<InputItem>& items,
                                                    const LLMModelConfig& llm_model_config, const ModelFiles& files)
{
  unique_ptr<mtmd_context, MtmdContextDeleter> temp_ctx = nullptr;

  if (!this->load_language_model(files, llm_model_config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load language model");
    return false;
  }

  // Need to temporarily initialize mtmd context to read if mmproj file audio/vision support is there.
  if (files.m_entries.contains("mmproj_model_path"))
  {
    const string& mmproj_path = files.m_entries.at("mmproj_model_path");
    temp_ctx = load_mmproj_for_info(mmproj_path, this->m_llmModel.get());
    if (temp_ctx == nullptr)
    {
      return false;
    }
  }

  bool supports_image = temp_ctx != nullptr ? mtmd_support_vision(temp_ctx.get()) : false;
  bool supports_audio = temp_ctx != nullptr ? mtmd_support_audio(temp_ctx.get()) : false;

  for (const auto& item : items)
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

unique_ptr<llama_context, LlamaContextDeleter> OdaiLlamaEngine::get_new_llama_context(ModelType model_type)
{
  unique_ptr<llama_context, LlamaContextDeleter> context = nullptr;
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

unique_ptr<llama_sampler, LlamaSamplerDeleter> OdaiLlamaEngine::get_new_llm_llama_sampler(const SamplerConfig& config)
{
  unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler = nullptr;

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
  (void)config;

  if (model_files.m_engineType != LLAMA_BACKEND_ENGINE ||
      !validate_model_file_entry(model_files.m_entries, "base_model_path", false) ||
      !validate_model_file_entry(model_files.m_entries, "mmproj_model_path", false))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "invalid model files passed");
    return nullopt;
  }

  const string& mmproj_path = model_files.m_entries.at("mmproj_model_path");

  if (!load_language_model(model_files, config))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load language model");
    return std::nullopt;
  }

  // We need to temporarily initialize mtmd context to read the audio bitrate.
  auto temp_ctx = load_mmproj_for_info(mmproj_path, this->m_llmModel.get());
  if (temp_ctx == nullptr)
  {
    return std::nullopt;
  }

  if (!mtmd_support_audio(temp_ctx.get()))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "{} does not support audio", mmproj_path);
    return std::nullopt;
  }

  int bitrate = mtmd_get_audio_bitrate(temp_ctx.get());

  if (bitrate <= 0)
  {
    return std::nullopt;
  }

  return OdaiAudioTargetSpec{static_cast<uint32_t>(bitrate), 1};
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

  string base_path = files.m_entries.at("base_model_path");

  ODAI_LOG(ODAI_LOG_INFO, "Cleared all chat contexts and mtmd context as new model is being loaded");

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
  this->m_llmModelConfig = config;
  this->m_llmModelFiles = files;

  ODAI_LOG(ODAI_LOG_INFO, "successfully loaded language model {}", base_path);
  return true;
}

vector<llama_token> OdaiLlamaEngine::tokenize(const string& input, bool is_first, ModelType model_type)
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

string OdaiLlamaEngine::detokenize(const vector<llama_token>& tokens)
{
  if (this->m_llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no LLM model loaded yet, so can't detokenize");
    return "";
  }

  string result;

  for (llama_token token : tokens)
  {
    char buf[128];
    int32_t n = llama_token_to_piece(this->m_llmVocab, token, buf, sizeof(buf), 0, false);

    if (n < 0)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to detokenize give input");
      return "";
    }

    result += string(buf, n);
  }

  ODAI_LOG(ODAI_LOG_DEBUG, "Input DeTokenized successfully");
  return result;
}

string OdaiLlamaEngine::flush_utf8_safe_output(vector<llama_token>& buffered_tokens, string& output_buffer)
{
  output_buffer += this->detokenize(buffered_tokens);

  size_t safe_len = get_safe_utf8_length(output_buffer);
  string safe_output_buffer = output_buffer.substr(0, safe_len);

  // updating output buffer so that it only contains remaining unsafe part
  output_buffer = output_buffer.substr(safe_len);
  // clearing buffered tokens so that its fresh next time
  buffered_tokens.clear();

  return safe_output_buffer;
}

void OdaiLlamaEngine::add_tokens_to_batch(const vector<llama_token>& tokens, llama_batch& batch, uint32_t& start_pos,
                                          const llama_seq_id seq_id, const bool set_logit_request_for_last_token)
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

bool OdaiLlamaEngine::load_tokens_into_context_impl(llama_context& model_context, const vector<llama_token>& tokens,
                                                    uint32_t& next_pos, const bool request_logits_for_last_token)
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

  unique_ptr<llama_batch, LlamaBatchDeleter> batch = nullptr;

  batch.reset(new llama_batch(llama_batch_init(tokens.size(), 0, 1)));

  OdaiLlamaEngine::add_tokens_to_batch(tokens, *batch, next_pos, 0, request_logits_for_last_token);

  if (llama_decode(&model_context, *batch) != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama_decode failed");
    return false;
  }

  return true;
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const string& prompt, uint32_t& next_pos,
                                        const bool request_logits_for_last_token)
{
  const bool is_first = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) == -1;

  vector<llama_token> prompt_tokens = this->tokenize(prompt, is_first, ModelType::LLM);

  if (prompt_tokens.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize prompt");
    return false;
  }

  return OdaiLlamaEngine::load_tokens_into_context_impl(model_context, prompt_tokens, next_pos,
                                                        request_logits_for_last_token);
}

bool OdaiLlamaEngine::load_into_context(llama_context& model_context, const vector<llama_token>& tokens,
                                        uint32_t& next_pos, const bool request_logits_for_last_token)
{
  return OdaiLlamaEngine::load_tokens_into_context_impl(model_context, tokens, next_pos, request_logits_for_last_token);
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

  out_token = generated_token;

  uint32_t next_pos = 0;

  if (append_to_context)
  {
    vector<llama_token> token_vec = {generated_token};
    if (!load_into_context(model_context, token_vec, next_pos, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to append generated token to context");
      return false;
    }
  }

  return true;
}

int32_t OdaiLlamaEngine::generate_streaming_response_impl(llama_context& model_context, llama_sampler& sampler,
                                                          const string& prompt, OdaiStreamRespCallbackFn callback,
                                                          void* user_data)
{
  uint32_t next_pos = 0;

  if (!load_into_context(model_context, prompt, next_pos, true))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load prompt into context");
    return -1;
  }

  llama_token generated_token = 0;
  vector<llama_token> buffered_tokens;
  int32_t total_tokens = 0;
  string output_buffer;

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
        string safe_output_buffer = this->flush_utf8_safe_output(buffered_tokens, output_buffer);
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
      string safe_output_buffer = this->flush_utf8_safe_output(buffered_tokens, output_buffer);
      if (!callback(safe_output_buffer.c_str(), user_data))
      {
        return total_tokens;
      }
    }
  }

  return total_tokens;
}

int32_t OdaiLlamaEngine::generate_streaming_response(const vector<InputItem>& prompt,
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

  unique_ptr<llama_context, LlamaContextDeleter> llm_llama_context = this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return -1;
  }

  unique_ptr<llama_sampler, LlamaSamplerDeleter> llm_llama_sampler =
      OdaiLlamaEngine::get_new_llm_llama_sampler(sampler_config);

  if (llm_llama_sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create sampler");
    return -1;
  }

  // for now just text only processing
  string text_prompt;
  for (const InputItem& item : prompt)
  {
    string part_prompt = byte_vector_to_string(item.m_data);
    text_prompt += part_prompt;
  }

  // tokenize prompt
  // load prompt into context
  // and then generate

  return generate_streaming_response_impl(*llm_llama_context, *llm_llama_sampler, text_prompt, callback, user_data);
}

string OdaiLlamaEngine::format_chat_messages_to_prompt(const vector<ChatMessage>& messages,
                                                       const bool add_generation_prompt)
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

  // Convert ChatMessage vector to llama_chat_message vector
  vector<string> formatted_contents;
  for (const auto& msg : messages)
  {
    string text_content;
    for (const auto& item : msg.m_contentItems)
    {
      MediaType media_type = item.get_media_type();
      if (media_type == MediaType::TEXT)
      {
        text_content.append(item.m_data.begin(), item.m_data.end());
      }
      else
      {
        ODAI_LOG(ODAI_LOG_ERROR, "Unsupported InputItemType for prompt");
        return "";
      }
    }
    formatted_contents.push_back(text_content);
  }

  vector<llama_chat_message> llama_messages;
  llama_messages.reserve(messages.size());
  for (size_t i = 0; i < messages.size(); ++i)
  {
    llama_messages.push_back({messages[i].m_role.c_str(), formatted_contents[i].c_str()});
  }

  if (messages.size() == 1 && messages[0].m_role == "system")
  {
    ODAI_LOG(ODAI_LOG_WARN, "Since only system message is present, appending empty user "
                            "message to avoid chat template issues where they expect at least "
                            "one user message if system message is present");
    llama_messages.push_back({"user", ""});
  }

  vector<char> formatted_buffer(2048);

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
  return string(formatted_buffer.data());
}

unique_ptr<llama_context, LlamaContextDeleter>
OdaiLlamaEngine::load_chat_messages_into_context(const vector<ChatMessage>& messages)
{

  if (!this->m_isInitialized)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet");
    return nullptr;
  }

  unique_ptr<llama_context, LlamaContextDeleter> llm_llama_context = this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return nullptr;
  }

  // Format the chat messages into a prompt string (without generation prompt)
  string formatted_prompt = this->format_chat_messages_to_prompt(messages, false);

  if (formatted_prompt.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to format chat messages into prompt");
    return nullptr;
  }

  // Load the formatted prompt into the context to build KV cache
  uint32_t next_pos = 0;
  if (!this->load_into_context(*llm_llama_context, formatted_prompt, next_pos, false))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load formatted prompt into context");
    return nullptr;
  }

  ODAI_LOG(ODAI_LOG_INFO, "Successfully loaded chat context");
  return llm_llama_context;
}

int32_t OdaiLlamaEngine::generate_streaming_chat_response(const vector<InputItem>& prompt,
                                                          const vector<ChatMessage>& chat_history,
                                                          const SamplerConfig& sampler_config,
                                                          OdaiStreamRespCallbackFn callback, void* user_data)
{

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return -1;
  }

  unique_ptr<llama_context, LlamaContextDeleter> chat_context = this->load_chat_messages_into_context(chat_history);

  if (!chat_context)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to load chat history into cache");
    return -1;
  }

  unique_ptr<llama_sampler, LlamaSamplerDeleter> sampler = OdaiLlamaEngine::get_new_llm_llama_sampler(sampler_config);

  if (sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create new sampler");
    return -1;
  }

  // turn prompt into formatted prompt using chat template
  ChatMessage user_msg;
  user_msg.m_role = "user";
  user_msg.m_contentItems = prompt;
  string formatted_prompt = this->format_chat_messages_to_prompt({user_msg}, true);

  // Use the cached context and sampler to generate streaming response
  return this->generate_streaming_response_impl(*chat_context, *sampler, formatted_prompt, callback, user_data);
}

OdaiLlamaEngine::~OdaiLlamaEngine()
{
  llama_backend_free();
}