#include "llama.h"

#include "backendEngine/odai_llama_backend_engine.h"
#include "odai_sdk.h"
#include "odai_sdk.h"
#include "utils/string_utils.h"

/// Redirects llama.cpp log messages to the Odai logging system.
/// Maps GGML log levels to Odai log levels and filters out debug messages.
/// @param level The GGML log level
/// @param text The log message text
/// @param user_data User data (unused)
static void llama_log_redirect(ggml_log_level level, const char *text,
                               void *user_data)
{
  // Map llama levels to Odai levels
  OdaiLogLevel our_level = ODAI_LOG_INFO;

  if (level == GGML_LOG_LEVEL_ERROR)
    our_level = ODAI_LOG_ERROR;
  else if (level == GGML_LOG_LEVEL_WARN)
    our_level = ODAI_LOG_WARN;
  else if (level == GGML_LOG_LEVEL_INFO)
    our_level = ODAI_LOG_INFO;
  else
    return; // Ignore debug/spam from llama if you want

  ODAI_LOG(our_level, "[llama.cpp] {}", text);
}

ODAILlamaEngine::ODAILlamaEngine(const BackendEngineConfig &backend_engine_config)
{
}

bool ODAILlamaEngine::initialize_engine()
{

  llama_backend_init();

  ODAI_LOG(ODAI_LOG_INFO, "Initialized llama backend");

  llama_log_set(llama_log_redirect, nullptr);

  this->isInitialized = true;

  return true;
}

unique_ptr<llama_context, llama_context_deleter>
ODAILlamaEngine::get_new_llama_context(ModelType model_type)
{
  unique_ptr<llama_context, llama_context_deleter> context = nullptr;
  llama_context_params context_params = llama_context_default_params();
  llama_model *model = nullptr;

  context_params.n_threads = 4;

  if (model_type == ModelType::LLM)
  {
    model = this->llmModel.get();
    context_params.n_ctx = 2048;
    context_params.embeddings = false;
  }
  else if (model_type == ModelType::EMBEDDING)
  {
    model = this->embeddingModel.get();
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

unique_ptr<llama_sampler, llama_sampler_deleter>
ODAILlamaEngine::get_new_llm_llama_sampler()
{
  unique_ptr<llama_sampler, llama_sampler_deleter> sampler = nullptr;

  llama_sampler_chain_params sampler_params =
      llama_sampler_chain_default_params();

  sampler.reset(llama_sampler_chain_init(sampler_params));

  if (sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to create llama sampler");
    return nullptr;
  }

  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(50));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(0.9f, 1));
  llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());

  return sampler;
}

bool ODAILlamaEngine::load_embedding_model(const EmbeddingModelConfig &config)
{
  if (this->isInitialized == false)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet");
    return false;
  }

  if (this->embedding_model_config.modelPath == config.modelPath)
  {
    ODAI_LOG(ODAI_LOG_INFO, "embedding model {} is already loaded",
             config.modelPath);
    // update config though, some other params might have changed
    this->embedding_model_config = config;
    return true;
  }

  llama_model_params embedding_model_params = llama_model_default_params();
  embedding_model_params.n_gpu_layers = 0; // Load entire model on CPU
  embedding_model_params.use_mlock = false;

  this->embeddingModel.reset(llama_model_load_from_file(
      config.modelPath.c_str(), embedding_model_params));

  if (this->embeddingModel == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load embedding model");
    return false;
  }

  this->embedding_model_config = config;

  ODAI_LOG(ODAI_LOG_INFO, "successfully loaded embedding model {}",
           config.modelPath);
  return true;
}

bool ODAILlamaEngine::load_language_model(const LLMModelConfig &config)
{
  if (this->isInitialized == false)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized");
    return false;
  }

  if (this->llm_model_config.modelPath == config.modelPath)
  {
    ODAI_LOG(ODAI_LOG_INFO, "language model {} is already loaded",
             config.modelPath);
    // update config though, some other params might have changed
    this->llm_model_config = config;
    return true;
  }

  // Clear all chat contexts since they depend on the old model
  this->chat_context.clear();
  ODAI_LOG(ODAI_LOG_INFO, "Cleared all chat contexts as new model is being loaded");

  llama_model_params llm_model_params = llama_model_default_params();
  llm_model_params.n_gpu_layers = 0; // Load entire model on CPU
  llm_model_params.use_mlock = false;

  this->llmModel.reset(
      llama_model_load_from_file(config.modelPath.c_str(), llm_model_params));

  if (this->llmModel == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load language model");
    return false;
  }

  this->llmVocab = llama_model_get_vocab(this->llmModel.get());

  if (this->llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load vocabulary");
    return false;
  }

  this->llm_model_config = config;

  ODAI_LOG(ODAI_LOG_INFO, "successfully loaded language model {}",
           config.modelPath);
  return true;
}

vector<llama_token> ODAILlamaEngine::tokenize(const string &input, bool is_first,
                                              ModelType model_type)
{

  const llama_vocab *vocab = nullptr;

  if (model_type == EMBEDDING)
  {
    // ToDo
  }
  else if (model_type == LLM)
  {
    vocab = this->llmVocab;
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
  int32_t n_tokens = -llama_tokenize(vocab, input.c_str(), input.length(), NULL,
                                     0, is_first, true);

  if (n_tokens < 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize give input");
    return {};
  }

  std::vector<llama_token> tokens(n_tokens);

  // 2. Perform the tokenization
  llama_tokenize(vocab, input.c_str(), input.length(), tokens.data(),
                 tokens.size(), is_first, true);

  ODAI_LOG(ODAI_LOG_DEBUG,
           "Input Tokenized successfully , total input tokens - {}",
           tokens.size());
  return tokens;
}

string ODAILlamaEngine::detokenize(const vector<llama_token> &tokens)
{
  if (this->llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no LLM model loaded yet, so can't detokenize");
    return "";
  }

  string result = "";

  for (llama_token token : tokens)
  {
    char buf[128];
    int32_t n =
        llama_token_to_piece(this->llmVocab, token, buf, sizeof(buf), 0, false);

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

string
ODAILlamaEngine::flush_utf8_safe_output(vector<llama_token> &buffered_tokens,
                                        string &output_buffer)
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

void ODAILlamaEngine::add_tokens_to_batch(const vector<llama_token> &tokens,
                                          llama_batch &batch,
                                          uint32_t &start_pos,
                                          const llama_seq_id seq_id,
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

bool ODAILlamaEngine::load_tokens_into_context_impl(llama_context &model_context, const vector<llama_token> &tokens, uint32_t &next_pos, const bool request_logits_for_last_token)
{
  if (tokens.size() == 0)
  {
    ODAI_LOG(ODAI_LOG_WARN, "empty token sequence passed");
    return true;
  }

  const uint32_t n_ctx = llama_n_ctx(&model_context);
  uint32_t n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) + 1;

  if (n_ctx_used + tokens.size() > n_ctx)
  {
    ODAI_LOG(ODAI_LOG_ERROR,
             "token sequence length {} exceeds model context window (used {}/{}).",
             tokens.size(), n_ctx_used, n_ctx);
    return false;
  }

  next_pos = n_ctx_used;

  unique_ptr<llama_batch, llama_batch_deleter> batch = nullptr;

  batch.reset(new llama_batch(llama_batch_init(tokens.size(), 0, 1)));

  this->add_tokens_to_batch(tokens, *batch, next_pos, 0, request_logits_for_last_token);

  if (llama_decode(&model_context, *batch) != 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama_decode failed");
    return false;
  }

  return true;
}

bool ODAILlamaEngine::load_into_context(llama_context &model_context, const string &prompt, uint32_t &next_pos, const bool request_logits_for_last_token)
{
  const bool is_first = llama_memory_seq_pos_max(llama_get_memory(&model_context), 0) == -1;

  vector<llama_token> prompt_tokens =
      this->tokenize(prompt, is_first, ModelType::LLM);

  if (prompt_tokens.size() == 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to tokenize prompt");
    return false;
  }

  return this->load_tokens_into_context_impl(model_context, prompt_tokens, next_pos, request_logits_for_last_token);
}

bool ODAILlamaEngine::load_into_context(llama_context &model_context, const vector<llama_token> &tokens, uint32_t &next_pos, const bool request_logits_for_last_token)
{
  return this->load_tokens_into_context_impl(model_context, tokens, next_pos, request_logits_for_last_token);
}

bool ODAILlamaEngine::generate_next_token(llama_context &model_context,
                                          llama_sampler &sampler,
                                          llama_token &out_token,
                                          const bool append_to_context)
{

  llama_token generated_token = llama_sampler_sample(
      &sampler, &model_context, -1);

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
    if (!load_into_context(model_context,
                           token_vec,
                           next_pos, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to append generated token to context");
      return false;
    }
  }

  return true;
}

int32_t ODAILlamaEngine::generate_streaming_response_impl(
    llama_context &model_context, llama_sampler &sampler,
    const string &prompt, odai_stream_resp_callback_fn callback,
    void *user_data)
{
  uint32_t next_pos = 0;

  if (!load_into_context(model_context, prompt, next_pos, true))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load prompt into context");
    return -1;
  }

  llama_token generated_token;
  vector<llama_token> buffered_tokens;
  int32_t total_tokens = 0;
  string output_buffer;

  while (true)
  {
    if (!generate_next_token(model_context, sampler, generated_token, true))
    {
      ODAI_LOG(ODAI_LOG_ERROR, "failed to generate next token");
      return -1;
    }

    // Check if model is done
    if (llama_vocab_is_eog(this->llmVocab, generated_token))
    {
      // flush left over tokens
      if (buffered_tokens.size() > 0)
      {
        string safe_output_buffer =
            this->flush_utf8_safe_output(buffered_tokens, output_buffer);
        if (!callback(safe_output_buffer.c_str(), user_data))
          return total_tokens;
      }
      break;
    }

    buffered_tokens.push_back(generated_token);
    total_tokens++;

    if ((buffered_tokens.size() % 20) == 0)
    {
      string safe_output_buffer =
          this->flush_utf8_safe_output(buffered_tokens, output_buffer);
      if (!callback(safe_output_buffer.c_str(), user_data))
        return total_tokens;
    }
  }

  return total_tokens;
}

int32_t ODAILlamaEngine::generate_streaming_response(
    const string &prompt, odai_stream_resp_callback_fn callback,
    void *user_data)
{
  if (this->isInitialized == false)
  {
    ODAI_LOG(
        ODAI_LOG_ERROR,
        "llama backend is not Initialized yet hence can't generate response");
    return -1;
  }

  if (this->llmModel == nullptr || this->llmVocab == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no model loaded yet, so can't generate response");
    return -1;
  }

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR,
             "empty callback is passed so can't stream response");
    return -1;
  }

  unique_ptr<llama_context, llama_context_deleter> llm_llama_context =
      this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return -1;
  }

  unique_ptr<llama_sampler, llama_sampler_deleter> llm_llama_sampler =
      this->get_new_llm_llama_sampler();

  if (llm_llama_sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create sampler");
    return -1;
  }

  return generate_streaming_response_impl(*llm_llama_context, *llm_llama_sampler,
                                          prompt, callback, user_data);
}

string
ODAILlamaEngine::format_chat_messages_to_prompt(const vector<ChatMessage> &messages, const bool add_generation_prompt)
{
  if (this->llmModel == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "no model loaded yet");
    return "";
  }

  // Get the chat template from the model
  const char *tmpl = llama_model_chat_template(this->llmModel.get(), nullptr);

  if (tmpl == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to get chat template from model");
    return "";
  }

  ODAI_LOG(ODAI_LOG_TRACE, "Got chat template from model: {}", tmpl);

  // Convert ChatMessage vector to llama_chat_message vector
  vector<llama_chat_message> llama_messages;
  for (const auto &msg : messages)
  {
    llama_messages.push_back({msg.role.c_str(), msg.content.c_str()});
  }

  if (messages.size() == 1 && messages[0].role == "system")
  {
    ODAI_LOG(ODAI_LOG_WARN, "Since only system message is present, appending empty user message to avoid chat template issues where they expect at least one user message if system message is present");
    llama_messages.push_back({"user", ""});
  }

  vector<char> formatted_buffer(2048);

  // Estimate needed buffer size first
  int32_t needed_size = llama_chat_apply_template(
      tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt, formatted_buffer.data(), formatted_buffer.size());

  if (needed_size <= 0)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to calculate required template buffer size");
    return "";
  }

  if (needed_size > static_cast<int32_t>(formatted_buffer.size()))
  {
    // Allocate buffer and apply template
    formatted_buffer.resize(needed_size);
    int32_t actual_size = llama_chat_apply_template(
        tmpl, llama_messages.data(), llama_messages.size(), add_generation_prompt,
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

bool ODAILlamaEngine::load_chat_messages_into_context(
    const ChatId &chat_id, const vector<ChatMessage> &messages)
{

  if (this->isInitialized == false)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "llama backend is not Initialized yet");
    return false;
  }

  if (chat_id.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "empty chat_id is passed");
    return false;
  }

  if (this->chat_context.find(chat_id) != this->chat_context.end())
  {
    ODAI_LOG(ODAI_LOG_INFO, "chat context for chat_id {} is already loaded", chat_id);
    return true;
  }

  unique_ptr<llama_context, llama_context_deleter> llm_llama_context =
      this->get_new_llama_context(ModelType::LLM);

  if (llm_llama_context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create context");
    return false;
  }

  unique_ptr<llama_sampler, llama_sampler_deleter> llm_llama_sampler =
      this->get_new_llm_llama_sampler();

  if (llm_llama_sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "something went wrong, couldn't create sampler");
    return false;
  }

  // Format the chat messages into a prompt string (without generation prompt)
  string formatted_prompt = this->format_chat_messages_to_prompt(messages, false);

  if (formatted_prompt.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to format chat messages into prompt");
    return false;
  }

  // Load the formatted prompt into the context to build KV cache
  uint32_t next_pos = 0;
  if (!this->load_into_context(*llm_llama_context, formatted_prompt, next_pos, false))
  {
    ODAI_LOG(ODAI_LOG_ERROR, "failed to load formatted prompt into context");
    return false;
  }

  // Store the context and sampler for this chat session
  this->chat_context[chat_id] = {move(llm_llama_context), move(llm_llama_sampler)};

  ODAI_LOG(ODAI_LOG_INFO, "Successfully loaded chat context for chat_id {}", chat_id);
  return true;
}

int32_t ODAILlamaEngine::generate_streaming_chat_response(const ChatId &chat_id, const string &prompt, odai_stream_resp_callback_fn callback, void *user_data)
{
  // Find the cached context and sampler for this chat session
  auto chat_ctx_iter = this->chat_context.find(chat_id);
  if (chat_ctx_iter == this->chat_context.end())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Chat context not found for chat_id: {}", chat_id);
    return -1;
  }

  ChatSessionLLMContext &chat_session = chat_ctx_iter->second;

  if (chat_session.context == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Chat context is null for chat_id: {}", chat_id);
    return -1;
  }

  if (chat_session.sampler == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Chat sampler is null for chat_id: {}", chat_id);
    return -1;
  }

  if (callback == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Callback is null");
    return -1;
  }

  // turn prompt into formatted prompt using chat template
  string formatted_prompt = this->format_chat_messages_to_prompt({{"user", prompt}}, true);

  // Use the cached context and sampler to generate streaming response
  return this->generate_streaming_response_impl(*chat_session.context, *chat_session.sampler, formatted_prompt, callback, user_data);
}

bool ODAILlamaEngine::is_chat_context_loaded(const ChatId &chat_id)
{
  return this->chat_context.find(chat_id) != this->chat_context.end();
}

bool ODAILlamaEngine::unload_chat_context(const ChatId &chat_id)
{
  auto it = this->chat_context.find(chat_id);
  if (it != this->chat_context.end())
  {
    this->chat_context.erase(it);
    ODAI_LOG(ODAI_LOG_INFO, "Unloaded chat context for chat_id: {}", chat_id);
    return true;
  }
  ODAI_LOG(ODAI_LOG_WARN, "Chat context not found for chat_id: {}, so nothing to unload", chat_id);
  return false;
}

ODAILlamaEngine::~ODAILlamaEngine() { llama_backend_free(); }