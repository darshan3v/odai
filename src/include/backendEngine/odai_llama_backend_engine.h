#pragma once

#include "backendEngine/odai_backend_engine.h"

#include <llama.h>
#include <memory>
#include <vector>
#include <unordered_map>

struct llama_model_deleter
{
    void operator()(llama_model *ptr) const
    {
        if (ptr)
            llama_model_free(ptr);
    }
    llama_model_deleter() = default; // explicitly default constructor
};

struct llama_context_deleter
{
    void operator()(llama_context *ptr) const
    {
        if (ptr)
            llama_free(ptr);
    }
    llama_context_deleter() = default; // explicitly default constructor
};

struct llama_sampler_deleter
{
    void operator()(llama_sampler *ptr) const
    {
        if (ptr)
            llama_sampler_free(ptr);
    }
    llama_sampler_deleter() = default; // explicitly default constructor
};

struct llama_batch_deleter
{
    void operator()(llama_batch *ptr) const
    {
        if (ptr)
            llama_batch_free(*ptr);
    }
    llama_batch_deleter() = default; // explicitly default constructor
};

struct ChatSessionLLMContext
{
    /// Llama context with loaded KV cache for the chat session
    unique_ptr<llama_context, llama_context_deleter> context;
    /// Sampler chain used for text generation in this chat session
    unique_ptr<llama_sampler, llama_sampler_deleter> sampler;

    LLMModelConfig llm_model_config;
};

/// Llama.cpp-based implementation of the backend engine for model loading and text generation.
/// Currenntly For LLM's supports Decoder-only LLMs.
class ODAILlamaEngine : public ODAIBackendEngine
{
public:
    ODAILlamaEngine(const BackendEngineConfig &backend_engine_config);

    /// Initializes the llama backend engine and sets up logging.
    /// @return true if initialization succeeded, false otherwise
    bool initialize_engine() override;

    /// Loads an embedding model from the specified configuration.
    /// If the same model is already loaded, only updates the configuration.
    /// @param config Configuration containing model path and parameters
    /// @return true if model loaded successfully, false otherwise
    bool load_embedding_model(const EmbeddingModelConfig &config) override;

    /// Loads a language model from the specified configuration.
    /// If the same model is already loaded, only updates the configuration.
    /// @param config Configuration containing model path and parameters
    /// @return true if model loaded successfully, false otherwise
    bool load_language_model(const LLMModelConfig &config) override;

    /// Loads the provided sequence of chat messages into the model's context for the specified chat session, to do this it uses the llm_model_config to load the model.
    /// This will compute the KV cache (key-value memory for transformer inference) and keep it in memory,
    /// so future generations for the same chat can use the existing context efficiently.
    /// If the chat context is already loaded and cached for the given chat_id, this function will return immediately.
    /// @param chat_id Unique identifier for the chat session to load context for
    /// @param messages Vector of chat messages (in order) to load into the context
    /// @return true if the context was successfully loaded or already cached, false otherwise
    bool load_chat_messages_into_context(const ChatId &chat_id, const vector<ChatMessage> &messages) override;

    /// Generates a streaming response for the given prompt using the loaded language model.
    /// @param prompt The input prompt to generate a response for
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_response(const string &prompt, odai_stream_resp_callback_fn callback, void *user_data) override;

    /// Generates a streaming chat response using the cached context and sampler for the given chat session.
    /// Loads the query into the cached context and then continues generation.
    /// @param chat_id Unique identifier for the chat session whose cached context will be used
    /// @param query The input query/message to generate a response for
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_chat_response(const ChatId &chat_id, const string &prompt, odai_stream_resp_callback_fn callback, void *user_data) override;

    /// Checks if the context for a specific chat session is currently loaded in memory.
    /// @param chat_id Unique identifier for the chat session
    /// @return true if context is loaded, false otherwise
    bool is_chat_context_loaded(const ChatId &chat_id) override;

    /// Unloads the context for a specific chat session from memory, freeing resources.
    /// @param chat_id Unique identifier for the chat session
    /// @return true if unloaded successfully (or was not loaded), false on error
    bool unload_chat_context(const ChatId &chat_id) override;

    /// Destructor that frees the llama backend resources.
    ~ODAILlamaEngine() override;

private:
private:
    bool isInitialized = false;

    EmbeddingModelConfig embedding_model_config;
    LLMModelConfig llm_model_config;

    unique_ptr<llama_model, llama_model_deleter> embeddingModel = nullptr;
    unique_ptr<llama_model, llama_model_deleter> llmModel = nullptr;

    // managed automatically by llama.cpp
    // so no need of unique_ptr
    const llama_vocab *llmVocab = nullptr;

    /// Unordered map to store cached chat session data keyed by chat_id
    /// Each entry contains the context with KV cache, sampler, and metadata
    unordered_map<ChatId, ChatSessionLLMContext> chat_context;

    /// Creates a new llama context for the specified model type.
    /// @param model_type Type of model (LLM or EMBEDDING) to create context for
    /// @return Unique pointer to the context, or nullptr on error
    unique_ptr<llama_context, llama_context_deleter> get_new_llama_context(ModelType model_type);

    /// Creates a new sampler chain for language model generation.
    /// The sampler uses top-k (50), top-p (0.9), and greedy sampling strategies.
    /// @return Unique pointer to the sampler, or nullptr on error
    unique_ptr<llama_sampler, llama_sampler_deleter> get_new_llm_llama_sampler();

    /// Tokenizes an input string into llama tokens using the appropriate model.
    /// @param input The string to tokenize
    /// @param is_first Whether this is the first prompt
    /// @param model_type Type of model (LLM or EMBEDDING) to use for tokenization
    /// @return Vector of tokens, or empty vector on error
    vector<llama_token> tokenize(const string &input, bool is_first, ModelType model_type);

    /// Adds tokens to a llama batch for processing. we set the logits request for the last token.
    /// @param tokens Vector of tokens to add
    /// @param batch The batch to add tokens to (modified in place)
    /// @param start_pos Starting position for tokens (incremented by number of tokens added) (0-indexed)
    /// @param seq_id Sequence ID to assign to the tokens
    /// @param set_logit_request_for_last_token Whether to request logits for the last token added
    void add_tokens_to_batch(const vector<llama_token> &tokens, llama_batch &batch, uint32_t &start_pos, const llama_seq_id seq_id, const bool set_logit_request_for_last_token);

    /// Converts a vector of tokens back into a string.
    /// @param tokens Vector of tokens to detokenize
    /// @return Detokenized string, or empty string on error
    string detokenize(const vector<llama_token> &tokens);

    /// Loads the given prompt into the provided llama context.
    /// If the context already has some other context (Eg: some message hisotry) then we don't clear it. we just append
    /// @param model_context Language Model context to load the prompt into
    /// @param prompt The prompt string to load
    /// @param next_pos This will be updated to the indicate next position after loading the prompt
    /// @param request_logits_for_last_token Whether to request logits for the last token in the prompt, (do it if you are generating next token after this)
    /// @return true if loading succeeded, false otherwise
    bool load_into_context(llama_context &model_context, const string &prompt, uint32_t &next_pos, const bool request_logits_for_last_token);

    /// Loads the given tokens into the provided llama context.
    /// If the context already has some other context (Eg: some message hisotry) then we don't clear it. we just append
    /// @param model_context Language Model context to load the tokens into
    /// @param tokens Vector of tokens to load
    /// @param next_pos This will be updated to the indicate next position after loading the tokens
    /// @param request_logits_for_last_token Whether to request logits for the last token, (do it if you are generating next token after this)
    /// @return true if loading succeeded, false otherwise
    bool load_into_context(llama_context &model_context, const vector<llama_token> &tokens, uint32_t &next_pos, const bool request_logits_for_last_token);

    /// Helper function that performs the common logic for loading tokens into context.
    /// @param model_context Language Model context to load tokens into
    /// @param tokens Vector of tokens to load
    /// @param next_pos Starting position (updated to next position after loading)
    /// @param request_logits_for_last_token Whether to request logits for the last token
    /// @return true if loading succeeded, false otherwise
    bool load_tokens_into_context_impl(llama_context &model_context, const vector<llama_token> &tokens, uint32_t &next_pos, const bool request_logits_for_last_token);

    /// Generates the next token using the provided llama context and sampler.
    /// @param model_context Language Model context (has KV cache of old tokens and other stuff) to use for generation
    /// @param sampler sampler to use for token sampling
    /// @param out_token Output parameter to receive the generated token
    /// @param append_to_context Whether to append the generated token to the context's memory
    /// @return false if token generation failed or if appending to context is set and failed, true otherwise
    bool generate_next_token(llama_context &model_context,
                             llama_sampler &sampler,
                             llama_token &out_token,
                             const bool append_to_context);

    /// Processes buffered tokens into a UTF-8 safe string for streaming.
    /// Detokenizes tokens, appends to output buffer, then splits at a safe UTF-8 boundary.
    /// Prevents sending incomplete multi-byte characters to the client.
    /// @param buffered_tokens Tokens to process (cleared after processing)
    /// @param output_buffer Accumulated output buffer (unsafe tail remains after return)
    /// @return Safe UTF-8 string that can be sent to client
    string flush_utf8_safe_output(vector<llama_token> &buffered_tokens, string &output_buffer);

    /// Formats a vector of chat messages into a single prompt string using the model's chat template.
    /// Uses llama_chat_apply_template to apply the model's configured chat template format.
    /// @param messages Vector of chat messages to format
    /// @param add_generation_prompt Whether to add generation prompt (set to true when expecting model response)
    /// @return Formatted prompt string, or empty string on error
    string format_chat_messages_to_prompt(const vector<ChatMessage> &messages, const bool add_generation_prompt);

    /// Core implementation of streaming response generation that handles token generation and buffering.
    /// Takes an already-initialized context and sampler to perform the streaming.
    /// @param model_context Llama context with KV cache and token state
    /// @param sampler Sampler chain for token sampling
    /// @param prompt The input prompt to generate a response for
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    int32_t generate_streaming_response_impl(llama_context &model_context, llama_sampler &sampler,
                                             const string &prompt, odai_stream_resp_callback_fn callback, void *user_data);
};