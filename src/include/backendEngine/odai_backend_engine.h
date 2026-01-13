#pragma once

#include "types/odai_types.h"
#include "odai_public.h"
#include <memory>
#include <vector>

/// Abstract base class for backend engines that handle model loading and text generation.
class ODAIBackendEngine
{

public:
    /// Initializes the backend engine. Must be called before loading models.
    /// @return true if initialization succeeded, false otherwise
    virtual bool initialize_engine() = 0;

    /// Loads an embedding model from the specified configuration.
    /// If a model is already loaded, it will be freed and replaced with the new one.
    /// @param config Configuration containing model path and parameters
    /// @return true if model loaded successfully, false otherwise
    virtual bool load_embedding_model(const EmbeddingModelConfig &config) = 0;

    /// Loads a language model from the specified configuration.
    /// If a model is already loaded, it will be freed and replaced with the new one.
    /// @param config Configuration containing model path and parameters
    /// @return true if model loaded successfully, false otherwise
    virtual bool load_language_model(const LLMModelConfig &config) = 0;

    /// Loads the provided sequence of chat messages into the model's context for the specified chat session.
    /// This will compute the KV cache (key-value memory for transformer inference) and keep it in memory,
    /// so future generations for the same chat can use the existing context efficiently.
    /// If the chat context is already loaded and cached for the given chat_id, this function will return immediately.
    /// @param chat_id Unique identifier for the chat session to load context for
    /// @param messages Vector of chat messages (in order) to load into the context
    /// @return true if the context was successfully loaded or already cached, false if there was an error
    virtual bool load_chat_messages_into_context(const ChatId &chat_id, const vector<ChatMessage> &messages) = 0;

    /// Generates a streaming response for the given prompt using the loaded language model.
    /// The response is streamed incrementally via the callback function.
    /// @param prompt The input prompt to generate a response for
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    virtual int32_t generate_streaming_response(string &prompt, odai_stream_resp_callback_fn callback, void *user_data) = 0;

    /// Generates a streaming response for the given query in the given chat session.
    /// @param chat_id Unique identifier for the chat session whose cached context will be used
    /// @param query The input query/message to generate a response for
    /// @param callback Function called for each chunk of generated text
    /// @param user_data User-provided data passed to the callback
    /// @return Total number of tokens generated (excluding EOG token), or -1 on error
    virtual int32_t generate_streaming_chat_response(const ChatId &chat_id, const string &prompt, odai_stream_resp_callback_fn callback, void *user_data) = 0;

    virtual ~ODAIBackendEngine() = default;
};

// it needs to be inline not static, because if static it will simply create multiple copies wherever header is included
inline unique_ptr<ODAIBackendEngine> g_backendEngine = nullptr;
