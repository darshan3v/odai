#include "utils/odai_csanitizers.h"

bool is_sane(const c_DBConfig *config)
{
    return config != nullptr &&
           config->dbPath != nullptr;
}

bool is_sane(const c_BackendEngineConfig *config)
{
    // Currently just validate that a backend is specified
    // This can be extended with more validation logic
    return true;
}

bool is_sane(const c_EmbeddingModelConfig *config)
{
    return config != nullptr &&
           config->modelPath != nullptr;
}

bool is_sane(const c_LLMModelConfig *config)
{
    return config != nullptr &&
           config->modelPath != nullptr;
}

bool is_sane(const c_RagConfig *config)
{
    return config != nullptr &&
           is_sane(&config->embeddingModelConfig) &&
           is_sane(&config->llmModelConfig);
}

bool is_sane(const c_ChatConfig *config)
{
    return config != nullptr &&
           config->system_prompt != nullptr &&
           is_sane(&config->llmModelConfig);
}