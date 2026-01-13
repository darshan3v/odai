#include "types/odai_type_conversions.h"

#include <cstring>

DBConfig toCpp(const c_DBConfig &c)
{
    return {c.dbType, string(c.dbPath)};
}

BackendEngineConfig toCpp(const c_BackendEngineConfig &c)
{
    return {c.engineType};
}

EmbeddingModelConfig toCpp(const c_EmbeddingModelConfig &c)
{
    return {string(c.modelPath)};
}

LLMModelConfig toCpp(const c_LLMModelConfig &c)
{
    return {string(c.modelPath)};
}

RagConfig toCpp(const c_RagConfig &c)
{
    return {toCpp(c.embeddingModelConfig), toCpp(c.llmModelConfig)};
}

ChatConfig toCpp(const c_ChatConfig &c)
{
    return {c.persistence, c.use_rag, string(c.system_prompt), toCpp(c.llmModelConfig)};
}

c_ChatMessage toC(const ChatMessage& cpp)
{
    c_ChatMessage result;
    
    // Copy role to fixed-size buffer (truncate if too long)
    strncpy(result.role, cpp.role.c_str(), sizeof(result.role) - 1);
    result.role[sizeof(result.role) - 1] = '\0';
    
    // Allocate and copy content
    result.content = strdup(cpp.content.c_str());
    
    // Allocate and copy message_metadata as JSON string
    string metadata_json = cpp.message_metadata.dump();
    result.message_metadata = strdup(metadata_json.c_str());
    
    result.created_at = cpp.created_at;
    
    return result;
}