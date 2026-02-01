#include "types/odai_type_conversions.h"

#include <cstring>

ModelType toCpp(c_ModelType c)
{
    if (c == ODAI_MODEL_TYPE_LLM)
        return ModelType::LLM;
    else if (c == ODAI_MODEL_TYPE_EMBEDDING)
        return ModelType::EMBEDDING;
    
    // Default to LLM if unknown, or handle error appropriately. 
    // Since we can't easily return error here, we assume valid input or handle at caller.
    return ModelType::LLM; 
}

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
    return {string(c.modelName)};
}

LLMModelConfig toCpp(const c_LLMModelConfig &c)
{
    return {string(c.modelName)};
}

ChunkingConfig toCpp(const c_ChunkingConfig &c)
{
    ChunkingConfig config;

    if (c.strategy == FIXED_SIZE_CHUNKING)
    {
        FixedSizeChunkingConfig fcc;
        fcc.chunkSize = c.config.fixedSizeConfig.chunkSize;
        fcc.chunkOverlap = c.config.fixedSizeConfig.chunkOverlap;
        config.config = fcc;
    }
    return config;
}

SemanticSpaceConfig toCpp(const c_SemanticSpaceConfig &c)
{
    SemanticSpaceConfig config;
    config.name = string(c.name);
    config.embeddingModelConfig = toCpp(c.embeddingModelConfig);
    config.chunkingConfig = toCpp(c.chunkingConfig);
    config.dimensions = c.dimensions;
    return config;
}

RagConfig toCpp(const c_RagConfig &c)
{
    RagConfig config;
    config.embeddingModelConfig = toCpp(c.embeddingModelConfig);
    config.llmModelConfig = toCpp(c.llmModelConfig);
    config.profile = c.profile;
    return config;
}

ChatConfig toCpp(const c_ChatConfig &c)
{
    return {c.persistence, c.use_rag, string(c.system_prompt), toCpp(c.llmModelConfig)};
}

c_EmbeddingModelConfig toC(const EmbeddingModelConfig& cpp)
{
    c_EmbeddingModelConfig c;
    c.modelName = strdup(cpp.modelName.c_str());
    return c;
}

c_ChunkingConfig toC(const ChunkingConfig& cpp)
{
    c_ChunkingConfig c;
    if (std::holds_alternative<FixedSizeChunkingConfig>(cpp.config))
    {
        c.strategy = FIXED_SIZE_CHUNKING;
        auto& conf = std::get<FixedSizeChunkingConfig>(cpp.config);
        c.config.fixedSizeConfig.chunkSize = conf.chunkSize;
        c.config.fixedSizeConfig.chunkOverlap = conf.chunkOverlap;
    }
    return c;
}

c_SemanticSpaceConfig toC(const SemanticSpaceConfig& cpp)
{
    c_SemanticSpaceConfig c;
    c.name = strdup(cpp.name.c_str());
    c.embeddingModelConfig = toC(cpp.embeddingModelConfig);
    c.chunkingConfig = toC(cpp.chunkingConfig);
    c.dimensions = cpp.dimensions;
    return c;
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

void to_json(json& j, const ChunkingConfig& p)
{
    if (std::holds_alternative<FixedSizeChunkingConfig>(p.config))
    {
        j = json{{"strategy", FIXED_SIZE_CHUNKING}};
        j["config"] = std::get<FixedSizeChunkingConfig>(p.config);
    }
}

void from_json(const json& j, ChunkingConfig& p)
{
    ChunkingStrategy strategy;
    j.at("strategy").get_to(strategy);
    if (strategy == FIXED_SIZE_CHUNKING)
    {
        if (j.contains("config"))
        {
            FixedSizeChunkingConfig conf;
            j.at("config").get_to(conf);
            p.config = conf;
        }
    }
}