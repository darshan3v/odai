#pragma once

#include "db/odai_db.h"
#include "odai_test_helpers.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

namespace odai::test::db_contract
{
namespace fs = std::filesystem;
using odai::test::bytes_to_string;
using odai::test::expect_error;
using odai::test::string_to_bytes;

inline ModelFiles make_model_files(ModelType type, std::unordered_map<std::string, std::string> entries)
{
  ModelFiles files{};
  files.m_modelType = type;
  files.m_engineType = LLAMA_BACKEND_ENGINE;
  files.m_entries = std::move(entries);
  return files;
}

inline SemanticSpaceConfig make_semantic_space(const std::string& name)
{
  SemanticSpaceConfig config{};
  config.m_name = name;
  config.m_embeddingModelConfig = {"embedding-model"};
  FixedSizeChunkingConfig fixed_config{};
  fixed_config.m_chunkSize = 256;
  fixed_config.m_chunkOverlap = 32;
  config.m_chunkingConfig.m_config = fixed_config;
  config.m_dimensions = 384;
  return config;
}

inline ChatConfig make_chat_config()
{
  ChatConfig config{};
  config.m_persistence = true;
  config.m_systemPrompt = "You are concise.";
  config.m_llmModelConfig = {"llm-model", 1024};
  return config;
}

inline InputItem make_text_item(const std::string& text)
{
  return {InputItemType::MEMORY_BUFFER, string_to_bytes(text), "text/plain"};
}

inline ChatMessage make_chat_message(const std::string& role, const std::string& text)
{
  ChatMessage message{};
  message.m_role = role;
  message.m_contentItems.push_back(make_text_item(text));
  message.m_messageMetadata = {{"source", role}};
  return message;
}
} // namespace odai::test::db_contract
