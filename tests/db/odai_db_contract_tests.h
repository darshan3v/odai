#pragma once

#include "odai_db_test_helpers.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace odai::test::db_contract
{
template <typename Fixture>
class IOdaiDbContractTest : public Fixture
{
};

TYPED_TEST_SUITE_P(IOdaiDbContractTest);

TYPED_TEST_P(IOdaiDbContractTest, AllMethodsReturnNotInitializedBeforeInitialize)
{
  std::unique_ptr<IOdaiDb> db = this->make_uninitialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model.gguf"}});
  const std::string checksums = R"({"base_model_path":"abc"})";
  const InputItem text = make_text_item("plain text");
  const SemanticSpaceConfig space = make_semantic_space("space-a");
  const ChatConfig chat_config = make_chat_config();

  expect_error(db->begin_transaction(), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->commit_transaction(), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->rollback_transaction(), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->register_model_files("model-a", files, checksums), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->get_model_files("model-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->get_model_checksums("model-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->update_model_files("model-a", files, checksums), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->store_media_item(text), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->create_semantic_space(space), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->get_semantic_space_config("space-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->list_semantic_spaces(), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->delete_semantic_space("space-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->chat_id_exists("chat-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->create_chat("chat-a", chat_config), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->get_chat_config("chat-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->get_chat_history("chat-a"), OdaiResultEnum::NOT_INITIALIZED);
  expect_error(db->insert_chat_messages("chat-a", {make_chat_message("user", "Hello")}),
               OdaiResultEnum::NOT_INITIALIZED);
}

TYPED_TEST_P(IOdaiDbContractTest, RegisterModelFilesPersistsFilesAndChecksums)
{
  IOdaiDb& db = this->initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  const std::string checksums = R"({"base_model_path":"abc"})";

  ASSERT_TRUE(db.register_model_files("model-a", files, checksums).has_value());

  OdaiResult<ModelFiles> loaded_files = db.get_model_files("model-a");
  ASSERT_TRUE(loaded_files.has_value());
  EXPECT_EQ(loaded_files.value(), files);

  OdaiResult<std::string> loaded_checksums = db.get_model_checksums("model-a");
  ASSERT_TRUE(loaded_checksums.has_value());
  EXPECT_EQ(nlohmann::json::parse(loaded_checksums.value()), nlohmann::json::parse(checksums));
}

TYPED_TEST_P(IOdaiDbContractTest, RegisterModelFilesRejectsDuplicateModelName)
{
  IOdaiDb& db = this->initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  const std::string checksums = R"({"base_model_path":"abc"})";

  ASSERT_TRUE(db.register_model_files("model-a", files, checksums).has_value());
  expect_error(db.register_model_files("model-a", files, checksums), OdaiResultEnum::ALREADY_EXISTS);
}

TYPED_TEST_P(IOdaiDbContractTest, UpdateModelFilesReplacesStoredRecord)
{
  IOdaiDb& db = this->initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  const std::string checksums = R"({"base_model_path":"abc"})";
  ASSERT_TRUE(db.register_model_files("model-a", files, checksums).has_value());

  const ModelFiles replacement =
      make_model_files(ModelType::LLM, {{"mmproj_model_path", "/tmp/mm.gguf"}, {"tokenizer_path", "/tmp/tok.json"}});
  const std::string replacement_checksums = R"({"mmproj_model_path":"123","tokenizer_path":"tok"})";
  ASSERT_TRUE(db.update_model_files("model-a", replacement, replacement_checksums).has_value());

  OdaiResult<ModelFiles> loaded_files = db.get_model_files("model-a");
  ASSERT_TRUE(loaded_files.has_value());
  EXPECT_EQ(loaded_files.value(), replacement);
  EXPECT_FALSE(loaded_files->m_entries.contains("base_model_path"));

  OdaiResult<std::string> loaded_checksums = db.get_model_checksums("model-a");
  ASSERT_TRUE(loaded_checksums.has_value());
  EXPECT_EQ(nlohmann::json::parse(loaded_checksums.value()), nlohmann::json::parse(replacement_checksums));
}

TYPED_TEST_P(IOdaiDbContractTest, ModelFileLookupsReturnNotFoundForMissingModel)
{
  IOdaiDb& db = this->initialized_db();
  expect_error(db.get_model_files("missing-model"), OdaiResultEnum::NOT_FOUND);
  expect_error(db.get_model_checksums("missing-model"), OdaiResultEnum::NOT_FOUND);
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  expect_error(db.update_model_files("missing-model", files, R"({"base_model_path":"abc"})"),
               OdaiResultEnum::NOT_FOUND);
}

TYPED_TEST_P(IOdaiDbContractTest, SemanticSpacesCanBeCreatedListedReadAndDeleted)
{
  IOdaiDb& db = this->initialized_db();
  const SemanticSpaceConfig alpha = make_semantic_space("alpha");
  const SemanticSpaceConfig beta = make_semantic_space("beta");

  ASSERT_TRUE(db.create_semantic_space(beta).has_value());
  ASSERT_TRUE(db.create_semantic_space(alpha).has_value());

  OdaiResult<SemanticSpaceConfig> loaded = db.get_semantic_space_config("alpha");
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->m_name, alpha.m_name);
  EXPECT_EQ(loaded->m_embeddingModelConfig.m_modelName, alpha.m_embeddingModelConfig.m_modelName);
  EXPECT_EQ(loaded->m_dimensions, alpha.m_dimensions);

  OdaiResult<std::vector<SemanticSpaceConfig>> spaces = db.list_semantic_spaces();
  ASSERT_TRUE(spaces.has_value());
  ASSERT_EQ(spaces->size(), 2U);
  EXPECT_EQ((*spaces)[0].m_name, "alpha");
  EXPECT_EQ((*spaces)[1].m_name, "beta");

  ASSERT_TRUE(db.delete_semantic_space("alpha").has_value());
  spaces = db.list_semantic_spaces();
  ASSERT_TRUE(spaces.has_value());
  ASSERT_EQ(spaces->size(), 1U);
  EXPECT_EQ((*spaces)[0].m_name, "beta");
  expect_error(db.get_semantic_space_config("alpha"), OdaiResultEnum::NOT_FOUND);
}

TYPED_TEST_P(IOdaiDbContractTest, SemanticSpacesReportDuplicateAndMissingErrors)
{
  IOdaiDb& db = this->initialized_db();
  const SemanticSpaceConfig alpha = make_semantic_space("alpha");

  ASSERT_TRUE(db.create_semantic_space(alpha).has_value());
  expect_error(db.create_semantic_space(alpha), OdaiResultEnum::ALREADY_EXISTS);
  expect_error(db.get_semantic_space_config("missing-space"), OdaiResultEnum::NOT_FOUND);
  expect_error(db.delete_semantic_space("missing-space"), OdaiResultEnum::NOT_FOUND);
}

TYPED_TEST_P(IOdaiDbContractTest, ListSemanticSpacesReturnsEmptyWhenNoneExist)
{
  IOdaiDb& db = this->initialized_db();

  OdaiResult<std::vector<SemanticSpaceConfig>> spaces = db.list_semantic_spaces();
  ASSERT_TRUE(spaces.has_value());
  EXPECT_TRUE(spaces->empty());
}

TYPED_TEST_P(IOdaiDbContractTest, StoreMediaItemLeavesTextMemoryBufferUnchangedAndPreservesMimeCase)
{
  IOdaiDb& db = this->initialized_db();

  const InputItem text{InputItemType::MEMORY_BUFFER, string_to_bytes("plain text"), "Text/Plain"};
  OdaiResult<InputItem> stored_text = db.store_media_item(text);
  ASSERT_TRUE(stored_text.has_value());
  EXPECT_EQ(stored_text->m_type, InputItemType::MEMORY_BUFFER);
  EXPECT_EQ(stored_text->m_mimeType, "Text/Plain");
  EXPECT_EQ(bytes_to_string(stored_text->m_data), "plain text");
}

TYPED_TEST_P(IOdaiDbContractTest, StoreMediaItemCachesBinaryMemoryBufferAndDeduplicates)
{
  IOdaiDb& db = this->initialized_db();

  InputItem memory_image{InputItemType::MEMORY_BUFFER, {0x89, 'P', 'N', 'G', 0x0D, 0x0A}, "Image/PNG"};
  OdaiResult<InputItem> stored_memory = db.store_media_item(memory_image);
  ASSERT_TRUE(stored_memory.has_value());
  EXPECT_EQ(stored_memory->m_type, InputItemType::FILE_PATH);
  EXPECT_EQ(stored_memory->m_mimeType, "Image/PNG");
  const fs::path memory_cache_path = bytes_to_string(stored_memory->m_data);
  EXPECT_TRUE(fs::exists(memory_cache_path));

  OdaiResult<InputItem> stored_memory_again = db.store_media_item(memory_image);
  ASSERT_TRUE(stored_memory_again.has_value());
  EXPECT_EQ(bytes_to_string(stored_memory_again->m_data), memory_cache_path.string());
  EXPECT_EQ(stored_memory_again->m_mimeType, "Image/PNG");
}

TYPED_TEST_P(IOdaiDbContractTest, StoreMediaItemCachesSourceFileAndDeduplicates)
{
  IOdaiDb& db = this->initialized_db();
  const InputItem file_audio = this->make_source_file_item("source-audio.bin", "audio-bytes", "Audio/Wav");

  OdaiResult<InputItem> stored_file = db.store_media_item(file_audio);
  ASSERT_TRUE(stored_file.has_value());
  EXPECT_EQ(stored_file->m_type, InputItemType::FILE_PATH);
  EXPECT_EQ(stored_file->m_mimeType, "Audio/Wav");
  const fs::path file_cache_path = bytes_to_string(stored_file->m_data);
  EXPECT_TRUE(fs::exists(file_cache_path));

  OdaiResult<InputItem> stored_file_again = db.store_media_item(file_audio);
  ASSERT_TRUE(stored_file_again.has_value());
  EXPECT_EQ(bytes_to_string(stored_file_again->m_data), file_cache_path.string());
  EXPECT_EQ(stored_file_again->m_mimeType, "Audio/Wav");
}

TYPED_TEST_P(IOdaiDbContractTest, ChatCanBeCreatedReadAndExtendedWithChronologicalHistory)
{
  IOdaiDb& db = this->initialized_db();
  const ChatConfig config = make_chat_config();

  OdaiResult<bool> chat_exists = db.chat_id_exists("chat-a");
  ASSERT_TRUE(chat_exists.has_value());
  EXPECT_FALSE(chat_exists.value());

  ASSERT_TRUE(db.create_chat("chat-a", config).has_value());
  chat_exists = db.chat_id_exists("chat-a");
  ASSERT_TRUE(chat_exists.has_value());
  EXPECT_TRUE(chat_exists.value());

  OdaiResult<ChatConfig> loaded_config = db.get_chat_config("chat-a");
  ASSERT_TRUE(loaded_config.has_value());
  EXPECT_EQ(loaded_config->m_systemPrompt, config.m_systemPrompt);
  EXPECT_EQ(loaded_config->m_llmModelConfig, config.m_llmModelConfig);

  ASSERT_TRUE(
      db.insert_chat_messages("chat-a", {make_chat_message("user", "Hello"), make_chat_message("assistant", "Hi.")})
          .has_value());

  OdaiResult<std::vector<ChatMessage>> history = db.get_chat_history("chat-a");
  ASSERT_TRUE(history.has_value());
  ASSERT_EQ(history->size(), 3U);
  EXPECT_EQ((*history)[0].m_role, "system");
  EXPECT_EQ(bytes_to_string((*history)[0].m_contentItems[0].m_data), config.m_systemPrompt);
  EXPECT_EQ((*history)[1].m_role, "user");
  EXPECT_EQ(bytes_to_string((*history)[1].m_contentItems[0].m_data), "Hello");
  EXPECT_EQ((*history)[2].m_role, "assistant");
  EXPECT_EQ(bytes_to_string((*history)[2].m_contentItems[0].m_data), "Hi.");
  EXPECT_EQ((*history)[1].m_messageMetadata["source"], "user");
}

TYPED_TEST_P(IOdaiDbContractTest, ChatMethodsReportDuplicateMissingAndValidationErrors)
{
  IOdaiDb& db = this->initialized_db();
  const ChatConfig config = make_chat_config();

  ASSERT_TRUE(db.create_chat("chat-a", config).has_value());
  expect_error(db.create_chat("chat-a", config), OdaiResultEnum::ALREADY_EXISTS);
  expect_error(db.get_chat_config("missing-chat"), OdaiResultEnum::NOT_FOUND);
  expect_error(db.get_chat_history("missing-chat"), OdaiResultEnum::NOT_FOUND);
  expect_error(db.insert_chat_messages("missing-chat", {make_chat_message("user", "Hello")}),
               OdaiResultEnum::NOT_FOUND);
  expect_error(db.insert_chat_messages("chat-a", {}), OdaiResultEnum::VALIDATION_FAILED);
}

TYPED_TEST_P(IOdaiDbContractTest, InsertChatMessagesRollsBackWholeBatchWhenOneMessageIsInvalid)
{
  IOdaiDb& db = this->initialized_db();
  const ChatConfig config = make_chat_config();
  ASSERT_TRUE(db.create_chat("chat-rollback", config).has_value());

  ChatMessage valid_message = make_chat_message("user", "committed only if whole batch is valid");
  ChatMessage invalid_message = make_chat_message("assistant", "invalid text file path");
  invalid_message.m_contentItems[0].m_type = InputItemType::FILE_PATH;
  invalid_message.m_contentItems[0].m_data = string_to_bytes("not-a-supported-text-path.txt");

  expect_error(db.insert_chat_messages("chat-rollback", {valid_message, invalid_message}),
               OdaiResultEnum::VALIDATION_FAILED);

  OdaiResult<std::vector<ChatMessage>> history = db.get_chat_history("chat-rollback");
  ASSERT_TRUE(history.has_value());
  ASSERT_EQ(history->size(), 1U);
  EXPECT_EQ((*history)[0].m_role, "system");
}

TYPED_TEST_P(IOdaiDbContractTest, TransactionsCommitRollbackAndFlattenNestedCalls)
{
  IOdaiDb& db = this->initialized_db();
  const ModelFiles files = make_model_files(ModelType::EMBEDDING, {{"base_model_path", "/tmp/embed.gguf"}});
  const std::string checksums = R"({"base_model_path":"embed"})";

  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.register_model_files("committed-model", files, checksums).has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());
  EXPECT_TRUE(db.get_model_files("committed-model").has_value());

  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.register_model_files("rolled-back-model", files, checksums).has_value());
  ASSERT_TRUE(db.rollback_transaction().has_value());
  expect_error(db.get_model_files("rolled-back-model"), OdaiResultEnum::NOT_FOUND);

  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.register_model_files("nested-model", files, checksums).has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());
  EXPECT_TRUE(db.get_model_files("nested-model").has_value());
  ASSERT_TRUE(db.rollback_transaction().has_value());
  expect_error(db.get_model_files("nested-model"), OdaiResultEnum::NOT_FOUND);

  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.register_model_files("nested-committed-model", files, checksums).has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());
  EXPECT_TRUE(db.get_model_files("nested-committed-model").has_value());
}

TYPED_TEST_P(IOdaiDbContractTest, PersistenceSurvivesCloseAndReopen)
{
  IOdaiDb& db = this->initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/persist.gguf"}});
  const std::string checksums = R"({"base_model_path":"persist"})";
  ASSERT_TRUE(db.register_model_files("persist-model", files, checksums).has_value());

  this->reopen_db();

  OdaiResult<ModelFiles> loaded = this->initialized_db().get_model_files("persist-model");
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded.value(), files);
}

REGISTER_TYPED_TEST_SUITE_P(IOdaiDbContractTest, AllMethodsReturnNotInitializedBeforeInitialize,
                            RegisterModelFilesPersistsFilesAndChecksums, RegisterModelFilesRejectsDuplicateModelName,
                            UpdateModelFilesReplacesStoredRecord, ModelFileLookupsReturnNotFoundForMissingModel,
                            SemanticSpacesCanBeCreatedListedReadAndDeleted,
                            SemanticSpacesReportDuplicateAndMissingErrors, ListSemanticSpacesReturnsEmptyWhenNoneExist,
                            StoreMediaItemLeavesTextMemoryBufferUnchangedAndPreservesMimeCase,
                            StoreMediaItemCachesBinaryMemoryBufferAndDeduplicates,
                            StoreMediaItemCachesSourceFileAndDeduplicates,
                            ChatCanBeCreatedReadAndExtendedWithChronologicalHistory,
                            ChatMethodsReportDuplicateMissingAndValidationErrors,
                            InsertChatMessagesRollsBackWholeBatchWhenOneMessageIsInvalid,
                            TransactionsCommitRollbackAndFlattenNestedCalls, PersistenceSurvivesCloseAndReopen);

} // namespace odai::test::db_contract
