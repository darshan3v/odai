#include "db/odai_sqlite/odai_sqlite_db.h"

#include "odai_db_test_helpers.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <SQLiteCpp/SQLiteCpp.h>
#include <gtest/gtest.h>

namespace fs = std::filesystem;
using odai::test::bytes_to_string;
using odai::test::string_to_bytes;
using odai::test::db_contract::expect_error;
using odai::test::db_contract::make_chat_config;
using odai::test::db_contract::make_chat_message;
using odai::test::db_contract::make_model_files;
using odai::test::db_contract::make_semantic_space;

namespace
{
std::string read_stored_model_type(const DBConfig& db_config, const ModelName& name)
{
  SQLite::Database db(db_config.m_dbPath, SQLite::OPEN_READONLY);
  SQLite::Statement query(db, "SELECT type FROM models WHERE name = :name LIMIT 1");
  query.bind(":name", name);
  if (!query.executeStep())
  {
    return {};
  }
  return query.getColumn("type").getString();
}

class OdaiSqliteDbTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                        std::to_string(reinterpret_cast<std::uintptr_t>(this));
    m_rootPath = fs::temp_directory_path() / ("odai_sqlite_db_test_" + suffix);
    m_mediaPath = m_rootPath / "media";
    fs::create_directories(m_mediaPath);
  }

  void TearDown() override
  {
    if (m_db != nullptr)
    {
      m_db->close();
    }
    std::error_code ec;
    fs::remove_all(m_rootPath, ec);
  }

  DBConfig db_config() const { return {SQLITE_DB, (m_rootPath / "odai.db").string(), m_mediaPath.string()}; }

  OdaiSqliteDb& initialized_db()
  {
    if (m_db == nullptr)
    {
      m_db = std::make_unique<OdaiSqliteDb>(db_config());
      EXPECT_TRUE(m_db->initialize_db().has_value());
    }
    return *m_db;
  }

  fs::path m_rootPath;
  fs::path m_mediaPath;
  std::unique_ptr<OdaiSqliteDb> m_db;
};

TEST_F(OdaiSqliteDbTest, InitializeDbCreatesDatabaseFile)
{
  OdaiSqliteDb db(db_config());
  EXPECT_FALSE(fs::exists(m_rootPath / "odai.db"));

  EXPECT_TRUE(db.initialize_db().has_value());
  EXPECT_TRUE(fs::exists(m_rootPath / "odai.db"));
}

TEST_F(OdaiSqliteDbTest, CloseIsIdempotentAfterInitialize)
{
  OdaiSqliteDb db(db_config());
  ASSERT_TRUE(db.initialize_db().has_value());

  db.close();
  db.close();
}

TEST_F(OdaiSqliteDbTest, InitializeDbCanReopenAfterClose)
{
  OdaiSqliteDb db(db_config());
  ASSERT_TRUE(db.initialize_db().has_value());
  db.close();

  EXPECT_TRUE(db.initialize_db().has_value());
  db.close();
}

TEST_F(OdaiSqliteDbTest, ConstructorRejectsEmptyDbPath)
{
  EXPECT_THROW(OdaiSqliteDb(DBConfig{SQLITE_DB, "", m_mediaPath.string()}), std::invalid_argument);
}

TEST_F(OdaiSqliteDbTest, ConstructorRejectsEmptyMediaStorePath)
{
  EXPECT_THROW(OdaiSqliteDb(DBConfig{SQLITE_DB, (m_rootPath / "odai.db").string(), ""}), std::invalid_argument);
}

TEST_F(OdaiSqliteDbTest, ConstructorRejectsNonSqliteDbType)
{
  EXPECT_THROW(OdaiSqliteDb(DBConfig{static_cast<DBType>(1), (m_rootPath / "odai.db").string(), m_mediaPath.string()}),
               std::invalid_argument);
}

TEST_F(OdaiSqliteDbTest, InitializeDbCreatesMediaStoreDirectory)
{
  const fs::path auto_media = m_rootPath / "auto_created_media";
  ASSERT_FALSE(fs::exists(auto_media));

  OdaiSqliteDb db(DBConfig{SQLITE_DB, (m_rootPath / "auto.db").string(), auto_media.string()});
  ASSERT_TRUE(db.initialize_db().has_value());
  EXPECT_TRUE(fs::exists(auto_media));
  db.close();
}

TEST_F(OdaiSqliteDbTest, RegisterEmbeddingModelFilesPersistsSqliteType)
{
  OdaiSqliteDb& db = initialized_db();
  const ModelFiles files = make_model_files(ModelType::EMBEDDING, {{"base_model_path", "/tmp/embed.gguf"}});
  const std::string checksums = R"({"base_model_path":"embed_hash"})";

  ASSERT_TRUE(db.register_model_files("embed-model", files, checksums).has_value());

  OdaiResult<ModelFiles> loaded = db.get_model_files("embed-model");
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded.value(), files);
  EXPECT_EQ(loaded->m_modelType, ModelType::EMBEDDING);
  EXPECT_EQ(read_stored_model_type(db_config(), "embed-model"), "EMBEDDING");
}

TEST_F(OdaiSqliteDbTest, UpdateModelFilesReplacesStoredSqliteType)
{
  OdaiSqliteDb& db = initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  const std::string checksums = R"({"base_model_path":"llm_hash"})";
  ASSERT_TRUE(db.register_model_files("replace-type-model", files, checksums).has_value());
  ASSERT_EQ(read_stored_model_type(db_config(), "replace-type-model"), "LLM");

  const ModelFiles replacement = make_model_files(ModelType::EMBEDDING, {{"base_model_path", "/tmp/embed.gguf"}});
  const std::string replacement_checksums = R"({"base_model_path":"embed_hash"})";
  ASSERT_TRUE(db.update_model_files("replace-type-model", replacement, replacement_checksums).has_value());

  OdaiResult<ModelFiles> loaded = db.get_model_files("replace-type-model");
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(loaded->m_modelType, ModelType::EMBEDDING);
  EXPECT_EQ(read_stored_model_type(db_config(), "replace-type-model"), "EMBEDDING");
}

TEST_F(OdaiSqliteDbTest, RegisterModelFilesRejectsInvalidChecksumJson)
{
  OdaiSqliteDb& db = initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});

  expect_error(db.register_model_files("model-a", files, ""), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, RegisterModelFilesRejectsInvalidModelType)
{
  OdaiSqliteDb& db = initialized_db();
  ModelFiles invalid_type_files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  invalid_type_files.m_modelType = static_cast<ModelType>(255);

  expect_error(db.register_model_files("model-a", invalid_type_files, R"({"base_model_path":"abc"})"),
               OdaiResultEnum::INVALID_ARGUMENT);
}

TEST_F(OdaiSqliteDbTest, UpdateModelFilesRejectsEmptyChecksums)
{
  OdaiSqliteDb& db = initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  const std::string checksums = R"({"base_model_path":"abc"})";
  ASSERT_TRUE(db.register_model_files("model-empty-cksum", files, checksums).has_value());

  const ModelFiles updated_files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-b.gguf"}});
  expect_error(db.update_model_files("model-empty-cksum", updated_files, ""), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, UpdateModelFilesRejectsInvalidModelType)
{
  OdaiSqliteDb& db = initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-a.gguf"}});
  const std::string checksums = R"({"base_model_path":"abc"})";
  ASSERT_TRUE(db.register_model_files("model-bad-type", files, checksums).has_value());

  ModelFiles invalid_type_files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/model-b.gguf"}});
  invalid_type_files.m_modelType = static_cast<ModelType>(255);
  expect_error(db.update_model_files("model-bad-type", invalid_type_files, R"({"base_model_path":"def"})"),
               OdaiResultEnum::INVALID_ARGUMENT);
}

TEST_F(OdaiSqliteDbTest, CreateSemanticSpaceRejectsInvalidEmbeddingConfig)
{
  OdaiSqliteDb& db = initialized_db();

  SemanticSpaceConfig invalid_config = make_semantic_space("bad-embed");
  invalid_config.m_embeddingModelConfig.m_modelName.clear();

  expect_error(db.create_semantic_space(invalid_config), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, CreateSemanticSpaceRejectsEmptyName)
{
  OdaiSqliteDb& db = initialized_db();

  SemanticSpaceConfig invalid_space = make_semantic_space("");
  expect_error(db.create_semantic_space(invalid_space), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, CreateChatRejectsInvalidLlmModelConfig)
{
  OdaiSqliteDb& db = initialized_db();

  ChatConfig invalid_config = make_chat_config();
  invalid_config.m_llmModelConfig.m_modelName.clear();

  expect_error(db.create_chat("chat-invalid-llm", invalid_config), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, CreateChatRejectsEmptySystemPrompt)
{
  OdaiSqliteDb& db = initialized_db();
  ChatConfig invalid_config = make_chat_config();
  invalid_config.m_systemPrompt.clear();

  expect_error(db.create_chat("chat-a", invalid_config), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, InsertChatMessagesRejectsAudioMemoryBufferAndRollsBackBatch)
{
  OdaiSqliteDb& db = initialized_db();
  const ChatConfig config = make_chat_config();
  ASSERT_TRUE(db.create_chat("chat-audio-reject", config).has_value());

  ChatMessage valid_message = make_chat_message("user", "committed only if whole batch is valid");
  ChatMessage audio_message;
  audio_message.m_role = "assistant";
  audio_message.m_contentItems.push_back(InputItem{InputItemType::MEMORY_BUFFER, {0x01, 0x02}, "audio/wav"});
  audio_message.m_messageMetadata = {};

  expect_error(db.insert_chat_messages("chat-audio-reject", {valid_message, audio_message}),
               OdaiResultEnum::VALIDATION_FAILED);

  OdaiResult<std::vector<ChatMessage>> history = db.get_chat_history("chat-audio-reject");
  ASSERT_TRUE(history.has_value());
  ASSERT_EQ(history->size(), 1U);
  EXPECT_EQ((*history)[0].m_role, "system");
}

TEST_F(OdaiSqliteDbTest, MultipleBatchInsertsPreserveSequenceOrder)
{
  OdaiSqliteDb& db = initialized_db();
  const ChatConfig config = make_chat_config();
  ASSERT_TRUE(db.create_chat("chat-order", config).has_value());

  ASSERT_TRUE(db.insert_chat_messages("chat-order", {make_chat_message("user", "First")}).has_value());
  ASSERT_TRUE(db.insert_chat_messages("chat-order",
                                      {make_chat_message("assistant", "Second"), make_chat_message("user", "Third")})
                  .has_value());

  OdaiResult<std::vector<ChatMessage>> history = db.get_chat_history("chat-order");
  ASSERT_TRUE(history.has_value());
  ASSERT_EQ(history->size(), 4U);
  EXPECT_EQ((*history)[0].m_role, "system");
  EXPECT_EQ(bytes_to_string((*history)[1].m_contentItems[0].m_data), "First");
  EXPECT_EQ(bytes_to_string((*history)[2].m_contentItems[0].m_data), "Second");
  EXPECT_EQ(bytes_to_string((*history)[3].m_contentItems[0].m_data), "Third");
}

TEST_F(OdaiSqliteDbTest, ChatMessageCreatedAtTimestampIsPopulated)
{
  OdaiSqliteDb& db = initialized_db();
  const ChatConfig config = make_chat_config();
  ASSERT_TRUE(db.create_chat("chat-ts", config).has_value());
  ASSERT_TRUE(db.insert_chat_messages("chat-ts", {make_chat_message("user", "timestamped")}).has_value());

  OdaiResult<std::vector<ChatMessage>> history = db.get_chat_history("chat-ts");
  ASSERT_TRUE(history.has_value());
  ASSERT_GE(history->size(), 2U);

  for (const auto& msg : history.value())
  {
    EXPECT_GT(msg.m_createdAt, 0U);
  }
}

TEST_F(OdaiSqliteDbTest, StoreMediaItemFileContentMatchesOriginal)
{
  OdaiSqliteDb& db = initialized_db();

  const std::vector<uint8_t> original_data = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  InputItem memory_image{InputItemType::MEMORY_BUFFER, original_data, "image/png"};
  OdaiResult<InputItem> stored = db.store_media_item(memory_image);
  ASSERT_TRUE(stored.has_value());

  const fs::path cached_path = bytes_to_string(stored->m_data);
  std::ifstream file(cached_path, std::ios::binary);
  ASSERT_TRUE(file.is_open());
  const std::vector<uint8_t> cached_data{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  EXPECT_EQ(cached_data, original_data);
}

TEST_F(OdaiSqliteDbTest, StoreMediaItemSourceFileCachePathIsInsideMediaStoreAndNotOriginal)
{
  OdaiSqliteDb& db = initialized_db();
  const fs::path source_path = m_rootPath / "source-audio.bin";
  {
    std::ofstream out(source_path, std::ios::binary);
    out << "audio-bytes";
  }

  InputItem file_audio{InputItemType::FILE_PATH, string_to_bytes(source_path.string()), "audio/wav"};
  OdaiResult<InputItem> stored_file = db.store_media_item(file_audio);
  ASSERT_TRUE(stored_file.has_value());

  const fs::path file_cache_path = bytes_to_string(stored_file->m_data);
  EXPECT_TRUE(fs::exists(file_cache_path));
  EXPECT_NE(file_cache_path, source_path);
  EXPECT_EQ(file_cache_path.parent_path(), m_mediaPath);
}

TEST_F(OdaiSqliteDbTest, StoreMediaItemRejectsNonExistentFilePath)
{
  OdaiSqliteDb& db = initialized_db();

  InputItem missing_file{InputItemType::FILE_PATH, string_to_bytes((m_rootPath / "nonexistent.png").string()),
                         "image/png"};
  ASSERT_FALSE(db.store_media_item(missing_file).has_value());
}

TEST_F(OdaiSqliteDbTest, StoreMediaItemRejectsEmptyMemoryBuffer)
{
  OdaiSqliteDb& db = initialized_db();

  expect_error(db.store_media_item(InputItem{InputItemType::MEMORY_BUFFER, {}, "text/plain"}),
               OdaiResultEnum::INVALID_ARGUMENT);
}

TEST_F(OdaiSqliteDbTest, StoreMediaItemRejectsUnsupportedMemoryBufferMimeType)
{
  OdaiSqliteDb& db = initialized_db();

  expect_error(
      db.store_media_item(InputItem{InputItemType::MEMORY_BUFFER, string_to_bytes("data"), "application/json"}),
      OdaiResultEnum::INVALID_ARGUMENT);
}

TEST_F(OdaiSqliteDbTest, StoreMediaItemRejectsTextFilePathInput)
{
  OdaiSqliteDb& db = initialized_db();

  expect_error(db.store_media_item(InputItem{InputItemType::FILE_PATH, string_to_bytes("/tmp/text.txt"), "text/plain"}),
               OdaiResultEnum::INVALID_ARGUMENT);
}

TEST_F(OdaiSqliteDbTest, CommitTransactionReturnsValidationFailedWithoutActiveTransaction)
{
  OdaiSqliteDb& db = initialized_db();
  expect_error(db.commit_transaction(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST_F(OdaiSqliteDbTest, RollbackTransactionSucceedsWithoutActiveTransaction)
{
  OdaiSqliteDb& db = initialized_db();
  EXPECT_TRUE(db.rollback_transaction().has_value());
}

TEST_F(OdaiSqliteDbTest, TripleNestedTransactionBehavesCorrectly)
{
  OdaiSqliteDb& db = initialized_db();
  const ModelFiles files = make_model_files(ModelType::LLM, {{"base_model_path", "/tmp/triple.gguf"}});
  const std::string checksums = R"({"base_model_path":"triple"})";

  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.begin_transaction().has_value());
  ASSERT_TRUE(db.register_model_files("triple-nested", files, checksums).has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());
  ASSERT_TRUE(db.commit_transaction().has_value());

  EXPECT_TRUE(db.get_model_files("triple-nested").has_value());
}

} // namespace
