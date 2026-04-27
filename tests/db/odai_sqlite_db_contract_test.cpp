#include "odai_db_contract_tests.h"

#include "db/odai_sqlite/odai_sqlite_db.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

namespace odai::test::db_contract
{
class OdaiSqliteDbContractFixture : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                        std::to_string(reinterpret_cast<std::uintptr_t>(this));
    m_rootPath = fs::temp_directory_path() / ("odai_sqlite_db_contract_test_" + suffix);
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

  std::unique_ptr<IOdaiDb> make_uninitialized_db() { return std::make_unique<OdaiSqliteDb>(db_config()); }

  IOdaiDb& initialized_db()
  {
    if (m_db == nullptr)
    {
      m_db = std::make_unique<OdaiSqliteDb>(db_config());
      EXPECT_TRUE(m_db->initialize_db().has_value());
    }
    return *m_db;
  }

  void reopen_db()
  {
    if (m_db != nullptr)
    {
      m_db->close();
      m_db.reset();
    }
    m_db = std::make_unique<OdaiSqliteDb>(db_config());
    ASSERT_TRUE(m_db->initialize_db().has_value());
  }

  InputItem make_source_file_item(const std::string& file_name, const std::string& contents,
                                  const std::string& mime_type)
  {
    const fs::path source_path = m_rootPath / file_name;
    std::ofstream out(source_path, std::ios::binary);
    EXPECT_TRUE(out.is_open());
    out << contents;
    return {InputItemType::FILE_PATH, string_to_bytes(source_path.string()), mime_type};
  }

private:
  fs::path m_rootPath;
  fs::path m_mediaPath;
  std::unique_ptr<IOdaiDb> m_db;
};

using SqliteContractImplementations = ::testing::Types<OdaiSqliteDbContractFixture>;

INSTANTIATE_TYPED_TEST_SUITE_P(SQLite, IOdaiDbContractTest, SqliteContractImplementations);

} // namespace odai::test::db_contract
