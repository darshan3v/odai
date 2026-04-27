#pragma once

#include "types/odai_result.h"
#include "types/odai_types.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace odai::test
{
inline std::string bytes_to_string(const std::vector<uint8_t>& bytes)
{
  return std::string(bytes.begin(), bytes.end());
}

inline std::vector<uint8_t> string_to_bytes(const std::string& value)
{
  return std::vector<uint8_t>(value.begin(), value.end());
}

inline std::vector<uint8_t> read_file(const std::filesystem::path& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream.is_open())
  {
    throw std::runtime_error("Failed to open test data file: " + path.string());
  }

  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

inline InputItem file_input(const std::filesystem::path& path, const std::string& mime_type)
{
  return InputItem{InputItemType::FILE_PATH, string_to_bytes(path.string()), mime_type};
}

inline InputItem memory_input(const std::filesystem::path& path, const std::string& mime_type)
{
  return InputItem{InputItemType::MEMORY_BUFFER, read_file(path), mime_type};
}

template <typename T>
void expect_error(const OdaiResult<T>& result, OdaiResultEnum expected_error)
{
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), expected_error);
}
} // namespace odai::test
