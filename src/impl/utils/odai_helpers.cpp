#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "odai_logger.h"
#include "utils/odai_helpers.h"
#include "xxhash.h"
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

ChatId generate_chat_id()
{
  // Simple random ID generation
  return std::string("chat_") + std::to_string(rand()) + "_" + "t" + std::to_string(time(nullptr));
}

OdaiResult<std::string> calculate_file_checksum(const std::string& path)
{
  if (path.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Empty path passed for checksum calculation");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to open file for checksum calculation: {}", path);
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  // Use XXH3_64bits for speed and simple checksum
  XXH3_state_t* state = XXH3_createState();
  if (state == nullptr)
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to allocate XXH3 state for file checksum");
    return unexpected_internal_error();
  }

  if (XXH3_64bits_reset(state) != XXH_OK)
  {
    XXH3_freeState(state);
    ODAI_LOG(ODAI_LOG_ERROR, "Failed to reset XXH3 state for file checksum");
    return unexpected_internal_error();
  }

  const uint32_t buffer_size = static_cast<const uint32_t>(512 * BYTES_PER_KB); // 512KB buffer
  std::vector<char> buffer(buffer_size);

  while (file.read(buffer.data(), buffer_size))
  {
    XXH3_64bits_update(state, buffer.data(), file.gcount());
  }
  // Handle remaining bytes
  XXH3_64bits_update(state, buffer.data(), file.gcount());

  XXH64_hash_t hash = XXH3_64bits_digest(state);
  XXH3_freeState(state);

  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}

OdaiResult<std::string> calculate_data_checksum(const std::vector<uint8_t>& data)
{
  if (data.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "Empty data passed for checksum calculation");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  XXH64_hash_t hash = XXH3_64bits(data.data(), data.size());

  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}

OdaiResult<std::string> calculate_model_checksums(const ModelFiles& files)
{
  if (files.m_entries.empty())
  {
    ODAI_LOG(ODAI_LOG_ERROR, "No model file entries passed for checksum calculation");
    return tl::unexpected(OdaiResultEnum::INVALID_ARGUMENT);
  }

  nlohmann::json checksums_json = nlohmann::json::object();
  for (const auto& [key, path] : files.m_entries)
  {
    OdaiResult<std::string> checksum_res = calculate_file_checksum(path);
    if (!checksum_res)
    {
      ODAI_LOG(ODAI_LOG_ERROR, "Failed to calculate checksum for model file entry '{}', error code: {}", key,
               static_cast<std::uint32_t>(checksum_res.error()));
      return tl::unexpected(checksum_res.error());
    }

    checksums_json[key] = checksum_res.value();
  }

  return checksums_json.dump();
}

std::filesystem::path get_module_directory_from_address(const void* symbol_address)
{
#ifdef _WIN32
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(symbol_address), &module) == 0)
  {
    return std::filesystem::current_path();
  }

  std::wstring module_path(MAX_PATH, L'\0');
  DWORD path_length = GetModuleFileNameW(module, module_path.data(), static_cast<DWORD>(module_path.size()));
  while (path_length == module_path.size())
  {
    module_path.resize(module_path.size() * 2);
    path_length = GetModuleFileNameW(module, module_path.data(), static_cast<DWORD>(module_path.size()));
  }

  if (path_length == 0)
  {
    return std::filesystem::current_path();
  }

  module_path.resize(path_length);
  return std::filesystem::path(module_path).parent_path();
#else
  Dl_info module_info{};
  if (dladdr(const_cast<void*>(symbol_address), &module_info) == 0 || module_info.dli_fname == nullptr)
  {
    return std::filesystem::current_path();
  }

  return std::filesystem::path(module_info.dli_fname).parent_path();
#endif
}

uint64_t bytes_to_mb(uint64_t bytes)
{
  return bytes / BYTES_PER_MB;
}
