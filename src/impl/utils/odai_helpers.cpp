#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

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

std::string calculate_file_checksum(const std::string& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
  {
    return "";
  }

  // Use XXH3_64bits for speed and simple checksum
  XXH3_state_t* state = XXH3_createState();
  if (state == nullptr)
  {
    return "";
  }

  XXH3_64bits_reset(state);

  const uint32_t buffer_size = static_cast<const uint32_t>(64 * 1024); // 64KB buffer
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

std::string calculate_data_checksum(const std::vector<uint8_t>& data)
{
  if (data.empty())
  {
    return "";
  }

  XXH64_hash_t hash = XXH3_64bits(data.data(), data.size());

  std::stringstream ss;
  ss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return ss.str();
}

std::string calculate_model_checksums(const ModelFiles& files)
{
  nlohmann::json checksums_json;
  for (const auto& [key, path] : files.m_entries)
  {
    std::string checksum = calculate_file_checksum(path);
    if (!checksum.empty())
    {
      checksums_json[key] = checksum;
    }
    // if checksum fails, it might just be an optional file that doesn't exist, but we skip it.
  }

  if (checksums_json.empty() && !files.m_entries.empty())
  {
    return ""; // Failed to calculate any checksums
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