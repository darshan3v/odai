#pragma once

#include <filesystem>

#include "types/odai_result.h"
#include "types/odai_types.h"

/// Generates a unique chat session identifier.
/// The ID is composed of a "chat_" prefix, a random number, and the current timestamp.
/// @return A unique string identifier in the format "chat_<random>_<timestamp>"
ChatId generate_chat_id();

/// Calculates the XXH3 64-bit checksum of a file's contents.
/// @param path The absolute path to the file.
/// @return Hex checksum on success, or an unexpected OdaiResultEnum on failure.
OdaiResult<std::string> calculate_file_checksum(const std::string& path);

/// Calculates the XXH3 64-bit checksum of a byte array in memory.
/// @param data The vector containing the binary data.
/// @return Hex checksum on success, or an unexpected OdaiResultEnum on failure.
OdaiResult<std::string> calculate_data_checksum(const std::vector<uint8_t>& data);

/// Calculates checksums for all files in a ModelFiles struct.
/// @param files The Model Files struct containing paths.
/// @return JSON string containing key-value checksum pairs on success, or an unexpected OdaiResultEnum on failure.
OdaiResult<std::string> calculate_model_checksums(const ModelFiles& files);

/// Returns the directory of the loaded module that contains the given symbol address.
/// This can be used by any module to resolve its own shared library or executable directory by
/// passing the address of one of its own functions or static objects. Falls back to the process
/// current working directory if the module path cannot be resolved.
/// @param symbol_address Address of a symbol that resides in the target module.
/// @return Absolute path to the directory containing the shared library or executable for that symbol.
std::filesystem::path get_module_directory_from_address(const void* symbol_address);
