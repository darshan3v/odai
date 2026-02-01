#pragma once

#include "types/odai_types.h"

using namespace std;

/// Generates a unique chat session identifier.
/// The ID is composed of a "chat_" prefix, a random number, and the current timestamp.
/// @return A unique string identifier in the format "chat_<random>_<timestamp>"
ChatId generate_chat_id();

/// Calculates the XXHash checksum of a file.
/// @param path The path to the file.
/// @return The hex string representation of the checksum, or empty string on error.
string calculate_file_checksum(const string& path);