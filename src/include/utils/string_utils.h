#pragma once

#include <cstddef>
#include <string>

using namespace std;

/// Returns the length of the string that is safe to send as valid UTF-8.
/// Scans backwards from the end of the string to ensure the last character is complete.
/// If the string ends with an incomplete multi-byte UTF-8 sequence, returns the length excluding the incomplete bytes.
/// @param buffer The string buffer to check
/// @return The safe length in bytes that represents complete UTF-8 characters, or 0 if buffer is empty
size_t get_safe_utf8_length(const string& buffer);