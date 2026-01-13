#include <cstring>

#include "utils/string_utils.h"

// Helper: Returns the length of the string that is safe to send as valid UTF-8
size_t get_safe_utf8_length(const string& buffer) {
    size_t len = buffer.size();
    if (len == 0) return 0;
    
    // Scan backwards up to 4 bytes to find the start of the last character
    for (size_t i = 0; i < 4 && i < len; ++i) {
        unsigned char c = static_cast<unsigned char>(buffer[len - 1 - i]);
        // ASCII (0xxxxxxx) -> Safe immediately
        if ((c & 0x80) == 0x00) return len;
        // Continuation (10xxxxxx) -> Keep scanning
        if ((c & 0xC0) == 0x80) continue;
        // 2-byte start (110xxxxx) -> Need 2 bytes total
        if ((c & 0xE0) == 0xC0) return (i >= 1) ? len : len - 1 - i;
        // 3-byte start (1110xxxx) -> Need 3 bytes total
        if ((c & 0xF0) == 0xE0) return (i >= 2) ? len : len - 1 - i;
        // 4-byte start (11110xxx) -> Need 4 bytes total
        if ((c & 0xF8) == 0xF0) return (i >= 3) ? len : len - 1 - i;
    }
    return len;
}

void set_cstr_and_len(string& string_data, char* c_str, size_t* cstr_len)
{
    *cstr_len = string_data.size() + 1;
    strncpy(c_str, string_data.c_str(), *cstr_len);
}