#pragma once

#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

/// Converts a given string to lowercase.
/// @param str The string to convert.
/// @return A new string containing the lowercase version of the input.
std::string to_lower(const std::string& str);

/// Returns the length of the string that is safe to send as valid UTF-8.
/// Scans backwards from the end of the string to ensure the last character is complete.
/// If the string ends with an incomplete multi-byte UTF-8 sequence, returns the length excluding the incomplete bytes.
/// @param buffer The string buffer to check
/// @return The safe length in bytes that represents complete UTF-8 characters, or 0 if buffer is empty
size_t get_safe_utf8_length(const std::string& buffer);

template <typename ValueType>
concept StringViewConvertible = std::convertible_to<ValueType, std::string_view>;

/// Joins a range of string-like values using the provided separator.
template <std::ranges::input_range Range>
  requires StringViewConvertible<std::ranges::range_reference_t<Range>>
std::string join_strings(const Range& values, std::string_view separator)
{
  return join_strings(values, separator, [](const auto& value) -> std::string_view { return value; });
}

/// Joins a range after mapping each element to a string-like value.
template <std::ranges::input_range Range, typename Formatter>
  requires std::invocable<Formatter, std::ranges::range_reference_t<Range>> &&
           StringViewConvertible<std::invoke_result_t<Formatter, std::ranges::range_reference_t<Range>>>
std::string join_strings(const Range& values, std::string_view separator, Formatter&& formatter)
{
  std::string joined;
  bool first = true;
  for (const auto& value : values)
  {
    auto formatted_value = std::invoke(std::forward<Formatter>(formatter), value);
    std::string_view value_view{formatted_value};
    if (!first)
    {
      joined.append(separator.data(), separator.size());
    }
    joined.append(value_view.data(), value_view.size());
    first = false;
  }
  return joined;
}
