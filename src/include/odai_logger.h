#pragma once

#include <format>
#include <string>
#include <utility>

#include "odai_public.h"

class OdaiLogger;

/// Bridge function to retrieve the global logger instance without requiring odai_sdk.h.
/// This prevents circular dependencies between the logger macro and the SDK.
/// @return Pointer to the active OdaiLogger instance, or nullptr if not available.
OdaiLogger* get_odai_logger();

/// Logger class that provides formatted logging functionality with configurable
/// log levels and callbacks. Supports format string arguments and automatically
/// prefixes log messages with "[odai]".
class OdaiLogger
{

private:
  OdaiLogCallbackFn m_callback = nullptr;
  void* m_userData = nullptr;
  OdaiLogLevel m_logLevel = ODAI_LOG_INFO;

public:
  /// Sets the logging callback function and user data pointer.
  /// The callback will be invoked for each log message that passes the log
  /// level filter.
  /// @param callback Function to call for each log message, or nullptr to
  /// disable logging
  /// @param user_data User-provided data passed to the callback function
  void set_logger(OdaiLogCallbackFn callback, void* user_data);

  /// Sets the minimum log level for messages to be processed.
  /// Only messages with level less than or equal to this level will be logged.
  /// @param log_level Minimum log level (ODAI_LOG_ERROR, ODAI_LOG_WARN,
  /// ODAI_LOG_INFO, or ODAI_LOG_DEBUG)
  void set_log_level(OdaiLogLevel log_level);

  /// Logs a formatted message at the specified log level.
  /// The message is formatted using std::format and prefixed with "[odai] ".
  /// If no callback is set or the log level is above the configured threshold,
  /// the message is silently ignored. All exceptions during formatting are
  /// caught and ignored.
  /// @param level Log level for this message
  /// @param fmt Format string compatible with std::format
  /// @param args Arguments to format into the message
  template <typename... Args>
  void log(OdaiLogLevel level, std::format_string<Args...> fmt, Args&&... args)
  {
    try
    {
      if (!this->m_callback)
      {
        return;
      }

      if (level > this->m_logLevel)
      {
        return;
      }

      std::string msg = format(fmt, std::forward<Args>(args)...);

      msg.insert(0, "[odai] ");

      this->m_callback(level, msg.c_str(), this->m_userData);
    }
    catch (...)
    {
    }
  }
};

#define ODAI_LOG(level, fmt, ...)                                                                                      \
  if (OdaiLogger* logger = get_odai_logger())                                                                          \
  logger->log(level, "[{}:{}] " fmt, __func__, __LINE__, ##__VA_ARGS__)