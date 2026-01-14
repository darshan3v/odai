#pragma once

#include <string>
#include <format>
#include <utility>

#include "odai_public.h"

/// Logger class that provides formatted logging functionality with configurable log levels and callbacks.
/// Supports format string arguments and automatically prefixes log messages with "[odai]".
class ODAILogger
{

private:
    odai_log_callback_fn callback = nullptr;
    void *user_data = nullptr;
    OdaiLogLevel log_level = ODAI_LOG_INFO;

public:
    /// Sets the logging callback function and user data pointer.
    /// The callback will be invoked for each log message that passes the log level filter.
    /// @param callback Function to call for each log message, or nullptr to disable logging
    /// @param user_data User-provided data passed to the callback function
    void odai_set_logger(odai_log_callback_fn callback, void *user_data);
    
    /// Sets the minimum log level for messages to be processed.
    /// Only messages with level less than or equal to this level will be logged.
    /// @param log_level Minimum log level (ODAI_LOG_ERROR, ODAI_LOG_WARN, ODAI_LOG_INFO, or ODAI_LOG_DEBUG)
    void odai_set_log_level(OdaiLogLevel log_level);

    /// Logs a formatted message at the specified log level.
    /// The message is formatted using std::format and prefixed with "[odai] ".
    /// If no callback is set or the log level is above the configured threshold, the message is silently ignored.
    /// All exceptions during formatting are caught and ignored.
    /// @param level Log level for this message
    /// @param fmt Format string compatible with std::format
    /// @param args Arguments to format into the message
    template <typename... Args>
    void log(OdaiLogLevel level, std::format_string<Args...> fmt, Args &&...args)
    {
        try
        {
            if (!this->callback)
                return;

            if(level > this->log_level)
                return;

            std::string msg = format(
                fmt,
                std::forward<Args>(args)...);

            msg.insert(0, "[odai] ");

            this->callback(level, msg.c_str(), this->user_data);
        }
        catch (...)
        {
        }
    }
};

// Global logger is now managed by ODAISdk singleton in odai_sdk.h
// Macro ODAI_LOG is also defined there