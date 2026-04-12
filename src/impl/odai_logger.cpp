#include "odai_logger.h"

#include <chrono>
#include <ctime>

std::string get_odai_log_timestamp()
{
  const auto now = std::chrono::system_clock::now();
  const auto now_time_t = std::chrono::system_clock::to_time_t(now);

  std::tm local_time{};
#ifdef _WIN32
  localtime_s(&local_time, &now_time_t);
#else
  localtime_r(&now_time_t, &local_time);
#endif

  return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}", local_time.tm_year + 1900, local_time.tm_mon + 1,
                     local_time.tm_mday, local_time.tm_hour, local_time.tm_min, local_time.tm_sec);
}

void OdaiLogger::set_logger(OdaiLogCallbackFn callback, void* user_data)
{
  this->m_callback = callback;
  this->m_userData = user_data;
}

void OdaiLogger::set_log_level(OdaiLogLevel log_level)
{
  this->m_logLevel = log_level;
}
