#pragma once

#include "types/odai_types.h"

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace odai::test::decoder
{
inline std::filesystem::path data_path(const std::string& relative_path)
{
  return std::filesystem::path(TEST_DATA_DIR) / relative_path;
}

inline void expect_valid_image_shape(const OdaiDecodedImage& image)
{
  EXPECT_GT(image.m_width, 0U);
  EXPECT_GT(image.m_height, 0U);
  EXPECT_GT(image.m_channels, 0U);
  EXPECT_EQ(image.m_pixels.size(), static_cast<size_t>(image.m_width) * static_cast<size_t>(image.m_height) *
                                       static_cast<size_t>(image.m_channels));
}

inline void expect_valid_audio_shape(const OdaiDecodedAudio& audio)
{
  EXPECT_GT(audio.m_sampleRate, 0U);
  EXPECT_GT(audio.m_channels, 0U);
  EXPECT_FALSE(audio.m_samples.empty());
  EXPECT_EQ(audio.m_samples.size() % audio.m_channels, 0U);
  for (const float sample : audio.m_samples)
  {
    EXPECT_TRUE(std::isfinite(sample));
  }
}
} // namespace odai::test::decoder
