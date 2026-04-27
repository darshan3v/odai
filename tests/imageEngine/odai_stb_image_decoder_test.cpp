#include "imageEngine/odai_stb_image_decoder.h"
#include "odai_decoder_test_helpers.h"
#include "odai_test_helpers.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

using odai::test::file_input;
using odai::test::memory_input;
using odai::test::decoder::data_path;
using odai::test::decoder::expect_valid_image_shape;

namespace
{

struct ImageFixtureCase
{
  std::string relative_path;
  std::string mime_type;
  uint32_t expected_width;
  uint32_t expected_height;
};

void expect_decodes_supported_image_format(const ImageFixtureCase& fixture)
{
  OdaiStbImageDecoder decoder;
  const OdaiImageTargetSpec target_spec{0, 0, 3};
  OdaiDecodedImage image;

  const auto result =
      decoder.decode_to_spec(memory_input(data_path(fixture.relative_path), fixture.mime_type), target_spec, image);

  ASSERT_TRUE(result.has_value()) << fixture.relative_path;
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, fixture.expected_width) << fixture.relative_path;
  EXPECT_EQ(image.m_height, fixture.expected_height) << fixture.relative_path;
  EXPECT_EQ(image.m_channels, 3U) << fixture.relative_path;
}

} // namespace

TEST(OdaiStbImageDecoderTest, ReportsExactSupportedFormatList)
{
  OdaiStbImageDecoder decoder;

  for (const std::string& format : {"jpg", "jpeg", "png", "bmp", "tga", "gif", "hdr", "pnm", "ppm", "pgm", "psd"})
  {
    EXPECT_TRUE(decoder.is_supported(format)) << format;
  }

  for (const std::string& format : {"pic", "webp", "pdf", "mp3", "ogg", ""})
  {
    EXPECT_FALSE(decoder.is_supported(format)) << format;
  }
}

TEST(OdaiStbImageDecoderTest, TreatsSupportedFormatsCaseInsensitively)
{
  OdaiStbImageDecoder decoder;

  for (const std::string& format : {"PNG", "JpEg", "PsD", "PgM"})
  {
    EXPECT_TRUE(decoder.is_supported(format)) << format;
  }
}

TEST(OdaiStbImageDecoderTest, DecodesStbImageFixtureMatrixToTargetSpec)
{
  const std::vector<ImageFixtureCase> fixtures{
      {"images/tiny_rgba.png", "image/png", 4U, 3U},
      {"images/tiny_rgb.bmp", "image/bmp", 4U, 3U},
      {"images/sample_chamaleon.jpg", "image/jpeg", 300U, 168U},
      {"images/tiny_rgb.tga", "image/x-tga", 4U, 3U},
      {"images/tiny_rgb.gif", "image/gif", 4U, 3U},
      {"images/tiny_rgb.hdr", "image/vnd.radiance", 4U, 3U},
      {"images/tiny_rgb.ppm", "image/x-portable-pixmap", 4U, 3U},
      {"images/tiny_rgb.pnm", "image/x-portable-anymap", 4U, 3U},
      {"images/tiny_gray.pgm", "image/x-portable-graymap", 4U, 3U},
      {"images/tiny_rgb.psd", "image/vnd.adobe.photoshop", 4U, 3U},
  };

  for (const ImageFixtureCase& fixture : fixtures)
  {
    expect_decodes_supported_image_format(fixture);
  }
}

TEST(OdaiStbImageDecoderTest, DecodesBmpFromFilePath)
{
  OdaiStbImageDecoder decoder;
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 3};

  const auto result =
      decoder.decode_to_spec(file_input(data_path("images/tiny_rgb.bmp"), "image/bmp"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, 4U);
  EXPECT_EQ(image.m_height, 3U);
  EXPECT_EQ(image.m_channels, 3U);
}

TEST(OdaiStbImageDecoderTest, PreservesNativeChannelCountWhenTargetChannelsIsZero)
{
  OdaiStbImageDecoder decoder;
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result = decoder.decode_to_spec(
      memory_input(data_path("images/tiny_gray.pgm"), "image/x-portable-graymap"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_channels, 1U);
}
