#include "imageEngine/odai_image_decoder.h"
#include "odai_decoder_test_helpers.h"
#include "odai_sdk.h"
#include "odai_test_helpers.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>

using odai::test::file_input;
using odai::test::memory_input;
using odai::test::read_file;
using odai::test::decoder::data_path;
using odai::test::decoder::expect_valid_image_shape;

namespace
{
std::unique_ptr<IOdaiImageDecoder> make_decoder()
{
  std::unique_ptr<IOdaiImageDecoder> decoder = OdaiSdk::get_new_odai_image_decoder_instance();
  if (decoder == nullptr)
  {
    throw std::runtime_error("No image decoder implementation is enabled for this build");
  }
  return decoder;
}
} // namespace

TEST(IOdaiImageDecoderContractTest, DecodesImageFromFilePath)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result =
      decoder->decode_to_spec(file_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, 4U);
  EXPECT_EQ(image.m_height, 3U);
  EXPECT_EQ(image.m_channels, 4U);
}

TEST(IOdaiImageDecoderContractTest, DecodesImageFromMemoryBuffer)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, 4U);
  EXPECT_EQ(image.m_height, 3U);
  EXPECT_EQ(image.m_channels, 4U);
}

TEST(IOdaiImageDecoderContractTest, AcceptsImageMimeTypePrefixCaseInsensitively)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "Image/PNG"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
}

TEST(IOdaiImageDecoderContractTest, ConvertsToTargetChannelCount)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 3};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, 4U);
  EXPECT_EQ(image.m_height, 3U);
  EXPECT_EQ(image.m_channels, 3U);
}

TEST(IOdaiImageDecoderContractTest, ResizesWithinTargetBounds)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{2, 2, 4};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_LE(image.m_width, 2U);
  EXPECT_LE(image.m_height, 2U);
  EXPECT_EQ(image.m_channels, 4U);
}

TEST(IOdaiImageDecoderContractTest, ResizesWithWidthConstraintOnly)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{2, 0, 4};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_LE(image.m_width, 2U);
  EXPECT_EQ(image.m_width, 2U);
  EXPECT_EQ(image.m_height, 1U);
  EXPECT_EQ(image.m_channels, 4U);
}

TEST(IOdaiImageDecoderContractTest, ResizesWithHeightConstraintOnly)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 2, 4};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_LE(image.m_height, 2U);
  EXPECT_EQ(image.m_width, 2U);
  EXPECT_EQ(image.m_height, 2U);
  EXPECT_EQ(image.m_channels, 4U);
}

TEST(IOdaiImageDecoderContractTest, KeepsOriginalDimensionsWhenBoundsAreZero)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;

  const auto result = decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"),
                                              OdaiImageTargetSpec{0, 0, 4}, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, 4U);
  EXPECT_EQ(image.m_height, 3U);
}

TEST(IOdaiImageDecoderContractTest, KeepsOriginalDimensionsWhenBoundsExceedImageDimensions)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;

  const auto result = decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"),
                                              OdaiImageTargetSpec{64, 64, 4}, image);

  ASSERT_TRUE(result.has_value());
  expect_valid_image_shape(image);
  EXPECT_EQ(image.m_width, 4U);
  EXPECT_EQ(image.m_height, 3U);
}

TEST(IOdaiImageDecoderContractTest, DecodedBufferShapeMatchesMetadata)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 3};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("images/tiny_rgba.png"), "image/png"), target_spec, image);

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(image.m_width, 4U);
  ASSERT_EQ(image.m_height, 3U);
  ASSERT_EQ(image.m_channels, 3U);
  EXPECT_EQ(image.m_pixels.size(), 36U);
}

TEST(IOdaiImageDecoderContractTest, RejectsUnsupportedMimeTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, read_file(data_path("images/tiny_rgba.png")), "text/plain"}, target_spec,
      image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiImageDecoderContractTest, RejectsNonImageMimeTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, read_file(data_path("images/tiny_rgba.png")), "audio/wav"}, target_spec,
      image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiImageDecoderContractTest, RejectsEmptyMimeTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, read_file(data_path("images/tiny_rgba.png")), ""}, target_spec, image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiImageDecoderContractTest, RejectsEmptyImageBufferWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result =
      decoder->decode_to_spec(InputItem{InputItemType::MEMORY_BUFFER, {}, "image/png"}, target_spec, image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiImageDecoderContractTest, RejectsCorruptImageBufferWithValidationFailed)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, std::vector<uint8_t>{0x00, 0x01, 0x02, 0x03}, "image/png"}, target_spec,
      image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST(IOdaiImageDecoderContractTest, RejectsCorruptFilePathContentWithValidationFailed)
{
  const auto corrupt_path = std::filesystem::temp_directory_path() / "odai_corrupt_image_contract_test.bin";
  {
    std::ofstream out(corrupt_path, std::ios::binary);
    out << "not-real-image-data";
  }

  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result = decoder->decode_to_spec(file_input(corrupt_path, "image/png"), target_spec, image);

  std::filesystem::remove(corrupt_path);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST(IOdaiImageDecoderContractTest, RejectsNonExistentFilePathWithValidationFailed)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result =
      decoder->decode_to_spec(file_input(data_path("images/does_not_exist.png"), "image/png"), target_spec, image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST(IOdaiImageDecoderContractTest, RejectsUnsupportedInputItemTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedImage image;
  const OdaiImageTargetSpec target_spec{0, 0, 0};

  const auto result =
      decoder->decode_to_spec(InputItem{static_cast<InputItemType>(255), {0x01}, "image/png"}, target_spec, image);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}
