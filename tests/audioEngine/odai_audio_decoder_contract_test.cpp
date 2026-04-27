#include "audioEngine/odai_audio_decoder.h"
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
using odai::test::decoder::expect_valid_audio_shape;

namespace
{
std::unique_ptr<IOdaiAudioDecoder> make_decoder()
{
  std::unique_ptr<IOdaiAudioDecoder> decoder = OdaiSdk::get_new_odai_audio_decoder_instance();
  if (decoder == nullptr)
  {
    throw std::runtime_error("No audio decoder implementation is enabled for this build");
  }
  return decoder;
}
} // namespace

TEST(IOdaiAudioDecoderContractTest, DecodesAudioFromFilePath)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{44'100, 2};

  const auto result =
      decoder->decode_to_spec(file_input(data_path("audio/tiny_stereo_44100.wav"), "audio/wav"), target_spec, audio);

  ASSERT_TRUE(result.has_value());
  expect_valid_audio_shape(audio);
  EXPECT_EQ(audio.m_sampleRate, 44'100U);
  EXPECT_EQ(audio.m_channels, 2U);
}

TEST(IOdaiAudioDecoderContractTest, DecodesAudioFromMemoryBuffer)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{44'100, 2};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("audio/tiny_stereo_44100.wav"), "audio/wav"), target_spec, audio);

  ASSERT_TRUE(result.has_value());
  expect_valid_audio_shape(audio);
  EXPECT_EQ(audio.m_sampleRate, 44'100U);
  EXPECT_EQ(audio.m_channels, 2U);
}

TEST(IOdaiAudioDecoderContractTest, AcceptsAudioMimeTypePrefixCaseInsensitively)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{44'100, 2};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("audio/tiny_stereo_44100.wav"), "Audio/WAV"), target_spec, audio);

  ASSERT_TRUE(result.has_value());
  expect_valid_audio_shape(audio);
}

TEST(IOdaiAudioDecoderContractTest, ResamplesToTargetSampleRate)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 2};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("audio/tiny_stereo_44100.wav"), "audio/wav"), target_spec, audio);

  ASSERT_TRUE(result.has_value());
  expect_valid_audio_shape(audio);
  EXPECT_EQ(audio.m_sampleRate, 16'000U);
  EXPECT_EQ(audio.m_channels, 2U);
}

TEST(IOdaiAudioDecoderContractTest, ConvertsToTargetChannelCount)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{44'100, 1};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("audio/tiny_stereo_44100.wav"), "audio/wav"), target_spec, audio);

  ASSERT_TRUE(result.has_value());
  expect_valid_audio_shape(audio);
  EXPECT_EQ(audio.m_sampleRate, 44'100U);
  EXPECT_EQ(audio.m_channels, 1U);
}

TEST(IOdaiAudioDecoderContractTest, DecodedSampleBufferShapeMatchesMetadata)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{44'100, 2};

  const auto result =
      decoder->decode_to_spec(memory_input(data_path("audio/tiny_stereo_44100.wav"), "audio/wav"), target_spec, audio);

  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(audio.m_sampleRate, 44'100U);
  ASSERT_EQ(audio.m_channels, 2U);
  EXPECT_GT(audio.m_samples.size(), 0U);
  EXPECT_EQ(audio.m_samples.size() % 2U, 0U);
}

TEST(IOdaiAudioDecoderContractTest, RejectsUnsupportedMimeTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, read_file(data_path("audio/tiny_stereo_44100.wav")), "text/plain"},
      target_spec, audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiAudioDecoderContractTest, RejectsNonAudioMimeTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, read_file(data_path("audio/tiny_stereo_44100.wav")), "image/png"},
      target_spec, audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiAudioDecoderContractTest, RejectsEmptyMimeTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, read_file(data_path("audio/tiny_stereo_44100.wav")), ""}, target_spec,
      audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiAudioDecoderContractTest, RejectsEmptyAudioBufferWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result =
      decoder->decode_to_spec(InputItem{InputItemType::MEMORY_BUFFER, {}, "audio/wav"}, target_spec, audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}

TEST(IOdaiAudioDecoderContractTest, RejectsCorruptAudioBufferWithValidationFailed)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result = decoder->decode_to_spec(
      InputItem{InputItemType::MEMORY_BUFFER, std::vector<uint8_t>{0x00, 0x01, 0x02, 0x03}, "audio/wav"}, target_spec,
      audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST(IOdaiAudioDecoderContractTest, RejectsCorruptFilePathContentWithValidationFailed)
{
  const auto corrupt_path = std::filesystem::temp_directory_path() / "odai_corrupt_audio_contract_test.bin";
  {
    std::ofstream out(corrupt_path, std::ios::binary);
    out << "not-real-audio-data";
  }

  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result = decoder->decode_to_spec(file_input(corrupt_path, "audio/wav"), target_spec, audio);

  std::filesystem::remove(corrupt_path);
  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST(IOdaiAudioDecoderContractTest, RejectsNonExistentFilePathWithValidationFailed)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result =
      decoder->decode_to_spec(file_input(data_path("audio/does_not_exist.wav"), "audio/wav"), target_spec, audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::VALIDATION_FAILED);
}

TEST(IOdaiAudioDecoderContractTest, RejectsUnsupportedInputItemTypeWithInvalidArgument)
{
  const auto decoder = make_decoder();
  OdaiDecodedAudio audio;
  const OdaiAudioTargetSpec target_spec{16'000, 1};

  const auto result =
      decoder->decode_to_spec(InputItem{static_cast<InputItemType>(255), {0x01}, "audio/wav"}, target_spec, audio);

  ASSERT_FALSE(result.has_value());
  EXPECT_EQ(result.error(), OdaiResultEnum::INVALID_ARGUMENT);
}
