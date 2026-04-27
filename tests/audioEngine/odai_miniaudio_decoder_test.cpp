#include "audioEngine/odai_miniaudio_decoder.h"
#include "odai_decoder_test_helpers.h"
#include "odai_test_helpers.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using odai::test::file_input;
using odai::test::memory_input;
using odai::test::decoder::data_path;
using odai::test::decoder::expect_valid_audio_shape;

namespace
{

struct AudioFixtureCase
{
  std::string relative_path;
  std::string mime_type;
};

void expect_decodes_audio_fixture_to_target_spec(const AudioFixtureCase& fixture)
{
  OdaiMiniAudioDecoder decoder;
  const OdaiAudioTargetSpec target_spec{16'000, 1};
  OdaiDecodedAudio audio;

  const auto result =
      decoder.decode_to_spec(memory_input(data_path(fixture.relative_path), fixture.mime_type), target_spec, audio);

  ASSERT_TRUE(result.has_value()) << fixture.relative_path;
  expect_valid_audio_shape(audio);
  EXPECT_EQ(audio.m_sampleRate, 16'000U) << fixture.relative_path;
  EXPECT_EQ(audio.m_channels, 1U) << fixture.relative_path;
}

void expect_decodes_audio_file_fixture_to_target_spec(const AudioFixtureCase& fixture)
{
  OdaiMiniAudioDecoder decoder;
  const OdaiAudioTargetSpec target_spec{16'000, 1};
  OdaiDecodedAudio audio;

  const auto result =
      decoder.decode_to_spec(file_input(data_path(fixture.relative_path), fixture.mime_type), target_spec, audio);

  ASSERT_TRUE(result.has_value()) << fixture.relative_path;
  expect_valid_audio_shape(audio);
  EXPECT_EQ(audio.m_sampleRate, 16'000U) << fixture.relative_path;
  EXPECT_EQ(audio.m_channels, 1U) << fixture.relative_path;
}

} // namespace

TEST(OdaiMiniAudioDecoderTest, ReportsExactSupportedFormatList)
{
  OdaiMiniAudioDecoder decoder;

  for (const std::string& format : {"wav", "mp3", "flac"})
  {
    EXPECT_TRUE(decoder.is_supported(format)) << format;
  }

  for (const std::string& format : {"ogg", "aac", "m4a", "opus", "png", ""})
  {
    EXPECT_FALSE(decoder.is_supported(format)) << format;
  }
}

TEST(OdaiMiniAudioDecoderTest, TreatsSupportedFormatsCaseInsensitively)
{
  OdaiMiniAudioDecoder decoder;

  for (const std::string& format : {"WAV", "Mp3", "FlAc"})
  {
    EXPECT_TRUE(decoder.is_supported(format)) << format;
  }
}

TEST(OdaiMiniAudioDecoderTest, DecodesMiniaudioFixtureMatrixToTargetSpec)
{
  const std::vector<AudioFixtureCase> fixtures{
      {"audio/tiny_stereo_44100.wav", "audio/wav"},
      {"audio/Echoes_of_Unseen_Light.mp3", "audio/mpeg"},
      {"audio/tiny_stereo_44100.flac", "audio/flac"},
  };

  for (const AudioFixtureCase& fixture : fixtures)
  {
    expect_decodes_audio_fixture_to_target_spec(fixture);
  }
}

TEST(OdaiMiniAudioDecoderTest, DecodesCompressedFilePathFixturesToTargetSpec)
{
  const std::vector<AudioFixtureCase> fixtures{
      {"audio/Echoes_of_Unseen_Light.mp3", "audio/mpeg"},
      {"audio/tiny_stereo_44100.flac", "audio/flac"},
  };

  for (const AudioFixtureCase& fixture : fixtures)
  {
    expect_decodes_audio_file_fixture_to_target_spec(fixture);
  }
}
