#pragma once

#include <cstdint>
#include <string>
#include <vector>

/// Specifies the desired output format for the decoded audio.
/// By default, configured for multimodal models which expect 16kHz mono audio.
struct OdaiAudioTargetSpec
{
  uint32_t m_sampleRate;
  uint8_t m_channels;
};

/// Holds the raw float32 PCM samples and metadata after decoding.
struct OdaiDecodedAudio
{
  std::vector<float> m_samples;
  uint32_t m_sampleRate;
  uint8_t m_channels;
};

/// Pure virtual interface for decoding audio files.
class IOdaiAudioDecoder
{
public:
  virtual ~IOdaiAudioDecoder() = default;
  IOdaiAudioDecoder() = default;

  IOdaiAudioDecoder(const IOdaiAudioDecoder&) = delete;
  IOdaiAudioDecoder& operator=(const IOdaiAudioDecoder&) = delete;
  IOdaiAudioDecoder(IOdaiAudioDecoder&&) = delete;
  IOdaiAudioDecoder& operator=(IOdaiAudioDecoder&&) = delete;

  /// Decodes an audio file and resamples it to match the target specification.
  /// @param file_path The absolute to the input audio file.
  /// @param target_spec The required ouput's spec.
  /// @param decoded_audio The output decoded audio.
  /// @return true if decoding was successful, false otherwise.
  virtual bool decode_to_spec(const std::string& file_path, const OdaiAudioTargetSpec& target_spec,
                              OdaiDecodedAudio& decoded_audio) = 0;
};