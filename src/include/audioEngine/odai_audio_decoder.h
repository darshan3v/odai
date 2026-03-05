#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types/odai_types.h"

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

  /// Checks if the given audio format is supported by this decoder.
  /// Provides documentation on what kind of file types/formats are supported.
  /// @param format The audio format extension (e.g., "wav", "mp3", "flac", "ogg").
  /// @return true if the format is supported, false otherwise.
  virtual bool is_supported(const std::string& format) const = 0;

  /// Decodes an audio InputItem and processes it to match the target specification.
  /// @param input The InputItem containing audio data.
  /// @param target_spec The required output's spec.
  /// @param decoded_audio The output decoded audio.
  /// @return true if decoding was successful, false otherwise.
  virtual bool decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec,
                              OdaiDecodedAudio& decoded_audio) = 0;
};