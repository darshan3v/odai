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

  /// Decodes an audio InputItem and processes it to match the target specification.
  /// @param input The InputItem containing audio data.
  /// @param target_spec The required output's spec.
  /// @param decoded_audio The output decoded audio.
  /// @return true if decoding was successful, false otherwise.
  virtual bool decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec,
                              OdaiDecodedAudio& decoded_audio) = 0;
};