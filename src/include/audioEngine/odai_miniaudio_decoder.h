#pragma once

#ifdef ODAI_ENABLE_MINIAUDIO
#include "audioEngine/odai_audio_decoder.h"

/// Concrete implementation of IOdaiAudioDecoder using the miniaudio library.
class OdaiMiniAudioDecoder : public IOdaiAudioDecoder
{
public:
  OdaiMiniAudioDecoder() = default;
  ~OdaiMiniAudioDecoder() override = default;

  OdaiMiniAudioDecoder(const OdaiMiniAudioDecoder&) = delete;
  OdaiMiniAudioDecoder& operator=(const OdaiMiniAudioDecoder&) = delete;
  OdaiMiniAudioDecoder(OdaiMiniAudioDecoder&&) = delete;
  OdaiMiniAudioDecoder& operator=(OdaiMiniAudioDecoder&&) = delete;

  /// Checks if the given audio format is supported by this decoder.
  /// Formats supported by this decoder: "wav", "mp3", "flac"
  /// @param format The audio format extension without a leading dot (e.g., "wav", "mp3", "flac", "ogg"). Matching is
  /// case-insensitive.
  /// @return true if the format is supported, false otherwise
  bool is_supported(const std::string& format) override;

  /// Decodes an audio InputItem and processes it to match the target specification.
  /// @param input The InputItem containing audio data.
  /// @param target_spec The required output's spec.
  /// @param decoded_audio The output decoded audio.
  /// @return empty expected if decoding was successful, or an unexpected OdaiResultEnum indicating the error.
  OdaiResult<void> do_decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec,
                                     OdaiDecodedAudio& decoded_audio) override;
};

#endif // ODAI_ENABLE_MINIAUDIO
