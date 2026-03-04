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

  /// Decodes an audio file and resamples it to match the target specification.
  /// @param file_path The absolute path to the input audio file.
  /// @param target_spec The required target spec.
  /// @param decoded_audio The output decoded audio.
  /// @return true if decoding was successful, false otherwise.
  bool decode_to_spec(const std::string& file_path, const OdaiAudioTargetSpec& target_spec,
                      OdaiDecodedAudio& decoded_audio) override;
};

#endif // ODAI_ENABLE_MINIAUDIO