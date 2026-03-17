#pragma once

#include <string>

class InputItem;
struct OdaiAudioTargetSpec;
struct OdaiDecodedAudio;

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
  virtual bool is_supported(const std::string& format) = 0;

  /// Decodes an audio InputItem and processes it to match the target specification.
  /// @param input The InputItem containing audio data.
  /// @param target_spec The required output's spec.
  /// @param decoded_audio The output decoded audio.
  /// @return true if decoding was successful, false otherwise.
  bool decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec, OdaiDecodedAudio& decoded_audio);

protected:
  /// Core implementation of decode_to_spec to be provided by the derived class.
  /// Input validation (sanity constraints, media type, empty data) is already handled by the base class.
  /// @param input The validated InputItem containing audio data.
  /// @param target_spec The required output's spec.
  /// @param decoded_audio The output decoded audio.
  /// @return true if decoding was successful, false otherwise.
  virtual bool do_decode_to_spec(const InputItem& input, const OdaiAudioTargetSpec& target_spec,
                                 OdaiDecodedAudio& decoded_audio) = 0;
};