#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

#include <functional>

namespace stemsplitter
{
// High-quality offline sample-rate conversion (Kaiser-windowed sinc, anti-aliased when
// downsampling). Output length is round(inputLength * outRate / inRate).
void resample (const juce::AudioBuffer<float>& input, double inRate, double outRate,
               juce::AudioBuffer<float>& output,
               const std::function<bool()>& shouldContinue = {});

// Reads any format the manager knows into a stereo buffer (mono is duplicated,
// extra channels are dropped). Returns false and fills `error` on failure.
bool readAudioFileAsStereo (juce::AudioFormatManager& formats, const juce::File& file,
                            juce::AudioBuffer<float>& output, double& sampleRate,
                            juce::String& error);

// Writes a 32-bit float WAV (float so loud stems can never clip).
bool writeWav (const juce::File& file, const juce::AudioBuffer<float>& buffer,
               double sampleRate, juce::String& error);

// Min/max peak pairs for drawing a waveform overview. Returns 2 * numBuckets values.
std::vector<float> computePeaks (const juce::AudioBuffer<float>& buffer, int numBuckets);
} // namespace stemsplitter
