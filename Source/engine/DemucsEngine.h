#pragma once

// Offline music source separation with Meta's HTDemucs (v4) running in ONNX Runtime.
//
// The STFT/iSTFT can't be exported to ONNX, so the network is exported without them
// (see tools/export_model.py) and this file reproduces, in C++, everything Demucs does
// around the network: normalisation, the random time shift, 7.8 s segments with 25 %
// overlap and triangular cross-fades, reflect padding, the STFT and the inverse STFT.
//
// The approach follows sevagh/demucs.onnx (MIT) and demucs/apply.py (MIT, Meta).

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>

#include <functional>
#include <memory>
#include <vector>

namespace stemsplitter
{
constexpr int kModelSampleRate = 44100;

// Return false from the callback to cancel. Progress is 0..1.
using ProgressCallback = std::function<bool (float progress)>;

enum class ComputeDevice
{
    cpu,
    gpu   // DirectML on Windows, CoreML on macOS; falls back to CPU if unavailable
};

struct SeparationOptions
{
    int shifts = 1;               // Demucs "shifts": >1 averages several time-shifted passes (slower, slightly better)
    float overlap = 0.25f;        // Overlap between segments, as in Demucs
    unsigned int seed = 12345;    // Fixed seed so the same file always gives the same stems
    int forcedShiftOffset = -1;   // Testing only: use this shift instead of a random one
};

//==============================================================================
// One loaded HTDemucs network (an ONNX Runtime session).
class DemucsModel
{
public:
    ~DemucsModel();

    static std::unique_ptr<DemucsModel> load (const juce::File& onnxFile,
                                              ComputeDevice device,
                                              int numThreads,
                                              juce::String& error);

    int getNumSources() const noexcept                   { return numSources; }
    const juce::StringArray& getSourceNames() const      { return sourceNames; }
    int getSegmentLength() const noexcept                { return segmentLength; }
    int getNumSpecFrames() const noexcept                { return numFrames; }
    juce::String getDeviceDescription() const            { return deviceDescription; }
    juce::File getFile() const                           { return file; }

    // Runs the network on one segment.
    //   mix     : [2][segmentLength]            time-domain input
    //   spec    : [4][2048][numFrames]          complex-as-channels spectrogram
    //   specOut : [S][4][2048][numFrames]       (resized by this call)
    //   waveOut : [S][2][segmentLength]         (resized by this call)
    void run (std::vector<float>& mix, std::vector<float>& spec,
              std::vector<float>& specOut, std::vector<float>& waveOut);

private:
    DemucsModel();
    struct Impl;
    std::unique_ptr<Impl> impl;

    juce::File file;
    int numSources = 0;
    int segmentLength = 0;
    int numFrames = 0;
    juce::StringArray sourceNames;
    juce::String deviceDescription;
};

//==============================================================================
// Demucs' spectrogram conventions (HTDemucs._spec / _ispec) for one segment.
class DemucsSpectrogram
{
public:
    static constexpr int nfft = 4096;
    static constexpr int hop = nfft / 4;
    static constexpr int numBins = nfft / 2;   // Demucs drops the Nyquist bin

    explicit DemucsSpectrogram (int segmentLength);

    int getNumFrames() const noexcept { return le; }

    // chunk: [2][segmentLength] -> spec: [4][numBins][le] (re/im interleaved per channel)
    void forward (const float* chunk, float* spec);

    // spec: [4][numBins][le] -> wave: [2][segmentLength]   (adds into `wave` if accumulate)
    void inverse (const float* spec, float* wave);

private:
    int segment, le, padDemucs, padRight, paddedLength;
    juce::dsp::FFT fft { 12 };   // 2^12 = 4096
    std::vector<float> window, fftBuffer, ola, envelope;
    std::vector<int> sourceIndex;  // maps positions in the fully padded signal to chunk samples
};

//==============================================================================
// Separates a stereo 44.1 kHz buffer into model.getNumSources() stereo stems.
// Returns false on error or cancellation (error is empty when cancelled).
bool separate (DemucsModel& model,
               const juce::AudioBuffer<float>& input,
               std::vector<juce::AudioBuffer<float>>& stems,
               const SeparationOptions& options,
               const ProgressCallback& progress,
               juce::String& error);

} // namespace stemsplitter
