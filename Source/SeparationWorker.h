#pragma once

#include <juce_audio_formats/juce_audio_formats.h>
#include "engine/DemucsEngine.h"

#include <atomic>
#include <memory>

namespace stemsplitter
{
// Stems on disk plus a copy in memory (at the host's sample rate) for previewing.
struct StemSet
{
    juce::File sourceFile, folder;
    juce::StringArray names;
    juce::Array<juce::File> files;
    std::vector<juce::AudioBuffer<float>> preview;
    std::vector<std::vector<float>> peaks;   // min/max pairs per stem, for drawing
    double previewRate = 0;
    int previewLength = 0;
};

// Runs separations (and reloads of existing stems) on a background thread so FL
// keeps playing while the model works.
class SeparationWorker : private juce::Thread
{
public:
    enum class Stage { idle, loadingModel, decoding, separating, writing, loadingPreview, finished, failed, cancelled };

    struct Job
    {
        enum class Kind { separate, loadExisting };
        Kind kind = Kind::separate;

        juce::File input, model;
        bool useGpu = false;
        int shifts = 1;
        bool writeInstrumental = true;
        double previewRate = 44100.0;

        // loadExisting only
        juce::File existingFolder;
        juce::StringArray existingNames;
        juce::Array<juce::File> existingFiles;
    };

    SeparationWorker();
    ~SeparationWorker() override;

    bool start (const Job& job);   // false if a job is already running
    void cancel();
    bool isBusy() const;

    Stage getStage() const noexcept      { return stage.load(); }
    float getProgress() const noexcept   { return progress.load(); }
    juce::String getMessage() const;
    juce::String getDeviceDescription() const;

    // Called on the worker thread when stems are ready (after a separation or reload).
    std::function<void (std::shared_ptr<StemSet>)> onStemsReady;

    static juce::File chooseOutputFolder (const juce::File& input);

private:
    void run() override;
    bool runSeparation (const Job& job);
    bool runLoadExisting (const Job& job);
    void setStage (Stage s, const juce::String& message, float progressValue);
    bool buildPreview (StemSet& set, std::vector<juce::AudioBuffer<float>>& buffers,
                       double buffersRate, double previewRate);

    juce::AudioFormatManager formats;
    Job currentJob;
    std::atomic<Stage> stage { Stage::idle };
    std::atomic<float> progress { 0.0f };
    juce::CriticalSection messageLock;
    juce::String message, deviceDescription;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SeparationWorker)
};
} // namespace stemsplitter
