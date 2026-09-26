#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "SeparationWorker.h"
#include "ModelLocator.h"

class StemSplitterProcessor : public juce::AudioProcessor
{
public:
    StemSplitterProcessor();
    ~StemSplitterProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using juce::AudioProcessor::processBlock;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Stem Splitter"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    // Called from the editor (message thread)
    void setInputFile (const juce::File& file);
    juce::File getInputFile() const { return inputFile; }

    bool startSeparation (juce::String& error);
    void cancelSeparation()                          { worker.cancel(); }
    stemsplitter::SeparationWorker& getWorker()      { return worker; }

    juce::File getModelFile() const                  { return modelFile; }
    void setModelFile (const juce::File& file);

    bool getUseGpu() const                           { return useGpu; }
    void setUseGpu (bool shouldUse);
    int getQualityShifts() const                     { return shifts; }
    void setQualityShifts (int newShifts)            { shifts = juce::jlimit (1, 4, newShifts); }

    // Stems currently loaded for preview (may be null). Thread-safe copy of the pointer.
    std::shared_ptr<const stemsplitter::StemSet> getStems() const;
    int getStemsVersion() const noexcept             { return stemsVersion.load(); }

    // Preview transport
    void setPlaying (bool shouldPlay);
    bool isPlaying() const noexcept                  { return playing.load(); }
    void setPlayPosition (double proportion);
    double getPlayPosition() const;                  // 0..1
    void setSyncToHost (bool shouldSync)             { syncToHost = shouldSync; }
    bool getSyncToHost() const noexcept              { return syncToHost.load(); }

    void setStemMuted (int index, bool muted);
    void setStemSoloed (int index, bool soloed);
    bool isStemMuted (int index) const noexcept      { return (muteMask.load() >> index) & 1u; }
    bool isStemSoloed (int index) const noexcept     { return (soloMask.load() >> index) & 1u; }

    bool getPassThroughInput() const noexcept        { return passThrough.load(); }
    void setPassThroughInput (bool shouldPass)       { passThrough = shouldPass; }

private:
    void installStems (std::shared_ptr<stemsplitter::StemSet> newStems);
    void reloadPreviewIfNeeded();

    juce::SharedResourcePointer<stemsplitter::SharedSettings> settings;
    stemsplitter::SeparationWorker worker;

    juce::File inputFile, modelFile;
    bool useGpu = false;
    int shifts = 1;

    mutable juce::SpinLock stemsLock;
    std::shared_ptr<const stemsplitter::StemSet> stems;
    std::atomic<int> stemsVersion { 0 };

    std::atomic<double> hostSampleRate { 44100.0 };
    std::atomic<bool> playing { false }, syncToHost { false }, passThrough { true };
    std::atomic<juce::int64> playPosition { 0 };
    std::atomic<juce::uint32> muteMask { 0 }, soloMask { 0 };
    std::atomic<bool> masksPending { false };
    juce::uint32 pendingMute = 0, pendingSolo = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemSplitterProcessor)
};
