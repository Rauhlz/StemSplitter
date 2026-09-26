// Headless test of the plugin's processor: separation job, preview playback,
// mute/solo, save/restore and cancellation. Uses a dummy model (tests/make_dummy_model.py)
// so it runs without the real weights. Optionally also loads the built .vst3 through
// JUCE's VST3 host to check the bundle loads, processes audio and round-trips state.
//
//   plugin-harness dummy.onnx song.wav long_song.wav [path/to/Stem Splitter.vst3]

#include "PluginProcessor.h"
#include "engine/AudioUtils.h"

#include <iostream>

using namespace stemsplitter;

static int failures = 0;

static void check (bool ok, const juce::String& name, const juce::String& detail = {})
{
    std::cout << (ok ? "PASS " : "FAIL ") << name << "  " << detail << std::endl;
    if (! ok) ++failures;
}

static bool waitForWorker (StemSplitterProcessor& p, double timeoutSeconds)
{
    const auto end = juce::Time::getMillisecondCounterHiRes() + timeoutSeconds * 1000.0;
    // give the thread a moment to start
    juce::Thread::sleep (20);
    while (p.getWorker().isBusy())
    {
        if (juce::Time::getMillisecondCounterHiRes() > end) return false;
        juce::Thread::sleep (20);
    }
    return true;
}

static float maxDiff (const juce::AudioBuffer<float>& a, const juce::AudioBuffer<float>& b, int n)
{
    float m = 0.0f;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < n; ++i)
            m = juce::jmax (m, std::abs (a.getSample (c, i) - b.getSample (c, i)));
    return m;
}

static void snapshot (juce::AudioProcessorEditor& editor, const juce::File& file)
{
    auto image = editor.createComponentSnapshot (editor.getLocalBounds(), true, 2.0f);
    file.deleteFile();
    juce::FileOutputStream out (file);
    juce::PNGImageFormat().writeImageToStream (image, out);
    std::cout << "     snapshot: " << file.getFullPathName() << std::endl;
}

int main (int argc, char* argv[])
{
    juce::ScopedJuceInitialiser_GUI juceInit;

    if (argc < 4)
    {
        std::cerr << "usage: plugin-harness dummy.onnx song.wav long_song.wav [plugin.vst3]\n";
        return 2;
    }

    const juce::File model (juce::File::getCurrentWorkingDirectory().getChildFile (argv[1]));
    const juce::File song (juce::File::getCurrentWorkingDirectory().getChildFile (argv[2]));
    const juce::File longSong (juce::File::getCurrentWorkingDirectory().getChildFile (argv[3]));
    const double hostRate = 48000.0;
    const int block = 512;

    juce::MemoryBlock savedState;

    {
        StemSplitterProcessor p;
        p.setPlayConfigDetails (2, 2, hostRate, block);
        p.prepareToPlay (hostRate, block);
        p.setModelFile (model);
        p.setInputFile (song);

        juce::String error;
        check (p.startSeparation (error), "separation starts", error);
        check (waitForWorker (p, 120), "separation finishes");
        check (p.getWorker().getStage() == SeparationWorker::Stage::finished, "stage is finished",
               p.getWorker().getMessage());

        auto stems = p.getStems();
        check (stems != nullptr, "stems installed");
        if (stems == nullptr) return 1;

        check (stems->names.joinIntoString (",") == "Drums,Bass,Other,Vocals,Instrumental", "stem names",
               stems->names.joinIntoString (","));

        bool allExist = true;
        for (auto& f : stems->files) allExist = allExist && f.existsAsFile();
        check (allExist && stems->files.size() == 5, "stem files written", stems->folder.getFullPathName());
        check (std::abs (stems->previewRate - hostRate) < 0.5, "preview resampled to host rate",
               juce::String (stems->previewRate));

        // Written files are at the song's own rate and length
        juce::AudioFormatManager fm; fm.registerBasicFormats();
        std::unique_ptr<juce::AudioFormatReader> src (fm.createReaderFor (song));
        std::unique_ptr<juce::AudioFormatReader> st (fm.createReaderFor (stems->files[0]));
        check (src && st && src->lengthInSamples == st->lengthInSamples && src->sampleRate == st->sampleRate,
               "stem file keeps source rate and length");

        // Instrumental = sum of non-vocal stems
        {
            juce::AudioBuffer<float> expected (2, stems->previewLength);
            expected.clear();
            for (int s = 0; s < 4; ++s)
                if (s != 3)
                    for (int c = 0; c < 2; ++c)
                        expected.addFrom (c, 0, stems->preview[(size_t) s], c, 0, stems->previewLength);
            const float d = maxDiff (expected, stems->preview[4], stems->previewLength);
            check (d < 1e-3f, "instrumental is the sum of drums+bass+other", juce::String (d));
        }

        // Preview playback: everything except Instrumental (muted by default), no input
        p.setPassThroughInput (false);
        p.setPlayPosition (0.0);
        p.setPlaying (true);

        juce::AudioBuffer<float> io (2, block);
        juce::MidiBuffer midi;
        juce::AudioBuffer<float> rendered (2, block * 20);

        for (int b = 0; b < 20; ++b)
        {
            for (int c = 0; c < 2; ++c) io.clear();
            p.processBlock (io, midi);
            for (int c = 0; c < 2; ++c) rendered.copyFrom (c, b * block, io, c, 0, block);
        }

        juce::AudioBuffer<float> expected (2, block * 20);
        expected.clear();
        for (int s = 0; s < 4; ++s)
            for (int c = 0; c < 2; ++c)
                expected.addFrom (c, 0, stems->preview[(size_t) s], c, 0, block * 20);

        check (maxDiff (rendered, expected, block * 20) < 1e-5f, "preview plays the unmuted stems");
        check (std::abs (p.getPlayPosition() - (double) (block * 20) / stems->previewLength) < 1e-6,
               "play position advances");

        // Solo bass
        p.setStemSoloed (1, true);
        p.setPlayPosition (0.0);
        for (int c = 0; c < 2; ++c) io.clear();
        p.processBlock (io, midi);
        check (maxDiff (io, stems->preview[1], block) < 1e-6f, "solo plays only that stem");

        // Pass-through adds the input on top
        p.setPassThroughInput (true);
        p.setPlaying (false);
        for (int c = 0; c < 2; ++c) io.clear();
        io.setSample (0, 10, 0.5f);
        p.processBlock (io, midi);
        check (std::abs (io.getSample (0, 10) - 0.5f) < 1e-7f, "input passes through when stopped");

        // Mute a stem and keep the solo, then save
        p.setStemMuted (2, true);
        p.getStateInformation (savedState);

        // Editor screenshots (software-rendered, no display needed)
        if (auto snapDir = juce::SystemStats::getEnvironmentVariable ("SNAPSHOT_DIR", {}); snapDir.isNotEmpty())
        {
            p.setPlayPosition (0.3);
            std::unique_ptr<juce::AudioProcessorEditor> editor (p.createEditor());
            snapshot (*editor, juce::File (snapDir).getChildFile ("editor_stems.png"));
            editor.reset();

            StemSplitterProcessor empty;
            std::unique_ptr<juce::AudioProcessorEditor> editor2 (empty.createEditor());
            snapshot (*editor2, juce::File (snapDir).getChildFile ("editor_empty.png"));
        }
    }

    // Restore into a fresh instance
    {
        StemSplitterProcessor p;
        p.setPlayConfigDetails (2, 2, 44100.0, block);
        p.prepareToPlay (44100.0, block);
        p.setStateInformation (savedState.getData(), (int) savedState.getSize());
        check (waitForWorker (p, 60), "restore reload finishes");

        auto stems = p.getStems();
        check (stems != nullptr && stems->names.size() == 5, "stems restored from project");
        check (p.isStemSoloed (1) && p.isStemMuted (2) && p.isStemMuted (4) && ! p.isStemMuted (0),
               "mute/solo restored");
        check (stems != nullptr && std::abs (stems->previewRate - 44100.0) < 0.5, "restored preview at new host rate");

        // Host sample-rate change reloads the preview
        p.prepareToPlay (96000.0, block);
        check (waitForWorker (p, 60), "rate-change reload finishes");
        stems = p.getStems();
        check (stems != nullptr && std::abs (stems->previewRate - 96000.0) < 0.5, "preview follows host rate change");
    }

    // Cancellation
    {
        StemSplitterProcessor p;
        p.prepareToPlay (44100.0, block);
        p.setModelFile (model);
        p.setInputFile (longSong);
        juce::String error;
        p.startSeparation (error);
        juce::Thread::sleep (600);
        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        p.cancelSeparation();
        check (waitForWorker (p, 30), "cancel stops the job");
        const auto ms = juce::Time::getMillisecondCounterHiRes() - t0;
        check (p.getWorker().getStage() == SeparationWorker::Stage::cancelled, "stage is cancelled",
               p.getWorker().getMessage() + " after " + juce::String (ms, 0) + " ms");
    }

    // Error path: missing model
    {
        StemSplitterProcessor p;
        p.setModelFile (juce::File::getCurrentWorkingDirectory().getChildFile ("nope.onnx"));
        p.setInputFile (song);
        juce::String error;
        const bool started = p.startSeparation (error);
        if (started) waitForWorker (p, 30);
        check (! started || p.getWorker().getStage() == SeparationWorker::Stage::failed,
               "missing model reports an error", error + p.getWorker().getMessage());
    }

    // The real bundle, through JUCE's VST3 host
    if (argc > 4)
    {
        juce::VST3PluginFormat format;
        juce::OwnedArray<juce::PluginDescription> descs;
        format.findAllTypesForFile (descs, argv[4]);
        check (descs.size() == 1, "VST3 bundle scans", juce::String (descs.size()) + " plugin(s)");

        if (! descs.isEmpty())
        {
            juce::String error;
            auto instance = format.createInstanceFromDescription (*descs[0], 48000.0, block, error);
            check (instance != nullptr, "VST3 instantiates", error);

            if (instance != nullptr)
            {
                std::cout << "     name=" << instance->getName() << " category=" << descs[0]->category
                          << " ins=" << instance->getTotalNumInputChannels()
                          << " outs=" << instance->getTotalNumOutputChannels() << std::endl;

                instance->prepareToPlay (48000.0, block);
                juce::AudioBuffer<float> io (2, block);
                juce::MidiBuffer midi;
                for (int i = 0; i < block; ++i) { io.setSample (0, i, 0.25f); io.setSample (1, i, -0.25f); }
                instance->processBlock (io, midi);
                check (std::abs (io.getSample (0, 100) - 0.25f) < 1e-6f, "VST3 passes audio through");

                // Hand the saved project state to the bundle the way a host would
                juce::XmlElement wrapped ("VST3PluginState");
                wrapped.createNewChildElement ("IComponent")->addTextElement (savedState.toBase64Encoding());
                juce::MemoryBlock wrappedState;
                juce::AudioProcessor::copyXmlToBinary (wrapped, wrappedState);
                instance->setStateInformation (wrappedState.getData(), (int) wrappedState.getSize());
                // The plugin reloads the stems in the background: poll its state
                int stemCount = -1;
                for (int attempt = 0; attempt < 100 && stemCount != 5; ++attempt)
                {
                    juce::Thread::sleep (200);
                    juce::MemoryBlock again;
                    instance->getStateInformation (again);

                    // JUCE's VST3 host wraps the plugin's own state in base64 inside its XML
                    auto outer = juce::AudioProcessor::getXmlFromBinary (again.getData(), (int) again.getSize());
                    if (outer == nullptr) continue;

                    if (auto* comp = outer->getChildByName ("IComponent"))
                    {
                        juce::MemoryBlock inner;
                        inner.fromBase64Encoding (comp->getAllSubText());
                        // binary header, then the XML text: search the raw bytes
                        const std::string bytes (static_cast<const char*> (inner.getData()), inner.getSize());
                        stemCount = 0;
                        for (auto i = bytes.find ("<Stem "); i != std::string::npos; i = bytes.find ("<Stem ", i + 1))
                            ++stemCount;
                    }
                }
                check (stemCount == 5, "VST3 state round-trips with stems", juce::String (stemCount) + " stems in state");
                instance->releaseResources();
            }
        }
    }

    std::cout << "\n" << (failures == 0 ? "ALL HARNESS TESTS PASSED" : juce::String (failures) + " FAILED") << std::endl;
    return failures == 0 ? 0 : 1;
}
