// stemsplitter-cli: the plugin's separation engine as a command-line tool.
// Handy for checking that your exported model works before loading the plugin in FL.
//
//   stemsplitter-cli --model htdemucs.onnx [--out folder] [--shifts N] [--gpu] song.wav
//
// Developer self-tests (used by tests/run_tests.py):
//   stemsplitter-cli --spec-test in.f32 out_spec.f32 out_wave.f32 [segment]

#include "../engine/DemucsEngine.h"
#include "../engine/AudioUtils.h"
#include "../engine/OrtLoader.h"

#include <iostream>

using namespace stemsplitter;

static int specTest (const juce::StringArray& args)
{
    // Reads [2][segment] float32, writes the Demucs spectrogram and its inverse.
    const int segment = args.size() > 5 ? args[5].getIntValue() : 343980;
    juce::MemoryBlock in;
    juce::File (args[2]).loadFileAsData (in);

    if ((int) in.getSize() != 2 * segment * (int) sizeof (float))
    {
        std::cerr << "spec-test: input must be 2 x " << segment << " float32\n";
        return 1;
    }

    DemucsSpectrogram spec (segment);
    std::vector<float> s ((size_t) 4 * DemucsSpectrogram::numBins * (size_t) spec.getNumFrames());
    std::vector<float> w ((size_t) 2 * (size_t) segment);

    spec.forward (static_cast<const float*> (in.getData()), s.data());
    spec.inverse (s.data(), w.data());

    juce::File (args[3]).replaceWithData (s.data(), s.size() * sizeof (float));
    juce::File (args[4]).replaceWithData (w.data(), w.size() * sizeof (float));
    return 0;
}

int main (int argc, char* argv[])
{
    juce::StringArray args;
    for (int i = 0; i < argc; ++i)
        args.add (juce::String::fromUTF8 (argv[i]));

    if (args.size() >= 5 && args[1] == "--spec-test")
        return specTest (args);

    juce::File model, outDir, input;
    int shifts = 1, shiftOffset = -1;
    bool gpu = false;

    for (int i = 1; i < args.size(); ++i)
    {
        const auto& a = args[i];
        if (a == "--model" && i + 1 < args.size())       model = juce::File::getCurrentWorkingDirectory().getChildFile (args[++i]);
        else if (a == "--out" && i + 1 < args.size())    outDir = juce::File::getCurrentWorkingDirectory().getChildFile (args[++i]);
        else if (a == "--shifts" && i + 1 < args.size()) shifts = args[++i].getIntValue();
        else if (a == "--shift-offset" && i + 1 < args.size()) shiftOffset = args[++i].getIntValue();
        else if (a == "--gpu")                            gpu = true;
        else                                              input = juce::File::getCurrentWorkingDirectory().getChildFile (a);
    }

    if (model == juce::File() || input == juce::File())
    {
        std::cout << "usage: stemsplitter-cli --model htdemucs.onnx [--out folder] [--shifts N] [--gpu] song.wav\n";
        return 1;
    }

    if (outDir == juce::File())
        outDir = input.getParentDirectory().getChildFile (input.getFileNameWithoutExtension() + " - Stems");

    juce::String error;
    const auto t0 = juce::Time::getMillisecondCounterHiRes();

    auto demucs = DemucsModel::load (model, gpu ? ComputeDevice::gpu : ComputeDevice::cpu, 0, error);
    if (demucs == nullptr)
    {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    std::cout << "ONNX Runtime: " << OrtLoader::getLoadedLibraryPath().getFullPathName()
              << "\nModel: " << model.getFileName() << " (" << demucs->getNumSources() << " stems: "
              << demucs->getSourceNames().joinIntoString (", ") << ")\nDevice: "
              << demucs->getDeviceDescription() << "\n";

    juce::AudioFormatManager formats;
    formats.registerBasicFormats();

    juce::AudioBuffer<float> audio, audio44;
    double sampleRate = 0;

    if (! readAudioFileAsStereo (formats, input, audio, sampleRate, error))
    {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    resample (audio, sampleRate, kModelSampleRate, audio44);

    SeparationOptions options;
    options.shifts = shifts;
    options.forcedShiftOffset = shiftOffset;

    std::vector<juce::AudioBuffer<float>> stems;
    int lastPercent = -1;

    const bool ok = separate (*demucs, audio44, stems, options, [&] (float p)
    {
        const int percent = (int) (p * 100.0f);
        if (percent != lastPercent)
        {
            lastPercent = percent;
            std::cout << "\rSeparating: " << percent << "%" << std::flush;
        }
        return true;
    }, error);

    std::cout << "\n";

    if (! ok)
    {
        std::cerr << "error: " << error << "\n";
        return 1;
    }

    outDir.createDirectory();

    for (size_t s = 0; s < stems.size(); ++s)
    {
        juce::AudioBuffer<float> out;
        resample (stems[s], kModelSampleRate, sampleRate, out);

        // Keep exactly the original length
        out.setSize (2, audio.getNumSamples(), true, true, false);

        auto file = outDir.getChildFile (input.getFileNameWithoutExtension() + " - "
                                         + demucs->getSourceNames()[(int) s] + ".wav");
        if (! writeWav (file, out, sampleRate, error))
        {
            std::cerr << "error: " << error << "\n";
            return 1;
        }
        std::cout << "wrote " << file.getFullPathName() << "\n";
    }

    std::cout << "done in " << juce::String ((juce::Time::getMillisecondCounterHiRes() - t0) / 1000.0, 1) << " s\n";
    return 0;
}
