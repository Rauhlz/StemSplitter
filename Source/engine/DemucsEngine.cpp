#include "DemucsEngine.h"
#include "OrtLoader.h"

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <random>
#include <stdexcept>

namespace stemsplitter
{
//==============================================================================
namespace
{
Ort::Env& getOrtEnv()
{
    // One environment per process, created after OrtLoader has initialised the API.
    static Ort::Env env (ORT_LOGGING_LEVEL_WARNING, "StemSplitter");
    return env;
}

juce::StringArray defaultSourceNames (int numSources)
{
    if (numSources == 4) return { "Drums", "Bass", "Other", "Vocals" };
    if (numSources == 6) return { "Drums", "Bass", "Other", "Vocals", "Guitar", "Piano" };
    if (numSources == 2) return { "Vocals", "Instrumental" };

    juce::StringArray names;
    for (int i = 0; i < numSources; ++i)
        names.add ("Stem " + juce::String (i + 1));
    return names;
}

inline int reflectIndex (int i, int length) noexcept
{
    // Same as torch.nn.functional.pad(mode="reflect"): the edge sample is not repeated.
    if (i < 0)        i = -i;
    if (i >= length)  i = 2 * (length - 1) - i;
    return i;
}
} // namespace

//==============================================================================
struct DemucsModel::Impl
{
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> inputNames, outputNames;
    int mixInput = 0, specInput = 1;      // input indices, identified by rank
    int specOutput = 0, waveOutput = 1;   // output indices, identified by rank
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
};

DemucsModel::DemucsModel() : impl (std::make_unique<Impl>()) {}
DemucsModel::~DemucsModel() = default;

std::unique_ptr<DemucsModel> DemucsModel::load (const juce::File& onnxFile,
                                                ComputeDevice device,
                                                int numThreads,
                                                juce::String& error)
{
    if (! OrtLoader::ensureLoaded (error))
        return nullptr;

    if (! onnxFile.existsAsFile())
    {
        error = "Model file not found: " + onnxFile.getFullPathName();
        return nullptr;
    }

    std::unique_ptr<DemucsModel> model (new DemucsModel());
    model->file = onnxFile;

    try
    {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (numThreads <= 0)
            numThreads = juce::jmax (1, juce::SystemStats::getNumPhysicalCpus());

        options.SetIntraOpNumThreads (numThreads);
        model->deviceDescription = "CPU (" + juce::String (numThreads) + (numThreads == 1 ? " thread)" : " threads)");

        if (device == ComputeDevice::gpu)
        {
           #if JUCE_WINDOWS
            // Needs the DirectML build of onnxruntime.dll (STEMSPLITTER_DIRECTML=ON)
            try
            {
                options.DisableMemPattern();
                options.SetExecutionMode (ExecutionMode::ORT_SEQUENTIAL);
                options.AppendExecutionProvider ("DML", {});
                model->deviceDescription = "GPU (DirectML)";
            }
            catch (const Ort::Exception&)
            {
                model->deviceDescription << " - this build has no GPU support";
            }
           #elif JUCE_MAC
            // CoreML ships in the macOS onnxruntime build; its factory is a plain C export.
            using CoreMLFactory = OrtStatus* (ORT_API_CALL*) (OrtSessionOptions*, uint32_t);
            constexpr uint32_t createMLProgram = 0x010;   // COREML_FLAG_CREATE_MLPROGRAM

            if (auto* append = reinterpret_cast<CoreMLFactory> (OrtLoader::getSymbol ("OrtSessionOptionsAppendExecutionProvider_CoreML")))
            {
                if (auto* status = append (options, createMLProgram))
                {
                    model->deviceDescription << " - CoreML unavailable: " << Ort::GetApi().GetErrorMessage (status);
                    Ort::GetApi().ReleaseStatus (status);
                }
                else
                {
                    model->deviceDescription = "GPU/Neural Engine (CoreML)";
                }
            }
            else
            {
                model->deviceDescription << " - CoreML not in this onnxruntime build";
            }
           #else
            model->deviceDescription << " - no GPU support on this platform";
           #endif
        }

       #if JUCE_WINDOWS
        model->impl->session = std::make_unique<Ort::Session> (getOrtEnv(),
                                                               onnxFile.getFullPathName().toWideCharPointer(),
                                                               options);
       #else
        model->impl->session = std::make_unique<Ort::Session> (getOrtEnv(),
                                                               onnxFile.getFullPathName().toRawUTF8(),
                                                               options);
       #endif

        auto& session = *model->impl->session;
        Ort::AllocatorWithDefaultOptions allocator;

        if (session.GetInputCount() != 2 || session.GetOutputCount() != 2)
        {
            error = "This ONNX file doesn't look like an exported HTDemucs model "
                    "(expected 2 inputs and 2 outputs).";
            return nullptr;
        }

        std::vector<int64_t> mixShape, specOutShape;

        for (size_t i = 0; i < 2; ++i)
        {
            model->impl->inputNames.push_back (session.GetInputNameAllocated (i, allocator).get());
            auto shape = session.GetInputTypeInfo (i).GetTensorTypeAndShapeInfo().GetShape();

            if (shape.size() == 3)      { model->impl->mixInput = (int) i;  mixShape = shape; }
            else if (shape.size() == 4) { model->impl->specInput = (int) i; }
        }

        for (size_t i = 0; i < 2; ++i)
        {
            model->impl->outputNames.push_back (session.GetOutputNameAllocated (i, allocator).get());
            auto shape = session.GetOutputTypeInfo (i).GetTensorTypeAndShapeInfo().GetShape();

            if (shape.size() == 5)      { model->impl->specOutput = (int) i; specOutShape = shape; }
            else if (shape.size() == 4) { model->impl->waveOutput = (int) i; }
        }

        if (mixShape.size() != 3 || specOutShape.size() != 5
            || model->impl->mixInput == model->impl->specInput
            || model->impl->specOutput == model->impl->waveOutput)
        {
            error = "Unexpected tensor shapes - is this an HTDemucs model exported with tools/export_model.py?";
            return nullptr;
        }

        model->segmentLength = mixShape[2] > 0 ? (int) mixShape[2] : 343980;   // 7.8 s at 44.1 kHz
        model->numFrames = (model->segmentLength + DemucsSpectrogram::hop - 1) / DemucsSpectrogram::hop;
        model->numSources = specOutShape[1] > 0 ? (int) specOutShape[1] : 4;

        // Stem names can be stored in the model's metadata by the export script.
        auto metadata = session.GetModelMetadata();
        auto names = metadata.LookupCustomMetadataMapAllocated ("sources", allocator);

        if (names != nullptr)
        {
            juce::StringArray parsed;
            parsed.addTokens (juce::String (names.get()), ",", "");
            parsed.trim();

            for (auto& n : parsed)
                n = n.substring (0, 1).toUpperCase() + n.substring (1);

            if (parsed.size() == model->numSources)
                model->sourceNames = parsed;
        }

        if (model->sourceNames.isEmpty())
            model->sourceNames = defaultSourceNames (model->numSources);
    }
    catch (const Ort::Exception& e)
    {
        error = "ONNX Runtime couldn't load the model: " + juce::String (e.what());
        return nullptr;
    }

    return model;
}

void DemucsModel::run (std::vector<float>& mix, std::vector<float>& spec,
                       std::vector<float>& specOut, std::vector<float>& waveOut)
{
    auto& im = *impl;

    const int64_t mixShape[]  = { 1, 2, segmentLength };
    const int64_t specShape[] = { 1, 4, DemucsSpectrogram::numBins, numFrames };

    jassert ((int64_t) mix.size()  == 2 * (int64_t) segmentLength);
    jassert ((int64_t) spec.size() == 4 * (int64_t) DemucsSpectrogram::numBins * numFrames);

    Ort::Value inputs[2] = { Ort::Value (nullptr), Ort::Value (nullptr) };
    inputs[im.mixInput]  = Ort::Value::CreateTensor<float> (im.memoryInfo, mix.data(),  mix.size(),  mixShape, 3);
    inputs[im.specInput] = Ort::Value::CreateTensor<float> (im.memoryInfo, spec.data(), spec.size(), specShape, 4);

    const char* inNames[]  = { im.inputNames[0].c_str(),  im.inputNames[1].c_str() };
    const char* outNames[] = { im.outputNames[0].c_str(), im.outputNames[1].c_str() };

    auto outputs = im.session->Run (Ort::RunOptions { nullptr }, inNames, inputs, 2, outNames, 2);

    auto copyOut = [] (Ort::Value& v, std::vector<float>& dest, size_t expected)
    {
        auto count = v.GetTensorTypeAndShapeInfo().GetElementCount();
        if (count != expected)
            throw std::runtime_error ("Model returned " + std::to_string (count)
                                      + " values, expected " + std::to_string (expected));
        const float* src = v.GetTensorData<float>();
        dest.assign (src, src + count);
    };

    copyOut (outputs[(size_t) im.specOutput], specOut,
             (size_t) numSources * 4 * DemucsSpectrogram::numBins * (size_t) numFrames);
    copyOut (outputs[(size_t) im.waveOutput], waveOut,
             (size_t) numSources * 2 * (size_t) segmentLength);
}

//==============================================================================
DemucsSpectrogram::DemucsSpectrogram (int segmentLength)
    : segment (segmentLength),
      le ((segmentLength + hop - 1) / hop),
      padDemucs (hop / 2 * 3),
      padRight (padDemucs + le * hop - segmentLength),
      paddedLength (le * hop + 2 * padDemucs)
{
    // Periodic Hann window, same as torch.hann_window(4096)
    window.resize (nfft);
    for (int n = 0; n < nfft; ++n)
        window[(size_t) n] = (float) (0.5 - 0.5 * std::cos (2.0 * juce::MathConstants<double>::pi * n / nfft));

    fftBuffer.resize (2 * nfft);

    // torch.stft(center=True) reflect-pads nfft/2 on each side of the signal that
    // Demucs already reflect-padded by (padDemucs, padRight). Precompute where each
    // sample of the doubly padded signal comes from in the original chunk.
    const int totalLength = paddedLength + nfft;
    sourceIndex.resize ((size_t) totalLength);

    for (int j = 0; j < totalLength; ++j)
    {
        const int i1 = reflectIndex (j - nfft / 2, paddedLength);
        sourceIndex[(size_t) j] = reflectIndex (i1 - padDemucs, segment);
    }

    // The iSTFT normalises by the summed squared window of every frame (le + 4 frames).
    ola.resize ((size_t) totalLength);
    envelope.assign ((size_t) totalLength, 0.0f);

    for (int t = 0; t < le + 4; ++t)
        for (int n = 0; n < nfft; ++n)
            envelope[(size_t) (t * hop + n)] += window[(size_t) n] * window[(size_t) n];
}

void DemucsSpectrogram::forward (const float* chunk, float* spec)
{
    const float scale = 1.0f / std::sqrt ((float) nfft);   // torch.stft(normalized=True)
    const size_t plane = (size_t) numBins * (size_t) le;

    for (int c = 0; c < 2; ++c)
    {
        const float* x = chunk + (size_t) c * (size_t) segment;
        float* re = spec + (size_t) (2 * c) * plane;
        float* im = spec + (size_t) (2 * c + 1) * plane;

        for (int f = 0; f < le; ++f)
        {
            // Demucs keeps STFT frames 2 .. 2 + le
            const int start = (f + 2) * hop;

            for (int n = 0; n < nfft; ++n)
                fftBuffer[(size_t) n] = x[sourceIndex[(size_t) (start + n)]] * window[(size_t) n];

            std::fill (fftBuffer.begin() + nfft, fftBuffer.end(), 0.0f);
            fft.performRealOnlyForwardTransform (fftBuffer.data(), true);

            for (int k = 0; k < numBins; ++k)
            {
                re[(size_t) k * (size_t) le + (size_t) f] = fftBuffer[(size_t) (2 * k)]     * scale;
                im[(size_t) k * (size_t) le + (size_t) f] = fftBuffer[(size_t) (2 * k + 1)] * scale;
            }
        }
    }
}

void DemucsSpectrogram::inverse (const float* spec, float* wave)
{
    // JUCE's inverse FFT is already scaled by 1/N (like irfft); torch.istft(normalized=True)
    // additionally multiplies by sqrt(N).
    const float scale = std::sqrt ((float) nfft);
    const size_t plane = (size_t) numBins * (size_t) le;
    const int offset = nfft / 2 + padDemucs;   // undo the centre padding, then Demucs' padding

    for (int c = 0; c < 2; ++c)
    {
        const float* re = spec + (size_t) (2 * c) * plane;
        const float* im = spec + (size_t) (2 * c + 1) * plane;

        std::fill (ola.begin(), ola.end(), 0.0f);

        // Frames 0, 1, le+2 and le+3 are the zero frames Demucs pads with, so skip them.
        for (int f = 0; f < le; ++f)
        {
            std::fill (fftBuffer.begin(), fftBuffer.end(), 0.0f);

            for (int k = 0; k < numBins; ++k)
            {
                fftBuffer[(size_t) (2 * k)]     = re[(size_t) k * (size_t) le + (size_t) f];
                fftBuffer[(size_t) (2 * k + 1)] = im[(size_t) k * (size_t) le + (size_t) f];
            }
            // Nyquist bin (index numBins) stays zero, as in Demucs.

            fft.performRealOnlyInverseTransform (fftBuffer.data());

            const int start = (f + 2) * hop;
            for (int n = 0; n < nfft; ++n)
                ola[(size_t) (start + n)] += fftBuffer[(size_t) n] * scale * window[(size_t) n];
        }

        float* out = wave + (size_t) c * (size_t) segment;
        for (int i = 0; i < segment; ++i)
            out[i] = ola[(size_t) (offset + i)] / envelope[(size_t) (offset + i)];
    }
}

//==============================================================================
bool separate (DemucsModel& model,
               const juce::AudioBuffer<float>& input,
               std::vector<juce::AudioBuffer<float>>& stems,
               const SeparationOptions& options,
               const ProgressCallback& progress,
               juce::String& error)
{
    error.clear();

    if (input.getNumChannels() != 2)
    {
        error = "separate() needs a stereo buffer";
        return false;
    }

    const int length = input.getNumSamples();
    const int numSources = model.getNumSources();
    const int segment = model.getSegmentLength();
    const int numBins = DemucsSpectrogram::numBins;

    if (length == 0)
    {
        error = "The audio file is empty.";
        return false;
    }

    // --- Normalise like demucs/separate.py: ref = mono mix, (wav - mean) / std ---
    double mean = 0.0, safeStd = 1.0;
    {
        const float* l = input.getReadPointer (0);
        const float* r = input.getReadPointer (1);

        for (int i = 0; i < length; ++i)
            mean += 0.5 * ((double) l[i] + (double) r[i]);
        mean /= length;

        double var = 0.0;
        for (int i = 0; i < length; ++i)
        {
            const double d = 0.5 * ((double) l[i] + (double) r[i]) - mean;
            var += d * d;
        }

        const double stdDev = length > 1 ? std::sqrt (var / (length - 1)) : 0.0;
        // Pure silence would divide by zero in Demucs; treat it as unit scale instead.
        safeStd = stdDev > 1.0e-8 ? stdDev : 1.0;
    }

    stems.clear();

    {
        // --- Zero-padded, normalised mix with max_shift on both sides (apply_model shifts) ---
        const int maxShift = kModelSampleRate / 2;
        const int paddedLength = length + 2 * maxShift;
        std::vector<float> padded ((size_t) 2 * (size_t) paddedLength, 0.0f);

        for (int c = 0; c < 2; ++c)
        {
            const float* src = input.getReadPointer (c);
            float* dst = padded.data() + (size_t) c * (size_t) paddedLength + (size_t) maxShift;

            for (int i = 0; i < length; ++i)
                dst[i] = (float) (((double) src[i] - mean) / safeStd);
        }

        // Triangular cross-fade weights (apply_model, transition_power = 1)
        std::vector<float> weight ((size_t) segment);
        {
            const int half = segment / 2;
            for (int i = 0; i < segment; ++i)
                weight[(size_t) i] = (float) (i < half ? i + 1 : segment - i);

            const float maxW = (float) juce::jmax (half, segment - half);
            for (auto& w : weight)
                w /= maxW;
        }

        const int stride = juce::jmax (1, (int) ((1.0f - options.overlap) * (float) segment));
        const int shifts = juce::jmax (1, options.shifts);

        // Total number of network calls, for progress reporting
        std::mt19937 rng (options.seed);
        std::vector<int> shiftOffsets;
        int totalChunks = 0;

        for (int s = 0; s < shifts; ++s)
        {
            std::uniform_int_distribution<int> dist (0, maxShift);
            const int off = options.forcedShiftOffset >= 0 ? juce::jmin (options.forcedShiftOffset, maxShift)
                                                           : dist (rng);
            shiftOffsets.push_back (off);
            const int shiftedLength = length + maxShift - off;
            totalChunks += (shiftedLength + stride - 1) / stride;
        }

        // Final output accumulators
        std::vector<std::vector<float>> result ((size_t) numSources * 2, std::vector<float> ((size_t) length, 0.0f));

        DemucsSpectrogram spectrogram (segment);
        const int numFrames = spectrogram.getNumFrames();
        jassert (numFrames == model.getNumSpecFrames());

        std::vector<float> mixChunk ((size_t) 2 * (size_t) segment);
        std::vector<float> spec ((size_t) 4 * (size_t) numBins * (size_t) numFrames);
        std::vector<float> specOut, waveOut;
        std::vector<float> sourceWave ((size_t) 2 * (size_t) segment);

        int chunksDone = 0;

        if (progress && ! progress (0.0f))
            return false;

        for (int s = 0; s < shifts; ++s)
        {
            const int off = shiftOffsets[(size_t) s];
            const int shiftedLength = length + maxShift - off;

            std::vector<float> shiftOut ((size_t) numSources * 2 * (size_t) shiftedLength, 0.0f);
            std::vector<float> sumWeight ((size_t) shiftedLength, 0.0f);

            for (int chunkStart = 0; chunkStart < shiftedLength; chunkStart += stride)
            {
                const int chunkLength = juce::jmin (segment, shiftedLength - chunkStart);
                const int delta = segment - chunkLength;

                // TensorChunk.padded(): take real context from the padded mix where it exists
                const int readStart = off + chunkStart - delta / 2;

                for (int c = 0; c < 2; ++c)
                {
                    const float* src = padded.data() + (size_t) c * (size_t) paddedLength;
                    float* dst = mixChunk.data() + (size_t) c * (size_t) segment;

                    for (int i = 0; i < segment; ++i)
                    {
                        const int idx = readStart + i;
                        dst[i] = (idx >= 0 && idx < paddedLength) ? src[idx] : 0.0f;
                    }
                }

                spectrogram.forward (mixChunk.data(), spec.data());

                try
                {
                    model.run (mixChunk, spec, specOut, waveOut);
                }
                catch (const std::exception& e)
                {
                    error = "Inference failed: " + juce::String (e.what());
                    return false;
                }

                const size_t specPerSource = (size_t) 4 * (size_t) numBins * (size_t) numFrames;
                const int trim = delta / 2;   // center_trim back to chunkLength

                for (int src = 0; src < numSources; ++src)
                {
                    spectrogram.inverse (specOut.data() + (size_t) src * specPerSource, sourceWave.data());

                    for (int c = 0; c < 2; ++c)
                    {
                        const float* freqBranch = sourceWave.data() + (size_t) c * (size_t) segment;
                        const float* timeBranch = waveOut.data() + ((size_t) src * 2 + (size_t) c) * (size_t) segment;
                        float* dst = shiftOut.data() + ((size_t) src * 2 + (size_t) c) * (size_t) shiftedLength
                                                     + (size_t) chunkStart;

                        for (int i = 0; i < chunkLength; ++i)
                            dst[i] += weight[(size_t) i] * (freqBranch[trim + i] + timeBranch[trim + i]);
                    }
                }

                for (int i = 0; i < chunkLength; ++i)
                    sumWeight[(size_t) (chunkStart + i)] += weight[(size_t) i];

                ++chunksDone;

                if (progress && ! progress ((float) chunksDone / (float) totalChunks))
                    return false;
            }

            // Normalise the cross-fades and drop the shift padding
            const int trimStart = maxShift - off;

            for (int k = 0; k < numSources * 2; ++k)
            {
                const float* src = shiftOut.data() + (size_t) k * (size_t) shiftedLength;
                auto& dst = result[(size_t) k];

                for (int i = 0; i < length; ++i)
                    dst[(size_t) i] += src[trimStart + i] / sumWeight[(size_t) (trimStart + i)];
            }
        }

        // Average the shifts and undo the normalisation
        stems.resize ((size_t) numSources);

        for (int src = 0; src < numSources; ++src)
        {
            stems[(size_t) src].setSize (2, length);

            for (int c = 0; c < 2; ++c)
            {
                const auto& acc = result[(size_t) src * 2 + (size_t) c];
                float* dst = stems[(size_t) src].getWritePointer (c);

                for (int i = 0; i < length; ++i)
                    dst[i] = (float) ((double) acc[(size_t) i] / shifts * safeStd + mean);
            }
        }
    }

    return true;
}

} // namespace stemsplitter
