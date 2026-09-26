#include "AudioUtils.h"

#include <cmath>

namespace stemsplitter
{
namespace
{
double besselI0 (double x)
{
    double sum = 1.0, term = 1.0;
    for (int k = 1; k < 50; ++k)
    {
        term *= (x / (2.0 * k)) * (x / (2.0 * k));
        sum += term;
        if (term < 1.0e-12 * sum)
            break;
    }
    return sum;
}

// sinc(u) * kaiser(u / zeroCrossings), tabulated for u in [0, zeroCrossings]
struct SincTable
{
    static constexpr int zeroCrossings = 32;
    static constexpr int oversampling = 512;
    static constexpr double beta = 9.0;

    std::vector<float> table;

    SincTable()
    {
        const int size = zeroCrossings * oversampling + 2;
        table.resize ((size_t) size);
        const double i0Beta = besselI0 (beta);

        for (int i = 0; i < size; ++i)
        {
            const double u = (double) i / oversampling;
            const double r = u / zeroCrossings;
            const double win = r < 1.0 ? besselI0 (beta * std::sqrt (1.0 - r * r)) / i0Beta : 0.0;
            const double sinc = u < 1.0e-12 ? 1.0 : std::sin (juce::MathConstants<double>::pi * u)
                                                 / (juce::MathConstants<double>::pi * u);
            table[(size_t) i] = (float) (sinc * win);
        }
    }

    float lookup (double u) const noexcept
    {
        u = std::abs (u);
        if (u >= zeroCrossings)
            return 0.0f;

        const double pos = u * oversampling;
        const int idx = (int) pos;
        const float frac = (float) (pos - idx);
        return table[(size_t) idx] + frac * (table[(size_t) idx + 1] - table[(size_t) idx]);
    }
};
} // namespace

void resample (const juce::AudioBuffer<float>& input, double inRate, double outRate,
               juce::AudioBuffer<float>& output, const std::function<bool()>& shouldContinue)
{
    const int numChannels = input.getNumChannels();
    const int inLength = input.getNumSamples();

    if (std::abs (inRate - outRate) < 1.0e-6 || inLength == 0)
    {
        output.makeCopyOf (input);
        return;
    }

    static const SincTable sinc;

    const double ratio = outRate / inRate;
    const int outLength = (int) std::llround ((double) inLength * ratio);
    const double cutoff = juce::jmin (1.0, ratio) * 0.97;   // leave a little transition band
    const double halfWidth = SincTable::zeroCrossings / cutoff; // in input samples

    output.setSize (numChannels, outLength, false, false, true);

    // Mirror the signal at both ends so the first/last samples don't droop
    // (a plain zero-padded filter would fade the edges in and out).
    const int margin = (int) std::ceil (halfWidth) + 2;
    std::vector<float> extended ((size_t) (inLength + 2 * margin));

    for (int c = 0; c < numChannels; ++c)
    {
        const float* x = input.getReadPointer (c);

        for (int i = -margin; i < inLength + margin; ++i)
        {
            int j = i;
            // Odd-symmetric (point) reflection keeps slopes continuous at the edges
            float v;
            if (j < 0)
            {
                j = juce::jmin (-j, inLength - 1);
                v = 2.0f * x[0] - x[j];
            }
            else if (j >= inLength)
            {
                j = juce::jmax (2 * (inLength - 1) - j, 0);
                v = 2.0f * x[inLength - 1] - x[j];
            }
            else
            {
                v = x[j];
            }
            extended[(size_t) (i + margin)] = v;
        }

        const float* xe = extended.data() + margin;
        float* out = output.getWritePointer (c);

        for (int n = 0; n < outLength; ++n)
        {
            if ((n & 0xffff) == 0 && shouldContinue && ! shouldContinue())
                return;

            const double t = (double) n / ratio;   // position in input samples
            const int first = (int) std::ceil (t - halfWidth);
            const int last  = (int) std::floor (t + halfWidth);
            double acc = 0.0;

            for (int i = first; i <= last; ++i)
                acc += (double) xe[i] * sinc.lookup ((t - i) * cutoff);

            out[n] = (float) (acc * cutoff);
        }
    }
}

bool readAudioFileAsStereo (juce::AudioFormatManager& formats, const juce::File& file,
                            juce::AudioBuffer<float>& output, double& sampleRate,
                            juce::String& error)
{
    std::unique_ptr<juce::AudioFormatReader> reader (formats.createReaderFor (file));

    if (reader == nullptr)
    {
        error = "Couldn't read " + file.getFileName() + " - unsupported or damaged audio file.";
        return false;
    }

    if (reader->lengthInSamples <= 0)
    {
        error = file.getFileName() + " contains no audio.";
        return false;
    }

    if (reader->lengthInSamples > (juce::int64) std::numeric_limits<int>::max() / 2)
    {
        error = file.getFileName() + " is too long.";
        return false;
    }

    const int length = (int) reader->lengthInSamples;
    sampleRate = reader->sampleRate;

    output.setSize (2, length);
    output.clear();

    if (reader->numChannels == 1)
    {
        reader->read (&output, 0, length, 0, true, false);
        output.copyFrom (1, 0, output, 0, 0, length);
    }
    else
    {
        reader->read (&output, 0, length, 0, true, true);
    }

    return true;
}

bool writeWav (const juce::File& file, const juce::AudioBuffer<float>& buffer,
               double sampleRate, juce::String& error)
{
    file.deleteFile();
    auto stream = file.createOutputStream();

    if (stream == nullptr)
    {
        error = "Couldn't write to " + file.getFullPathName();
        return false;
    }

    juce::WavAudioFormat wav;
    std::unique_ptr<juce::AudioFormatWriter> writer (
        wav.createWriterFor (stream.get(), sampleRate, (unsigned int) buffer.getNumChannels(),
                             32, {}, 0));

    if (writer == nullptr)
    {
        error = "Couldn't create a WAV writer for " + file.getFullPathName();
        return false;
    }

    stream.release();   // the writer owns the stream now

    if (! writer->writeFromAudioSampleBuffer (buffer, 0, buffer.getNumSamples()))
    {
        error = "Failed while writing " + file.getFullPathName() + " (disk full?)";
        return false;
    }

    return true;
}

std::vector<float> computePeaks (const juce::AudioBuffer<float>& buffer, int numBuckets)
{
    std::vector<float> peaks ((size_t) numBuckets * 2, 0.0f);
    const int length = buffer.getNumSamples();

    if (length == 0 || numBuckets <= 0)
        return peaks;

    for (int b = 0; b < numBuckets; ++b)
    {
        const int start = (int) ((juce::int64) b * length / numBuckets);
        const int end   = juce::jmax (start + 1, (int) ((juce::int64) (b + 1) * length / numBuckets));
        float lo = 0.0f, hi = 0.0f;

        for (int c = 0; c < buffer.getNumChannels(); ++c)
        {
            auto range = juce::FloatVectorOperations::findMinAndMax (buffer.getReadPointer (c, start),
                                                                     juce::jmin (end, length) - start);
            lo = juce::jmin (lo, range.getStart());
            hi = juce::jmax (hi, range.getEnd());
        }

        peaks[(size_t) b * 2]     = lo;
        peaks[(size_t) b * 2 + 1] = hi;
    }

    return peaks;
}
} // namespace stemsplitter
