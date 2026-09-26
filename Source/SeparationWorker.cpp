#include "SeparationWorker.h"
#include "engine/AudioUtils.h"

#include <mutex>

namespace stemsplitter
{
//==============================================================================
// The loaded network is ~170 MB and takes a few seconds to prepare, so every plugin
// instance in the project shares one copy.
struct ModelCache
{
    std::shared_ptr<DemucsModel> get (const juce::File& file, bool gpu, juce::String& error)
    {
        std::lock_guard<std::mutex> lock (mutex);

        if (model != nullptr && model->getFile() == file && gpu == modelUsesGpu)
            return model;

        model.reset();
        const int threads = juce::jmax (1, juce::SystemStats::getNumPhysicalCpus() - 1);   // leave a core for FL's audio
        std::shared_ptr<DemucsModel> loaded (DemucsModel::load (file, gpu ? ComputeDevice::gpu : ComputeDevice::cpu,
                                                                threads, error).release());
        model = loaded;
        modelUsesGpu = gpu;
        return model;
    }

    std::mutex mutex;
    std::shared_ptr<DemucsModel> model;
    bool modelUsesGpu = false;
};

//==============================================================================
SeparationWorker::SeparationWorker() : juce::Thread ("Stem Splitter worker")
{
    formats.registerBasicFormats();
}

SeparationWorker::~SeparationWorker()
{
    stopThread (30000);   // inference checks for cancellation between segments
}

bool SeparationWorker::start (const Job& job)
{
    if (isThreadRunning())
        return false;

    currentJob = job;
    setStage (Stage::idle, {}, 0.0f);
    return startThread (juce::Thread::Priority::low);
}

void SeparationWorker::cancel()
{
    signalThreadShouldExit();
}

bool SeparationWorker::isBusy() const
{
    return isThreadRunning();
}

juce::String SeparationWorker::getMessage() const
{
    const juce::ScopedLock sl (messageLock);
    return message;
}

juce::String SeparationWorker::getDeviceDescription() const
{
    const juce::ScopedLock sl (messageLock);
    return deviceDescription;
}

void SeparationWorker::setStage (Stage s, const juce::String& text, float progressValue)
{
    {
        const juce::ScopedLock sl (messageLock);
        message = text;
    }
    progress = progressValue;
    stage = s;
}

juce::File SeparationWorker::chooseOutputFolder (const juce::File& input)
{
    const auto folderName = input.getFileNameWithoutExtension() + " Stems";
    auto nextToSource = input.getParentDirectory().getChildFile (folderName);

    if (nextToSource.createDirectory().wasOk() && nextToSource.hasWriteAccess())
        return nextToSource;

    // e.g. the song lives on a read-only drive or inside FL's install folder
    auto fallback = juce::File::getSpecialLocation (juce::File::userDocumentsDirectory)
                        .getChildFile ("Stem Splitter").getChildFile (folderName);
    fallback.createDirectory();
    return fallback;
}

void SeparationWorker::run()
{
    const auto job = currentJob;
    const bool ok = job.kind == Job::Kind::separate ? runSeparation (job) : runLoadExisting (job);

    if (! ok && stage.load() != Stage::failed)
        setStage (Stage::cancelled, "Cancelled", 0.0f);
}

bool SeparationWorker::buildPreview (StemSet& set, std::vector<juce::AudioBuffer<float>>& buffers,
                                     double buffersRate, double previewRate)
{
    set.previewRate = previewRate;
    set.preview.clear();
    set.peaks.clear();

    for (auto& b : buffers)
    {
        if (threadShouldExit())
            return false;

        juce::AudioBuffer<float> resampled;
        resample (b, buffersRate, previewRate, resampled, [this] { return ! threadShouldExit(); });
        set.peaks.push_back (computePeaks (resampled, 1200));
        set.preview.push_back (std::move (resampled));
    }

    set.previewLength = set.preview.empty() ? 0 : set.preview.front().getNumSamples();
    return ! threadShouldExit();
}

bool SeparationWorker::runSeparation (const Job& job)
{
    juce::String error;
    auto fail = [this, &error]
    {
        setStage (Stage::failed, error, 0.0f);
        return false;
    };

    // 1. Model
    setStage (Stage::loadingModel, "Loading AI model...", 0.01f);
    // The cache lives while any separation is running, so the ~300 MB of model memory
    // is given back to FL once all jobs are done.
    juce::SharedResourcePointer<ModelCache> cache;
    auto model = cache->get (job.model, job.useGpu, error);
    if (model == nullptr)
        return fail();

    {
        const juce::ScopedLock sl (messageLock);
        deviceDescription = model->getDeviceDescription();
    }

    if (threadShouldExit()) return false;

    // 2. Decode + resample to 44.1 kHz
    setStage (Stage::decoding, "Reading " + job.input.getFileName() + "...", 0.03f);

    juce::AudioBuffer<float> audio, audio44;
    double sourceRate = 0.0;

    if (! readAudioFileAsStereo (formats, job.input, audio, sourceRate, error))
        return fail();

    resample (audio, sourceRate, kModelSampleRate, audio44, [this] { return ! threadShouldExit(); });
    if (threadShouldExit()) return false;

    // 3. Separate
    setStage (Stage::separating, "Separating stems...", 0.05f);

    SeparationOptions options;
    options.shifts = job.shifts;

    std::vector<juce::AudioBuffer<float>> stems;
    const auto startTime = juce::Time::getMillisecondCounterHiRes();

    const bool ok = separate (*model, audio44, stems, options, [this, startTime] (float p)
    {
        const double elapsed = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;
        juce::String text = "Separating stems... " + juce::String ((int) (p * 100.0f)) + "%";

        if (p > 0.02f)
        {
            const int remaining = (int) (elapsed / p * (1.0 - p));
            text << "  (about " << (remaining >= 60 ? juce::String (remaining / 60) + " min " : juce::String())
                 << (remaining % 60) << " s left)";
        }

        setStage (Stage::separating, text, 0.05f + 0.85f * p);
        return ! threadShouldExit();
    }, error);

    audio44.setSize (0, 0);

    if (! ok)
        return error.isEmpty() ? false : fail();

    // 4. Back to the file's own sample rate, exactly the original length, and write
    setStage (Stage::writing, "Writing stems...", 0.9f);

    auto set = std::make_shared<StemSet>();
    set->sourceFile = job.input;
    set->folder = chooseOutputFolder (job.input);
    set->names = model->getSourceNames();

    std::vector<juce::AudioBuffer<float>> outBuffers;

    for (auto& stem : stems)
    {
        juce::AudioBuffer<float> out;
        resample (stem, kModelSampleRate, sourceRate, out, [this] { return ! threadShouldExit(); });
        out.setSize (2, audio.getNumSamples(), true, true, false);
        stem.setSize (0, 0);
        outBuffers.push_back (std::move (out));

        if (threadShouldExit()) return false;
    }

    if (job.writeInstrumental)
    {
        const int vocals = set->names.indexOf ("Vocals", true);

        if (vocals >= 0 && outBuffers.size() > 2)
        {
            juce::AudioBuffer<float> instrumental (2, audio.getNumSamples());
            instrumental.clear();

            for (int s = 0; s < (int) outBuffers.size(); ++s)
                if (s != vocals)
                    for (int c = 0; c < 2; ++c)
                        instrumental.addFrom (c, 0, outBuffers[(size_t) s], c, 0, instrumental.getNumSamples());

            outBuffers.push_back (std::move (instrumental));
            set->names.add ("Instrumental");
        }
    }

    const auto baseName = job.input.getFileNameWithoutExtension();

    for (int s = 0; s < (int) outBuffers.size(); ++s)
    {
        auto file = set->folder.getChildFile (juce::File::createLegalFileName (baseName + " - " + set->names[s] + ".wav"));

        if (! writeWav (file, outBuffers[(size_t) s], sourceRate, error))
            return fail();

        set->files.add (file);
        progress = 0.9f + 0.05f * (float) (s + 1) / (float) outBuffers.size();
    }

    // 5. Preview copy at the host's sample rate
    setStage (Stage::loadingPreview, "Preparing preview...", 0.96f);

    if (! buildPreview (*set, outBuffers, sourceRate, job.previewRate))
        return false;

    if (onStemsReady)
        onStemsReady (set);

    const double seconds = (juce::Time::getMillisecondCounterHiRes() - startTime) / 1000.0;
    setStage (Stage::finished, "Done in " + juce::String (seconds, 1) + " s - drag the stems into your Playlist", 1.0f);
    return true;
}

bool SeparationWorker::runLoadExisting (const Job& job)
{
    setStage (Stage::loadingPreview, "Loading stems...", 0.0f);

    auto set = std::make_shared<StemSet>();
    set->sourceFile = job.input;
    set->folder = job.existingFolder;

    std::vector<juce::AudioBuffer<float>> buffers;
    double rate = 0.0;

    for (int i = 0; i < job.existingFiles.size(); ++i)
    {
        juce::AudioBuffer<float> b;
        double r = 0.0;
        juce::String error;

        if (! readAudioFileAsStereo (formats, job.existingFiles[i], b, r, error))
            continue;   // a stem was deleted or moved - just skip it

        if (rate <= 0.0)
            rate = r;

        if (std::abs (r - rate) > 0.5)
        {
            juce::AudioBuffer<float> converted;
            resample (b, r, rate, converted);
            b = std::move (converted);
        }

        set->names.add (job.existingNames[i]);
        set->files.add (job.existingFiles[i]);
        buffers.push_back (std::move (b));

        if (threadShouldExit()) return false;
    }

    if (buffers.empty())
    {
        setStage (Stage::idle, {}, 0.0f);
        return true;
    }

    // Stems should all be the same length, but be tolerant of edited files
    int length = 0;
    for (auto& b : buffers) length = juce::jmax (length, b.getNumSamples());
    for (auto& b : buffers) b.setSize (2, length, true, true, false);

    if (! buildPreview (*set, buffers, rate, job.previewRate))
        return false;

    if (onStemsReady)
        onStemsReady (set);

    setStage (Stage::finished, "Loaded " + juce::String (set->files.size()) + " stems", 1.0f);
    return true;
}
} // namespace stemsplitter
