#include "PluginProcessor.h"
#include "PluginEditor.h"

using namespace stemsplitter;

StemSplitterProcessor::StemSplitterProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true))
{
    modelFile = ModelLocator::findDefaultModel();
    useGpu = ModelLocator::getSettings().getBoolValue ("useGpu", false);

    worker.onStemsReady = [this] (std::shared_ptr<StemSet> newStems) { installStems (std::move (newStems)); };
}

StemSplitterProcessor::~StemSplitterProcessor()
{
    worker.onStemsReady = nullptr;
    worker.cancel();
}

//==============================================================================
bool StemSplitterProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto out = layouts.getMainOutputChannelSet();

    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;

    const auto in = layouts.getMainInputChannelSet();
    return in.isDisabled() || in == out;
}

void StemSplitterProcessor::prepareToPlay (double sampleRate, int)
{
    hostSampleRate = sampleRate;
    reloadPreviewIfNeeded();
}

void StemSplitterProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const int numSamples = buffer.getNumSamples();
    const int numOut = getTotalNumOutputChannels();

    for (int c = getTotalNumInputChannels(); c < numOut; ++c)
        buffer.clear (c, 0, numSamples);

    if (! passThrough.load())
        buffer.clear();

    const juce::SpinLock::ScopedTryLockType lock (stemsLock);

    if (! lock.isLocked() || stems == nullptr || stems->preview.empty())
        return;

    const auto& set = *stems;
    const juce::int64 length = set.previewLength;

    // Where are we?
    bool active = false;
    juce::int64 pos = 0;

    if (syncToHost.load())
    {
        if (auto* hostPlayHead = getPlayHead())
        {
            if (auto position = hostPlayHead->getPosition())
            {
                if (auto seconds = position->getTimeInSeconds())
                {
                    pos = (juce::int64) std::llround (*seconds * set.previewRate);
                    active = position->getIsPlaying();
                    playPosition = juce::jlimit ((juce::int64) 0, length, pos);
                }
            }
        }
    }
    else
    {
        active = playing.load();
        pos = playPosition.load();
    }

    if (! active)
        return;

    // Which stems?
    const auto solo = soloMask.load();
    const auto mute = muteMask.load();
    const int numStems = (int) set.preview.size();

    if (pos >= 0 && pos < length)
    {
        const int n = (int) juce::jmin ((juce::int64) numSamples, length - pos);

        for (int s = 0; s < numStems && s < 32; ++s)
        {
            const bool audible = solo != 0 ? ((solo >> s) & 1u) != 0 : ((mute >> s) & 1u) == 0;
            if (! audible)
                continue;

            const auto& stem = set.preview[(size_t) s];

            if (numOut == 1)
            {
                buffer.addFrom (0, 0, stem, 0, (int) pos, n, 0.5f);
                buffer.addFrom (0, 0, stem, 1, (int) pos, n, 0.5f);
            }
            else
            {
                for (int c = 0; c < juce::jmin (numOut, 2); ++c)
                    buffer.addFrom (c, 0, stem, c, (int) pos, n);
            }
        }
    }

    if (! syncToHost.load())
    {
        pos += numSamples;

        if (pos >= length)
        {
            pos = 0;
            playing = false;
        }

        playPosition = pos;
    }
}

//==============================================================================
void StemSplitterProcessor::setInputFile (const juce::File& file)
{
    inputFile = file;
}

void StemSplitterProcessor::setModelFile (const juce::File& file)
{
    modelFile = file;
    ModelLocator::setPreferredModel (file);
}

void StemSplitterProcessor::setUseGpu (bool shouldUse)
{
    useGpu = shouldUse;
    ModelLocator::getSettings().setValue ("useGpu", shouldUse);
    ModelLocator::getSettings().saveIfNeeded();
}

bool StemSplitterProcessor::startSeparation (juce::String& error)
{
    if (worker.isBusy())
    {
        error = "Already working - cancel first.";
        return false;
    }

    if (! inputFile.existsAsFile())
    {
        error = "Drop an audio file onto the plugin first.";
        return false;
    }

    if (! modelFile.existsAsFile())
        modelFile = ModelLocator::findDefaultModel();

    if (! modelFile.existsAsFile())
    {
        error = "No AI model found. Click \"Model...\" to choose htdemucs.onnx.";
        return false;
    }

    setPlaying (false);

    SeparationWorker::Job job;
    job.kind = SeparationWorker::Job::Kind::separate;
    job.input = inputFile;
    job.model = modelFile;
    job.useGpu = useGpu;
    job.shifts = shifts;
    job.previewRate = hostSampleRate.load();

    return worker.start (job);
}

std::shared_ptr<const StemSet> StemSplitterProcessor::getStems() const
{
    const juce::SpinLock::ScopedLockType lock (stemsLock);
    return stems;
}

void StemSplitterProcessor::installStems (std::shared_ptr<StemSet> newStems)
{
    // Instrumental is a mix of the other stems, so don't play it by default
    juce::uint32 newMute = 0;
    for (int i = 0; i < newStems->names.size() && i < 32; ++i)
        if (newStems->names[i] == "Instrumental")
            newMute |= (1u << i);

    std::shared_ptr<const StemSet> old = std::move (newStems);

    {
        const juce::SpinLock::ScopedLockType lock (stemsLock);
        std::swap (stems, old);

        const bool sameLayout = old != nullptr && old->names == stems->names;

        if (masksPending.exchange (false))
        {
            muteMask = pendingMute;
            soloMask = pendingSolo;
        }
        else if (! sameLayout)
        {
            muteMask = newMute;
            soloMask = 0;
        }

        if (playPosition.load() >= stems->previewLength)
            playPosition = 0;
    }

    ++stemsVersion;
    // `old` is released here, outside the lock and off the audio thread
}

void StemSplitterProcessor::reloadPreviewIfNeeded()
{
    auto current = getStems();

    if (current == nullptr || worker.isBusy())
        return;

    if (std::abs (current->previewRate - hostSampleRate.load()) < 0.5)
        return;

    SeparationWorker::Job job;
    job.kind = SeparationWorker::Job::Kind::loadExisting;
    job.input = current->sourceFile;
    job.existingFolder = current->folder;
    job.existingFiles = current->files;
    job.existingNames = current->names;
    job.previewRate = hostSampleRate.load();
    worker.start (job);
}

//==============================================================================
void StemSplitterProcessor::setPlaying (bool shouldPlay)
{
    playing = shouldPlay;
}

void StemSplitterProcessor::setPlayPosition (double proportion)
{
    if (auto s = getStems())
        playPosition = (juce::int64) (juce::jlimit (0.0, 1.0, proportion) * (double) s->previewLength);
}

double StemSplitterProcessor::getPlayPosition() const
{
    if (auto s = getStems())
        if (s->previewLength > 0)
            return juce::jlimit (0.0, 1.0, (double) playPosition.load() / (double) s->previewLength);

    return 0.0;
}

void StemSplitterProcessor::setStemMuted (int index, bool muted)
{
    if (index < 0 || index >= 32) return;
    const auto bit = 1u << index;
    auto m = muteMask.load();
    muteMask = muted ? (m | bit) : (m & ~bit);
}

void StemSplitterProcessor::setStemSoloed (int index, bool soloed)
{
    if (index < 0 || index >= 32) return;
    const auto bit = 1u << index;
    auto m = soloMask.load();
    soloMask = soloed ? (m | bit) : (m & ~bit);
}

//==============================================================================
void StemSplitterProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    juce::XmlElement xml ("StemSplitter");
    xml.setAttribute ("version", 1);
    xml.setAttribute ("input", inputFile.getFullPathName());
    xml.setAttribute ("model", modelFile.getFullPathName());
    xml.setAttribute ("shifts", shifts);
    xml.setAttribute ("sync", syncToHost.load());
    xml.setAttribute ("passThrough", passThrough.load());
    xml.setAttribute ("mute", (int) muteMask.load());
    xml.setAttribute ("solo", (int) soloMask.load());

    if (auto s = getStems())
    {
        xml.setAttribute ("folder", s->folder.getFullPathName());

        for (int i = 0; i < s->files.size(); ++i)
        {
            auto* e = xml.createNewChildElement ("Stem");
            e->setAttribute ("name", s->names[i]);
            e->setAttribute ("file", s->files[i].getFullPathName());
        }
    }

    copyXmlToBinary (xml, destData);
}

void StemSplitterProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto xml = getXmlFromBinary (data, sizeInBytes);

    if (xml == nullptr || ! xml->hasTagName ("StemSplitter"))
        return;

    auto pathOrEmpty = [] (const juce::String& p) { return juce::File::isAbsolutePath (p) ? juce::File (p) : juce::File(); };

    inputFile = pathOrEmpty (xml->getStringAttribute ("input"));

    auto savedModel = pathOrEmpty (xml->getStringAttribute ("model"));
    if (savedModel.existsAsFile())
        modelFile = savedModel;

    shifts = juce::jlimit (1, 4, xml->getIntAttribute ("shifts", 1));
    syncToHost = xml->getBoolAttribute ("sync", false);
    passThrough = xml->getBoolAttribute ("passThrough", true);

    SeparationWorker::Job job;
    job.kind = SeparationWorker::Job::Kind::loadExisting;
    job.input = inputFile;
    job.existingFolder = pathOrEmpty (xml->getStringAttribute ("folder"));
    job.previewRate = hostSampleRate.load();

    for (auto* e : xml->getChildWithTagNameIterator ("Stem"))
    {
        auto f = pathOrEmpty (e->getStringAttribute ("file"));
        if (f.existsAsFile())
        {
            job.existingFiles.add (f);
            job.existingNames.add (e->getStringAttribute ("name"));
        }
    }

    const auto savedMute = (juce::uint32) xml->getIntAttribute ("mute", 0);
    const auto savedSolo = (juce::uint32) xml->getIntAttribute ("solo", 0);

    if (! job.existingFiles.isEmpty())
    {
        // Mute/solo are applied once the stems are back in memory
        pendingMute = savedMute;
        pendingSolo = savedSolo;
        masksPending = true;

        if (! worker.start (job))
            masksPending = false;
    }
}

//==============================================================================
juce::AudioProcessorEditor* StemSplitterProcessor::createEditor()
{
    return new StemSplitterEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new StemSplitterProcessor();
}
