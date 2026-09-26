#include "PluginEditor.h"

using namespace stemsplitter;
namespace SC = stemsplitter::Colours;

namespace
{
juce::Font font (float height, bool bold = false)
{
    return juce::Font (juce::FontOptions (height, bold ? juce::Font::bold : juce::Font::plain));
}

juce::String formatTime (double seconds)
{
    const int s = (int) seconds;
    return juce::String (s / 60) + ":" + juce::String (s % 60).paddedLeft ('0', 2);
}

void startExternalDrag (const juce::StringArray& files, juce::Component* source)
{
    if (! files.isEmpty())
        juce::DragAndDropContainer::performExternalDragDropOfFiles (files, false, source);
}
} // namespace

//==============================================================================
bool DropZone::isAudioFile (const juce::File& f)
{
    return f.hasFileExtension ("wav;mp3;flac;ogg;aif;aiff;m4a;wma");
}

void DropZone::setFileInfo (const juce::String& newTitle, const juce::String& newDetails)
{
    title = newTitle;
    details = newDetails;
    repaint();
}

void DropZone::paint (juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced (1.0f);

    g.setColour (dragOver ? SC::accent.withAlpha (0.12f) : SC::panel);
    g.fillRoundedRectangle (bounds, 8.0f);

    juce::Path border;
    border.addRoundedRectangle (bounds, 8.0f);
    const float dashes[] = { 6.0f, 5.0f };
    juce::PathStrokeType (1.5f).createDashedStroke (border, border, dashes, 2);
    g.setColour (dragOver ? SC::accent : SC::outline);
    g.fillPath (border);

    auto text = getLocalBounds().reduced (20, 14);

    if (title.isEmpty())
    {
        g.setColour (SC::text);
        g.setFont (font (18.0f, true));
        g.drawText ("Drop a song here", text.removeFromTop (text.getHeight() / 2), juce::Justification::bottomLeft);
        g.setColour (SC::textDim);
        g.setFont (font (13.0f));
        g.drawText ("or click to browse  -  WAV, MP3, FLAC, OGG, AIFF", text, juce::Justification::topLeft);
    }
    else
    {
        g.setColour (SC::text);
        g.setFont (font (17.0f, true));
        g.drawFittedText (title, text.removeFromTop (text.getHeight() / 2), juce::Justification::bottomLeft, 1);
        g.setColour (SC::textDim);
        g.setFont (font (13.0f));
        g.drawFittedText (details, text, juce::Justification::topLeft, 1);
    }
}

void DropZone::mouseUp (const juce::MouseEvent& e)
{
    if (! acceptDrops || ! e.mouseWasClicked())
        return;

    chooser = std::make_unique<juce::FileChooser> ("Choose a song to separate", juce::File(),
                                                   "*.wav;*.mp3;*.flac;*.ogg;*.aif;*.aiff;*.m4a;*.wma");
    chooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                          [this] (const juce::FileChooser& fc)
                          {
                              auto f = fc.getResult();
                              if (f.existsAsFile() && onFileChosen)
                                  onFileChosen (f);
                          });
}

bool DropZone::isInterestedInFileDrag (const juce::StringArray& files)
{
    if (! acceptDrops)
        return false;

    for (auto& f : files)
        if (isAudioFile (juce::File (f)))
            return true;

    return false;
}

void DropZone::filesDropped (const juce::StringArray& files, int, int)
{
    dragOver = false;
    repaint();

    for (auto& f : files)
    {
        juce::File file (f);
        if (isAudioFile (file) && onFileChosen)
        {
            onFileChosen (file);
            return;
        }
    }
}

//==============================================================================
void FileDragButton::paint (juce::Graphics& g)
{
    const bool enabled = isEnabled();
    auto bounds = getLocalBounds().toFloat().reduced (0.5f);

    g.setColour (enabled && isMouseOver() ? SC::panelLight.brighter (0.12f) : SC::panelLight);
    g.fillRoundedRectangle (bounds, 5.0f);

    // grip dots
    g.setColour (enabled ? SC::accent : SC::textDim);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 2; ++col)
            g.fillEllipse (10.0f + col * 5.0f, bounds.getCentreY() - 6.0f + row * 5.0f, 2.5f, 2.5f);

    g.setColour (enabled ? SC::text : SC::textDim);
    g.setFont (font (13.0f));
    g.drawText (label, getLocalBounds().withTrimmedLeft (24), juce::Justification::centred);
}

void FileDragButton::mouseDrag (const juce::MouseEvent& e)
{
    if (dragging || ! isEnabled() || e.getDistanceFromDragStart() < 4 || getFiles == nullptr)
        return;

    dragging = true;
    startExternalDrag (getFiles(), this);
}

//==============================================================================
StemRow::StemRow (StemSplitterProcessor& p, int i) : processor (p), index (i)
{
    for (auto* b : { &muteButton, &soloButton })
    {
        addAndMakeVisible (b);
        b->setClickingTogglesState (true);
    }

    muteButton.setTooltip ("Mute this stem in the preview");
    soloButton.setTooltip ("Solo this stem in the preview");
    muteButton.setColour (juce::TextButton::buttonOnColourId, SC::danger);
    soloButton.setColour (juce::TextButton::buttonOnColourId, juce::Colour (0xffffd74a));

    muteButton.onClick = [this] { processor.setStemMuted (index, muteButton.getToggleState()); };
    soloButton.onClick = [this] { processor.setStemSoloed (index, soloButton.getToggleState()); };

    refreshButtons();
}

void StemRow::refreshButtons()
{
    muteButton.setToggleState (processor.isStemMuted (index), juce::dontSendNotification);
    soloButton.setToggleState (processor.isStemSoloed (index), juce::dontSendNotification);
}

juce::Rectangle<int> StemRow::getHandleArea() const
{
    return getLocalBounds().removeFromLeft (130);
}

juce::Rectangle<int> StemRow::getWaveArea() const
{
    return getLocalBounds().withTrimmedLeft (138).withTrimmedRight (72).reduced (0, 6);
}

void StemRow::resized()
{
    auto r = getLocalBounds().removeFromRight (64).reduced (0, 11);
    soloButton.setBounds (r.removeFromRight (28));
    r.removeFromRight (6);
    muteButton.setBounds (r.removeFromRight (28));
}

void StemRow::paint (juce::Graphics& g)
{
    auto stems = processor.getStems();
    if (stems == nullptr || index >= stems->names.size())
        return;

    const auto name = stems->names[index];
    const auto colour = SC::forStem (name);
    auto bounds = getLocalBounds().toFloat();

    g.setColour (SC::panel);
    g.fillRoundedRectangle (bounds.reduced (0.0f, 2.0f), 6.0f);

    // Handle: colour strip, grip, name
    auto handle = getHandleArea().toFloat();
    const bool hot = isMouseOver() && getHandleArea().contains (getMouseXYRelative());

    if (hot)
    {
        g.setColour (SC::panelLight);
        g.fillRoundedRectangle (handle.reduced (0.0f, 2.0f), 6.0f);
    }

    g.setColour (colour);
    g.fillRoundedRectangle (handle.removeFromLeft (5.0f).reduced (0.0f, 6.0f), 2.0f);

    g.setColour (hot ? SC::text : SC::textDim);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 2; ++col)
            g.fillEllipse (14.0f + col * 5.0f, bounds.getCentreY() - 6.0f + row * 5.0f, 2.5f, 2.5f);

    g.setColour (SC::text);
    g.setFont (font (15.0f, true));
    g.drawText (name, getHandleArea().withTrimmedLeft (30), juce::Justification::centredLeft);

    // Waveform
    auto wave = getWaveArea();
    if (index < (int) stems->peaks.size() && ! wave.isEmpty())
    {
        const auto& peaks = stems->peaks[(size_t) index];
        const int buckets = (int) peaks.size() / 2;
        const float mid = (float) wave.getCentreY();
        const float halfH = wave.getHeight() * 0.5f;

        bool anySolo = false;
        for (int i = 0; i < stems->names.size(); ++i)
            anySolo = anySolo || processor.isStemSoloed (i);

        const bool audible = anySolo ? processor.isStemSoloed (index) : ! processor.isStemMuted (index);

        g.setColour (colour.withAlpha (audible ? 0.85f : 0.25f));

        for (int x = 0; x < wave.getWidth(); ++x)
        {
            const int b0 = x * buckets / wave.getWidth();
            const int b1 = juce::jmax (b0 + 1, (x + 1) * buckets / wave.getWidth());
            float lo = 0.0f, hi = 0.0f;

            for (int b = b0; b < b1 && b < buckets; ++b)
            {
                lo = juce::jmin (lo, peaks[(size_t) b * 2]);
                hi = juce::jmax (hi, peaks[(size_t) b * 2 + 1]);
            }

            const float top = mid - juce::jmin (1.0f, hi) * halfH;
            const float bottom = mid - juce::jmax (-1.0f, lo) * halfH;
            g.drawVerticalLine (wave.getX() + x, top, juce::jmax (top + 1.0f, bottom));
        }

        // Playhead
        const float px = (float) wave.getX() + (float) processor.getPlayPosition() * (float) wave.getWidth();
        g.setColour (SC::text.withAlpha (0.9f));
        g.drawLine (px, (float) wave.getY(), px, (float) wave.getBottom(), 1.5f);
    }
}

void StemRow::mouseMove (const juce::MouseEvent&)
{
    repaint();
    setMouseCursor (getHandleArea().contains (getMouseXYRelative()) ? juce::MouseCursor::DraggingHandCursor
                                                                     : juce::MouseCursor::NormalCursor);
}

void StemRow::mouseDown (const juce::MouseEvent& e)
{
    dragging = false;
    seeking = getWaveArea().contains (e.getPosition()) && ! processor.getSyncToHost();

    if (seeking)
        processor.setPlayPosition ((e.position.x - (float) getWaveArea().getX()) / (float) getWaveArea().getWidth());
}

void StemRow::mouseDrag (const juce::MouseEvent& e)
{
    if (seeking)
    {
        processor.setPlayPosition ((e.position.x - (float) getWaveArea().getX()) / (float) getWaveArea().getWidth());
        return;
    }

    if (dragging || e.getDistanceFromDragStart() < 4)
        return;

    auto stems = processor.getStems();
    if (stems == nullptr || index >= stems->files.size())
        return;

    dragging = true;
    startExternalDrag ({ stems->files[index].getFullPathName() }, this);
}

void StemRow::mouseUp (const juce::MouseEvent&)
{
    dragging = seeking = false;
}

//==============================================================================
StemSplitterEditor::StemSplitterEditor (StemSplitterProcessor& p)
    : AudioProcessorEditor (&p), processor (p)
{
    setLookAndFeel (&lookAndFeel);
    formats.registerBasicFormats();

    addAndMakeVisible (dropZone);
    dropZone.onFileChosen = [this] (const juce::File& f) { loadFile (f); };

    addAndMakeVisible (separateButton);
    separateButton.setColour (juce::TextButton::buttonColourId, SC::accent);
    separateButton.setColour (juce::TextButton::textColourOffId, juce::Colours::black);
    separateButton.onClick = [this]
    {
        juce::String error;
        if (! processor.startSeparation (error))
        {
            statusLabel.setText (error, juce::dontSendNotification);
            statusLabel.setColour (juce::Label::textColourId, SC::danger);

            if (! processor.getModelFile().existsAsFile())
                showModelHelp();
        }
    };

    addChildComponent (cancelButton);
    cancelButton.onClick = [this] { processor.cancelSeparation(); };

    addAndMakeVisible (modelButton);
    modelButton.onClick = [this] { showModelMenu(); };

    addAndMakeVisible (modelLabel);
    modelLabel.setJustificationType (juce::Justification::centredRight);
    modelLabel.setColour (juce::Label::textColourId, SC::textDim);
    modelLabel.setFont (font (12.0f));

    addAndMakeVisible (qualityBox);
    qualityBox.addItem ("Standard", 1);
    qualityBox.addItem ("High (2x slower)", 2);
    qualityBox.addItem ("Best (4x slower)", 4);
    qualityBox.setSelectedId (processor.getQualityShifts(), juce::dontSendNotification);
    qualityBox.setTooltip ("Runs the model several times with small time shifts and averages them");
    qualityBox.onChange = [this] { processor.setQualityShifts (qualityBox.getSelectedId()); };

    addAndMakeVisible (gpuToggle);
    gpuToggle.setToggleState (processor.getUseGpu(), juce::dontSendNotification);
    gpuToggle.setTooltip ("Use the graphics card (DirectML on Windows, CoreML on Mac) if this build supports it");
    gpuToggle.onClick = [this] { processor.setUseGpu (gpuToggle.getToggleState()); };

    addAndMakeVisible (progressBar);
    progressBar.setPercentageDisplay (false);

    addAndMakeVisible (statusLabel);
    statusLabel.setFont (font (13.0f));
    statusLabel.setColour (juce::Label::textColourId, SC::textDim);

    addAndMakeVisible (dragAllButton);
    dragAllButton.getFiles = [this]
    {
        juce::StringArray files;
        if (auto stems = processor.getStems())
            for (int i = 0; i < stems->files.size(); ++i)
                if (stems->names[i] != "Instrumental")   // it duplicates the others
                    files.add (stems->files[i].getFullPathName());
        return files;
    };

    addAndMakeVisible (openFolderButton);
    openFolderButton.onClick = [this]
    {
        if (auto stems = processor.getStems())
            if (! stems->files.isEmpty())
                stems->files.getFirst().revealToUser();
    };

    addAndMakeVisible (playButton);
    playButton.onClick = [this] { processor.setPlaying (! processor.isPlaying()); };

    addAndMakeVisible (syncToggle);
    syncToggle.setToggleState (processor.getSyncToHost(), juce::dontSendNotification);
    syncToggle.setTooltip ("Play the stems in time with FL's song position instead of the Play button");
    syncToggle.onClick = [this] { processor.setSyncToHost (syncToggle.getToggleState()); };

    addAndMakeVisible (passThroughToggle);
    passThroughToggle.setToggleState (processor.getPassThroughInput(), juce::dontSendNotification);
    passThroughToggle.setTooltip ("Let the mixer track's own audio through as well as the stem preview");
    passThroughToggle.onClick = [this] { processor.setPassThroughInput (passThroughToggle.getToggleState()); };

    addAndMakeVisible (timeLabel);
    timeLabel.setFont (font (13.0f));
    timeLabel.setJustificationType (juce::Justification::centredRight);
    timeLabel.setColour (juce::Label::textColourId, SC::textDim);

    if (processor.getInputFile().existsAsFile())
        loadFile (processor.getInputFile());

    updateModelLabel();
    setSize (780, 580);
    startTimerHz (30);
    timerCallback();
}

StemSplitterEditor::~StemSplitterEditor()
{
    stopTimer();
    setLookAndFeel (nullptr);
}

void StemSplitterEditor::paint (juce::Graphics& g)
{
    g.fillAll (SC::background);

    auto header = getLocalBounds().removeFromTop (58).reduced (20, 0);
    g.setColour (SC::accent);
    g.fillRoundedRectangle (juce::Rectangle<float> ((float) header.getX(), 20.0f, 6.0f, 20.0f), 2.0f);
    g.setColour (SC::text);
    g.setFont (font (20.0f, true));
    g.drawText ("STEM SPLITTER", header.withTrimmedLeft (14), juce::Justification::centredLeft);

    // Section title above the stems
    g.setColour (SC::textDim);
    g.setFont (font (12.0f, true));
    g.drawText ("STEMS", stemsTitleArea, juce::Justification::centredLeft);

    if (stemRows.isEmpty())
    {
        g.setFont (font (14.0f));
        g.drawText ("Separated stems will appear here. Drag them straight onto FL's Playlist.",
                    stemsListArea, juce::Justification::centred);
    }

    // Footer separator
    g.setColour (SC::outline);
    g.fillRect (0, getHeight() - 56, getWidth(), 1);
}

void StemSplitterEditor::resized()
{
    auto area = getLocalBounds();

    // Header
    auto header = area.removeFromTop (58).reduced (20, 14);
    modelButton.setBounds (header.removeFromRight (84));
    header.removeFromRight (8);
    gpuToggle.setBounds (header.removeFromRight (62));
    header.removeFromRight (6);
    qualityBox.setBounds (header.removeFromRight (150));
    header.removeFromRight (8);
    modelLabel.setBounds (header.withTrimmedLeft (180));

    area.reduce (20, 0);

    // Drop zone + Separate
    auto top = area.removeFromTop (110);
    auto buttons = top.removeFromRight (150).reduced (0, 30);
    separateButton.setBounds (buttons);
    cancelButton.setBounds (buttons);
    top.removeFromRight (12);
    dropZone.setBounds (top);

    area.removeFromTop (14);
    progressBar.setBounds (area.removeFromTop (10));
    area.removeFromTop (4);
    statusLabel.setBounds (area.removeFromTop (22).withTrimmedLeft (-4));

    // Stems header
    auto stemsHeader = area.removeFromTop (38).reduced (0, 4);
    stemsTitleArea = stemsHeader;
    openFolderButton.setBounds (stemsHeader.removeFromRight (104));
    stemsHeader.removeFromRight (8);
    dragAllButton.setBounds (stemsHeader.removeFromRight (180));

    // Footer
    auto footer = getLocalBounds().removeFromBottom (56).reduced (20, 12);
    playButton.setBounds (footer.removeFromLeft (90));
    footer.removeFromLeft (16);
    syncToggle.setBounds (footer.removeFromLeft (190));
    passThroughToggle.setBounds (footer.removeFromLeft (120));
    timeLabel.setBounds (footer);

    // Stem rows share what's left
    area.removeFromTop (4);
    area.removeFromBottom (64);
    stemsListArea = area;
    const int rowHeight = stemRows.isEmpty() ? 0 : juce::jmin (52, area.getHeight() / stemRows.size());

    for (auto* row : stemRows)
        row->setBounds (area.removeFromTop (rowHeight));
}

void StemSplitterEditor::loadFile (const juce::File& file)
{
    if (processor.getWorker().isBusy())
        return;

    processor.setInputFile (file);

    juce::String details;
    if (std::unique_ptr<juce::AudioFormatReader> reader { formats.createReaderFor (file) })
    {
        const double seconds = (double) reader->lengthInSamples / reader->sampleRate;
        details << formatTime (seconds) << "   " << juce::String (reader->sampleRate / 1000.0, 1) << " kHz   "
                << (reader->numChannels == 1 ? "mono" : "stereo") << "   " << file.getParentDirectory().getFullPathName();
    }
    else
    {
        details = "Can't read this file";
    }

    dropZone.setFileInfo (file.getFileName(), details);
}

void StemSplitterEditor::updateModelLabel()
{
    auto model = processor.getModelFile();

    if (model.existsAsFile())
    {
        modelLabel.setText ("Model: " + model.getFileNameWithoutExtension(), juce::dontSendNotification);
        modelLabel.setColour (juce::Label::textColourId, SC::textDim);
    }
    else
    {
        modelLabel.setText ("No model loaded", juce::dontSendNotification);
        modelLabel.setColour (juce::Label::textColourId, SC::danger);
    }
}

void StemSplitterEditor::showModelMenu()
{
    juce::PopupMenu menu;
    auto models = ModelLocator::findModels();
    const auto current = processor.getModelFile();

    for (int i = 0; i < models.size(); ++i)
        menu.addItem (100 + i, models[i].getFileName(), true, models[i] == current);

    if (! models.isEmpty())
        menu.addSeparator();

    menu.addItem (1, "Choose model file...");
    menu.addItem (2, "Open models folder");
    menu.addItem (3, "How do I get the model?");

    menu.showMenuAsync (juce::PopupMenu::Options().withTargetComponent (&modelButton),
                        [this, models] (int result)
    {
        if (result >= 100 && result - 100 < models.size())
        {
            processor.setModelFile (models[result - 100]);
            updateModelLabel();
        }
        else if (result == 1)
        {
            modelChooser = std::make_unique<juce::FileChooser> ("Choose an exported HTDemucs model",
                                                                ModelLocator::getUserModelFolder(), "*.onnx");
            modelChooser->launchAsync (juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                                       [this] (const juce::FileChooser& fc)
                                       {
                                           if (fc.getResult().existsAsFile())
                                           {
                                               processor.setModelFile (fc.getResult());
                                               updateModelLabel();
                                           }
                                       });
        }
        else if (result == 2)
        {
            auto folder = ModelLocator::getUserModelFolder();
            folder.createDirectory();
            folder.startAsProcess();
        }
        else if (result == 3)
        {
            showModelHelp();
        }
    });
}

void StemSplitterEditor::showModelHelp()
{
    auto folder = ModelLocator::getUserModelFolder();

    juce::AlertWindow::showMessageBoxAsync (
        juce::MessageBoxIconType::InfoIcon, "Getting the AI model",
        "Stem Splitter uses Meta's HTDemucs model, converted to ONNX format (about 170 MB).\n\n"
        "1. Install Python 3.10 or newer.\n"
        "2. In the plugin's source folder, run:\n"
        "     python tools/export_model.py\n"
        "   (add --six-stems for the 6-stem model with guitar and piano)\n"
        "3. The script saves htdemucs.onnx to:\n     " + folder.getFullPathName() + "\n\n"
        "Then reopen this window or pick the file with \"Model...\".",
        "OK", this);
}

void StemSplitterEditor::rebuildStemRows()
{
    stemRows.clear();

    if (auto stems = processor.getStems())
        for (int i = 0; i < stems->names.size(); ++i)
            addAndMakeVisible (stemRows.add (new StemRow (processor, i)));

    resized();
    repaint();
}

void StemSplitterEditor::timerCallback()
{
    auto& worker = processor.getWorker();
    const bool busy = worker.isBusy();
    const auto stage = worker.getStage();

    separateButton.setVisible (! busy);
    cancelButton.setVisible (busy);
    separateButton.setEnabled (processor.getInputFile().existsAsFile());
    dropZone.setEnabledForDrops (! busy);
    qualityBox.setEnabled (! busy);
    gpuToggle.setEnabled (! busy);
    modelButton.setEnabled (! busy);

    progressValue = (busy || stage == SeparationWorker::Stage::finished) ? worker.getProgress() : 0.0;

    auto message = worker.getMessage();
    const bool isError = stage == SeparationWorker::Stage::failed;

    if (message.isNotEmpty() || busy)
    {
        if (stage == SeparationWorker::Stage::finished && worker.getDeviceDescription().isNotEmpty())
            message << "   [" << worker.getDeviceDescription() << "]";

        statusLabel.setText (message, juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, isError ? SC::danger : SC::textDim);
    }
    else if (! processor.getModelFile().existsAsFile())
    {
        statusLabel.setText ("No AI model found yet - click \"Model...\" for setup instructions.", juce::dontSendNotification);
        statusLabel.setColour (juce::Label::textColourId, SC::danger);
    }

    if (processor.getStemsVersion() != shownStemsVersion)
    {
        shownStemsVersion = processor.getStemsVersion();
        rebuildStemRows();
    }

    auto stems = processor.getStems();
    const bool haveStems = stems != nullptr && ! stems->files.isEmpty();
    dragAllButton.setEnabled (haveStems);
    openFolderButton.setEnabled (haveStems);
    playButton.setEnabled (haveStems && ! processor.getSyncToHost());
    playButton.setButtonText (processor.isPlaying() ? "Stop" : "Play");

    if (haveStems && stems->previewRate > 0)
    {
        const double total = stems->previewLength / stems->previewRate;
        timeLabel.setText (formatTime (processor.getPlayPosition() * total) + " / " + formatTime (total),
                           juce::dontSendNotification);
    }
    else
    {
        timeLabel.setText ({}, juce::dontSendNotification);
    }

    for (auto* row : stemRows)
    {
        row->refreshButtons();
        row->repaint();
    }
}
