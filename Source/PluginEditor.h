#pragma once

#include "PluginProcessor.h"
#include "LookAndFeel.h"

//==============================================================================
// Big "drop a song here" area. Also accepts clicks to open a file browser.
class DropZone : public juce::Component,
                 public juce::FileDragAndDropTarget
{
public:
    std::function<void (const juce::File&)> onFileChosen;

    void setFileInfo (const juce::String& title, const juce::String& details);
    void setEnabledForDrops (bool shouldAccept) { acceptDrops = shouldAccept; repaint(); }

    void paint (juce::Graphics&) override;
    void mouseUp (const juce::MouseEvent&) override;

    bool isInterestedInFileDrag (const juce::StringArray& files) override;
    void fileDragEnter (const juce::StringArray&, int, int) override { dragOver = true;  repaint(); }
    void fileDragExit (const juce::StringArray&) override              { dragOver = false; repaint(); }
    void filesDropped (const juce::StringArray& files, int, int) override;

    static bool isAudioFile (const juce::File& f);

private:
    juce::String title, details;
    bool dragOver = false, acceptDrops = true;
    std::unique_ptr<juce::FileChooser> chooser;
};

//==============================================================================
// One separated stem: name, waveform, mute/solo, and drag-to-Playlist.
class StemRow : public juce::Component
{
public:
    StemRow (StemSplitterProcessor& p, int index);

    void paint (juce::Graphics&) override;
    void resized() override;
    void mouseDown (const juce::MouseEvent&) override;
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override;
    void mouseMove (const juce::MouseEvent&) override;

    void refreshButtons();

private:
    juce::Rectangle<int> getWaveArea() const;
    juce::Rectangle<int> getHandleArea() const;

    StemSplitterProcessor& processor;
    const int index;
    juce::TextButton muteButton { "M" }, soloButton { "S" };
    bool dragging = false, seeking = false;
};

//==============================================================================
// Looks like a button, but you drag it: starts an OS file drag so the files can be
// dropped straight onto FL's Playlist (or the Channel rack, Edison, Explorer...).
class FileDragButton : public juce::Component
{
public:
    explicit FileDragButton (const juce::String& text) : label (text) {}

    std::function<juce::StringArray()> getFiles;

    void paint (juce::Graphics&) override;
    void mouseEnter (const juce::MouseEvent&) override { repaint(); }
    void mouseExit (const juce::MouseEvent&) override  { repaint(); }
    void mouseDrag (const juce::MouseEvent&) override;
    void mouseUp (const juce::MouseEvent&) override    { dragging = false; }

private:
    juce::String label;
    bool dragging = false;
};

//==============================================================================
class StemSplitterEditor : public juce::AudioProcessorEditor,
                           private juce::Timer
{
public:
    explicit StemSplitterEditor (StemSplitterProcessor&);
    ~StemSplitterEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;
    void loadFile (const juce::File& file);
    void showModelMenu();
    void showModelHelp();
    void rebuildStemRows();
    void updateModelLabel();

    StemSplitterProcessor& processor;
    stemsplitter::StemLookAndFeel lookAndFeel;
    juce::AudioFormatManager formats;

    DropZone dropZone;
    juce::TextButton separateButton { "Separate" }, cancelButton { "Cancel" };
    juce::TextButton modelButton { "Model..." };
    juce::ComboBox qualityBox;
    juce::ToggleButton gpuToggle { "GPU" };

    double progressValue = 0.0;
    juce::ProgressBar progressBar { progressValue };
    juce::Label statusLabel, modelLabel;

    FileDragButton dragAllButton { "Drag all to Playlist" };
    juce::TextButton openFolderButton { "Open folder" };
    juce::OwnedArray<StemRow> stemRows;
    int shownStemsVersion = -1;
    juce::Rectangle<int> stemsTitleArea, stemsListArea;

    juce::TextButton playButton { "Play" };
    juce::ToggleButton syncToggle { "Sync to FL transport" }, passThroughToggle { "Hear input" };
    juce::Label timeLabel;

    std::unique_ptr<juce::FileChooser> modelChooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemSplitterEditor)
};
