#pragma once

#include <juce_data_structures/juce_data_structures.h>

namespace stemsplitter
{
// Per-user settings shared by every plugin instance. Each processor keeps a
// juce::SharedResourcePointer<SharedSettings> alive so the file stays loaded.
struct SharedSettings
{
    SharedSettings();
    juce::InterProcessLock processLock { "StemSplitterSettings" };
    std::unique_ptr<juce::PropertiesFile> file;
};

// Finds the exported HTDemucs .onnx file and stores global (per-user) settings.
struct ModelLocator
{
    // e.g. %APPDATA%\StemSplitter\Models  or  ~/Library/Application Support/StemSplitter/Models
    static juce::File getUserModelFolder();

    // All folders searched for *.onnx models, in priority order.
    static juce::Array<juce::File> getSearchFolders();

    static juce::Array<juce::File> findModels();

    // The user's chosen model if it still exists, otherwise htdemucs.onnx, otherwise any model.
    static juce::File findDefaultModel();

    static void setPreferredModel (const juce::File& file);

    // Shared settings file (model path, GPU choice...)
    static juce::PropertiesFile& getSettings();
};
} // namespace stemsplitter
