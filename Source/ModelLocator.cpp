#include "ModelLocator.h"

namespace stemsplitter
{
SharedSettings::SharedSettings()
{
    juce::PropertiesFile::Options options;
    options.applicationName = "StemSplitter";
    options.filenameSuffix = "settings";
    options.folderName = "StemSplitter";
    options.osxLibrarySubFolder = "Application Support";
    options.processLock = &processLock;   // several plugin instances may share it
    file = std::make_unique<juce::PropertiesFile> (options);
}

juce::PropertiesFile& ModelLocator::getSettings()
{
    // Callers (the processor) keep a SharedResourcePointer alive, so this is the shared instance.
    juce::SharedResourcePointer<SharedSettings> holder;
    return *holder->file;
}

juce::File ModelLocator::getUserModelFolder()
{
   #if JUCE_MAC
    return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
               .getChildFile ("Application Support/StemSplitter/Models");
   #else
    return juce::File::getSpecialLocation (juce::File::userApplicationDataDirectory)
               .getChildFile ("StemSplitter/Models");
   #endif
}

juce::Array<juce::File> ModelLocator::getSearchFolders()
{
    juce::Array<juce::File> folders;
    folders.add (getUserModelFolder());

    auto binaryDir = juce::File::getSpecialLocation (juce::File::currentExecutableFile).getParentDirectory();
    folders.add (binaryDir);
    folders.add (binaryDir.getSiblingFile ("Resources"));   // inside the .vst3 bundle

   #if JUCE_WINDOWS
    folders.add (juce::File::getSpecialLocation (juce::File::commonApplicationDataDirectory)
                     .getChildFile ("StemSplitter/Models"));
   #endif

    return folders;
}

juce::Array<juce::File> ModelLocator::findModels()
{
    juce::Array<juce::File> models;

    for (auto& folder : getSearchFolders())
        if (folder.isDirectory())
            for (auto& f : folder.findChildFiles (juce::File::findFiles, false, "*.onnx"))
                models.addIfNotAlreadyThere (f);

    return models;
}

juce::File ModelLocator::findDefaultModel()
{
    auto saved = getSettings().getValue ("modelPath");

    if (saved.isNotEmpty() && juce::File::isAbsolutePath (saved) && juce::File (saved).existsAsFile())
        return juce::File (saved);

    auto models = findModels();

    // 6 stems (adds guitar and piano) by default, then the 4-stem model
    for (auto* preferred : { "htdemucs_6s", "htdemucs" })
        for (auto& m : models)
            if (m.getFileNameWithoutExtension().equalsIgnoreCase (preferred))
                return m;

    return models.isEmpty() ? juce::File() : models.getFirst();
}

void ModelLocator::setPreferredModel (const juce::File& file)
{
    getSettings().setValue ("modelPath", file.getFullPathName());
    getSettings().saveIfNeeded();
}
} // namespace stemsplitter
