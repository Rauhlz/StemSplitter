#include "OrtLoader.h"

#include <onnxruntime_cxx_api.h>
#include <mutex>

#if JUCE_WINDOWS
 #ifndef NOMINMAX
  #define NOMINMAX
 #endif
 #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
 #endif
 #include <windows.h>
#else
 #include <dlfcn.h>
#endif

namespace stemsplitter
{
namespace
{
std::mutex loadMutex;
bool loaded = false;
juce::String lastError;
juce::File loadedPath;
void* libraryHandle = nullptr;

juce::StringArray libraryNames()
{
   #if JUCE_WINDOWS
    return { "onnxruntime.dll" };
   #elif JUCE_MAC
    return { "libonnxruntime.dylib" };
   #else
    return { "libonnxruntime.so", "libonnxruntime.so.1" };
   #endif
}

using GetApiBaseFn = const OrtApiBase* (ORT_API_CALL*) ();

void* findSymbol (void* handle, const char* name)
{
   #if JUCE_WINDOWS
    return reinterpret_cast<void*> (GetProcAddress (static_cast<HMODULE> (handle), name));
   #else
    return dlsym (handle, name);
   #endif
}

GetApiBaseFn openLibrary (const juce::File& file, void*& handleOut)
{
   #if JUCE_WINDOWS
    // LOAD_WITH_ALTERED_SEARCH_PATH lets onnxruntime.dll find DirectML.dll and
    // friends in its own folder rather than in the host's.
    void* handle = LoadLibraryExW (file.getFullPathName().toWideCharPointer(),
                                   nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
   #else
    void* handle = dlopen (file.getFullPathName().toRawUTF8(), RTLD_NOW | RTLD_LOCAL);
   #endif

    if (handle == nullptr)
        return nullptr;

    handleOut = handle;
    return reinterpret_cast<GetApiBaseFn> (findSymbol (handle, "OrtGetApiBase"));
}
} // namespace

juce::Array<juce::File> OrtLoader::getSearchFolders()
{
    juce::Array<juce::File> folders;

    // The plugin binary itself, e.g. StemSplitter.vst3/Contents/x86_64-win/
    auto binary = juce::File::getSpecialLocation (juce::File::currentExecutableFile);
    auto binaryDir = binary.getParentDirectory();
    folders.add (binaryDir);

   #if JUCE_MAC
    folders.add (binaryDir.getSiblingFile ("Frameworks"));
    folders.add (binaryDir.getSiblingFile ("Resources"));
   #endif

    // Optional override for development: STEMSPLITTER_ORT_DIR=/path/to/onnxruntime/lib
    auto envDir = juce::SystemStats::getEnvironmentVariable ("STEMSPLITTER_ORT_DIR", {});
    if (envDir.isNotEmpty())
        folders.insert (0, juce::File (envDir));

    return folders;
}

bool OrtLoader::ensureLoaded (juce::String& error)
{
    std::lock_guard<std::mutex> lock (loadMutex);

    if (loaded)
        return true;

    juce::StringArray tried;

    for (auto& folder : getSearchFolders())
    {
        for (auto& name : libraryNames())
        {
            auto file = folder.getChildFile (name);
            if (! file.existsAsFile())
                continue;

            tried.add (file.getFullPathName());
            void* handle = nullptr;
            auto getApiBase = openLibrary (file, handle);

            if (getApiBase == nullptr)
                continue;

            const OrtApi* api = getApiBase()->GetApi (ORT_API_VERSION);

            if (api == nullptr)
            {
                lastError = "ONNX Runtime at " + file.getFullPathName()
                          + " is too old (need API version " + juce::String (ORT_API_VERSION)
                          + ", found " + juce::String (getApiBase()->GetVersionString()) + ").";
                continue;
            }

            Ort::InitApi (api);
            loaded = true;
            loadedPath = file;
            libraryHandle = handle;
            return true;
        }
    }

    if (lastError.isEmpty())
    {
        lastError = "Couldn't find the ONNX Runtime library ("
                  + libraryNames()[0] + ") next to the plugin.";

        if (! tried.isEmpty())
            lastError << " Tried: " << tried.joinIntoString (", ");
    }

    error = lastError;
    return false;
}

void* OrtLoader::getSymbol (const char* name)
{
    std::lock_guard<std::mutex> lock (loadMutex);
    return libraryHandle != nullptr ? findSymbol (libraryHandle, name) : nullptr;
}

juce::File OrtLoader::getLoadedLibraryPath()
{
    std::lock_guard<std::mutex> lock (loadMutex);
    return loadedPath;
}
} // namespace stemsplitter
