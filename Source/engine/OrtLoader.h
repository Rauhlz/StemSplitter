#pragma once

// ONNX Runtime is loaded at runtime from the plugin's own folder instead of being
// linked. Windows resolves a plugin DLL's dependencies from the host's folder
// (next to FL64.exe), so a normal link-time dependency on onnxruntime.dll would fail
// to load, and loading by full path avoids clashing with any other copy already
// loaded in the host.
//
// ORT_API_MANUAL_INIT is defined for every translation unit by CMake, so the ONNX
// Runtime C++ wrapper doesn't touch the library until OrtLoader::ensureLoaded()
// has handed it an API table.

#include <juce_core/juce_core.h>

namespace stemsplitter
{
struct OrtLoader
{
    // Loads the ONNX Runtime shared library (once per process) and initialises the
    // C++ API. Returns false and fills `error` if the library can't be found or is
    // too old.
    static bool ensureLoaded (juce::String& error);

    // Path of the library that was loaded, for the About/status text.
    static juce::File getLoadedLibraryPath();

    // Looks up an exported C function in the loaded library (e.g. an execution-provider factory).
    static void* getSymbol (const char* name);

    // Folders searched for the library, in order.
    static juce::Array<juce::File> getSearchFolders();
};
} // namespace stemsplitter
