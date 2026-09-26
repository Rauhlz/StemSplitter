cmake_minimum_required(VERSION 3.22)

project(StemSplitter VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")

if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
endif()

option(STEMSPLITTER_DIRECTML "Windows: bundle the DirectML build of ONNX Runtime for GPU separation" OFF)
option(STEMSPLITTER_BUILD_PLUGIN "Build the VST3 plugin" ON)
option(STEMSPLITTER_BUILD_CLI "Build the command-line tool" ON)
option(STEMSPLITTER_BUILD_TESTS "Build the plugin test harness" OFF)
set(JUCE_DIR "" CACHE PATH "Path to a local JUCE checkout (downloaded automatically if empty)")

# ---------------------------------------------------------------- JUCE
if(JUCE_DIR)
    add_subdirectory("${JUCE_DIR}" JUCE)
else()
    include(FetchContent)
    FetchContent_Declare(JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.4
        GIT_SHALLOW    ON)
    FetchContent_MakeAvailable(JUCE)
endif()

# ---------------------------------------------------------------- ONNX Runtime
include(cmake/OnnxRuntime.cmake)

set(ENGINE_SOURCES
    Source/engine/DemucsEngine.cpp
    Source/engine/DemucsEngine.h
    Source/engine/OrtLoader.cpp
    Source/engine/OrtLoader.h
    Source/engine/AudioUtils.cpp
    Source/engine/AudioUtils.h)

set(COMMON_DEFINITIONS
    ORT_API_MANUAL_INIT=1                          # ONNX Runtime is loaded at runtime, see OrtLoader.cpp
    STEMSPLITTER_DIRECTML=$<BOOL:${STEMSPLITTER_DIRECTML}>
    JUCE_USE_MP3AUDIOFORMAT=1
    JUCE_USE_FLAC=1
    JUCE_USE_OGGVORBIS=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

if(MSVC)
    set(COMMON_OPTIONS /W4 /wd4244 /wd4267 /MP)
else()
    set(COMMON_OPTIONS -Wall -Wextra -Wno-sign-compare)
endif()

# ---------------------------------------------------------------- Plugin
if(STEMSPLITTER_BUILD_PLUGIN)
    juce_add_plugin(StemSplitter
        COMPANY_NAME "StemSplitter"
        BUNDLE_ID "com.stemsplitter.plugin"
        PLUGIN_MANUFACTURER_CODE Stsp
        PLUGIN_CODE Stm1
        FORMATS VST3
        PRODUCT_NAME "Stem Splitter"
        DESCRIPTION "AI stem separation (HTDemucs)"
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        EDITOR_WANTS_KEYBOARD_FOCUS FALSE
        COPY_PLUGIN_AFTER_BUILD FALSE
        VST3_CATEGORIES Fx Tools)

    target_sources(StemSplitter PRIVATE
        ${ENGINE_SOURCES}
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
        Source/PluginEditor.cpp
        Source/PluginEditor.h
        Source/SeparationWorker.cpp
        Source/SeparationWorker.h
        Source/ModelLocator.cpp
        Source/ModelLocator.h
        Source/LookAndFeel.h)

    target_include_directories(StemSplitter SYSTEM PRIVATE ${ORT_INCLUDE_DIRS})
    target_include_directories(StemSplitter PRIVATE Source)

    target_compile_definitions(StemSplitter PUBLIC
        ${COMMON_DEFINITIONS}
        JUCE_VST3_CAN_REPLACE_VST2=0)

    target_compile_options(StemSplitter PRIVATE ${COMMON_OPTIONS})

    target_link_libraries(StemSplitter
        PRIVATE
            juce::juce_audio_processors
            juce::juce_audio_formats
            juce::juce_dsp
            juce::juce_gui_extra
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_lto_flags
            juce::juce_recommended_warning_flags)

    # Put onnxruntime next to the plugin binary inside the .vst3 bundle
    stemsplitter_copy_ort(StemSplitter_VST3)

    if(APPLE)
        # Re-sign the bundle after adding the dylib (ad-hoc; use your Developer ID to distribute)
        add_custom_command(TARGET StemSplitter_VST3 POST_BUILD
            COMMAND codesign --force --deep --sign - "$<TARGET_BUNDLE_DIR:StemSplitter_VST3>"
            VERBATIM)
    endif()
endif()

# ---------------------------------------------------------------- CLI
if(STEMSPLITTER_BUILD_CLI)
    juce_add_console_app(StemSplitterCli PRODUCT_NAME "stemsplitter-cli")

    target_sources(StemSplitterCli PRIVATE ${ENGINE_SOURCES} Source/cli/Main.cpp)
    target_include_directories(StemSplitterCli SYSTEM PRIVATE ${ORT_INCLUDE_DIRS})
    target_include_directories(StemSplitterCli PRIVATE Source)
    set_target_properties(StemSplitterCli PROPERTIES OUTPUT_NAME "stemsplitter-cli")
    target_compile_definitions(StemSplitterCli PRIVATE ${COMMON_DEFINITIONS})
    target_compile_options(StemSplitterCli PRIVATE ${COMMON_OPTIONS})

    target_link_libraries(StemSplitterCli
        PRIVATE
            juce::juce_audio_formats
            juce::juce_dsp
        PUBLIC
            juce::juce_recommended_config_flags
            juce::juce_recommended_warning_flags)

    if(NOT WIN32 AND NOT APPLE)
        target_link_libraries(StemSplitterCli PRIVATE dl)
    endif()

    stemsplitter_copy_ort(StemSplitterCli)
endif()

# ---------------------------------------------------------------- Tests
if(STEMSPLITTER_BUILD_TESTS)
    juce_add_console_app(PluginHarness PRODUCT_NAME "plugin-harness")
    target_sources(PluginHarness PRIVATE
        ${ENGINE_SOURCES}
        tests/PluginHarness.cpp
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/SeparationWorker.cpp
        Source/ModelLocator.cpp)
    target_include_directories(PluginHarness SYSTEM PRIVATE ${ORT_INCLUDE_DIRS})
    target_include_directories(PluginHarness PRIVATE Source)
    target_compile_definitions(PluginHarness PRIVATE ${COMMON_DEFINITIONS} JucePlugin_Build_Standalone=0
        JUCE_PLUGINHOST_VST3=1)
    target_link_libraries(PluginHarness PRIVATE
        juce::juce_audio_processors juce::juce_audio_formats juce::juce_dsp juce::juce_gui_extra
        PUBLIC juce::juce_recommended_config_flags)
    stemsplitter_copy_ort(PluginHarness)
endif()
