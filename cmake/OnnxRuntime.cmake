# Downloads a prebuilt ONNX Runtime for the current platform (or uses ORT_ROOT) and sets:
#   ORT_INCLUDE_DIRS   - headers
#   ORT_RUNTIME_FILES  - list of "source;dest-name" pairs to copy next to the plugin binary
#
# We only need headers at build time: the library itself is loaded at runtime
# (see Source/engine/OrtLoader.cpp), so nothing is linked.

set(ORT_VERSION "1.20.1" CACHE STRING "ONNX Runtime version to download")
set(ORT_ROOT "" CACHE PATH "Use an existing ONNX Runtime folder (with include/ and lib/) instead of downloading")
set(DIRECTML_VERSION "1.15.2" CACHE STRING "DirectML redistributable version (Windows GPU builds)")

set(_ort_dl "${CMAKE_BINARY_DIR}/_deps/onnxruntime")

function(_ort_fetch url archive_name out_dir)
    if(NOT EXISTS "${out_dir}/.done")
        message(STATUS "Downloading ${url}")
        file(DOWNLOAD "${url}" "${_ort_dl}/${archive_name}" SHOW_PROGRESS STATUS _st TLS_VERIFY ON)
        list(GET _st 0 _code)
        if(NOT _code EQUAL 0)
            message(FATAL_ERROR "Download failed: ${url} (${_st}). Download it manually and pass -DORT_ROOT=...")
        endif()
        file(MAKE_DIRECTORY "${out_dir}")
        file(ARCHIVE_EXTRACT INPUT "${_ort_dl}/${archive_name}" DESTINATION "${out_dir}")
        file(WRITE "${out_dir}/.done" "")
    endif()
endfunction()

set(ORT_RUNTIME_FILES "")

if(ORT_ROOT)
    set(ORT_INCLUDE_DIRS "${ORT_ROOT}/include")
    if(WIN32)
        list(APPEND ORT_RUNTIME_FILES "${ORT_ROOT}/lib/onnxruntime.dll;onnxruntime.dll")
    elseif(APPLE)
        list(APPEND ORT_RUNTIME_FILES "${ORT_ROOT}/lib/libonnxruntime.dylib;libonnxruntime.dylib")
    else()
        list(APPEND ORT_RUNTIME_FILES "${ORT_ROOT}/lib/libonnxruntime.so;libonnxruntime.so")
    endif()

elseif(WIN32 AND STEMSPLITTER_DIRECTML)
    # NuGet packages are plain zip files
    set(_dir "${_ort_dl}/ort-dml-${ORT_VERSION}")
    _ort_fetch("https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/${ORT_VERSION}"
               "ort-dml-${ORT_VERSION}.zip" "${_dir}")
    set(_dml "${_ort_dl}/directml-${DIRECTML_VERSION}")
    _ort_fetch("https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/${DIRECTML_VERSION}"
               "directml-${DIRECTML_VERSION}.zip" "${_dml}")

    set(ORT_INCLUDE_DIRS "${_dir}/build/native/include")
    list(APPEND ORT_RUNTIME_FILES "${_dir}/runtimes/win-x64/native/onnxruntime.dll;onnxruntime.dll")
    list(APPEND ORT_RUNTIME_FILES "${_dml}/bin/x64-win/DirectML.dll;DirectML.dll")

elseif(WIN32)
    set(_name "onnxruntime-win-x64-${ORT_VERSION}")
    _ort_fetch("https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_name}.zip"
               "${_name}.zip" "${_ort_dl}")
    set(ORT_INCLUDE_DIRS "${_ort_dl}/${_name}/include")
    list(APPEND ORT_RUNTIME_FILES "${_ort_dl}/${_name}/lib/onnxruntime.dll;onnxruntime.dll")

elseif(APPLE)
    set(_name "onnxruntime-osx-universal2-${ORT_VERSION}")
    _ort_fetch("https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_name}.tgz"
               "${_name}.tgz" "${_ort_dl}")
    set(ORT_INCLUDE_DIRS "${_ort_dl}/${_name}/include")
    list(APPEND ORT_RUNTIME_FILES "${_ort_dl}/${_name}/lib/libonnxruntime.${ORT_VERSION}.dylib;libonnxruntime.dylib")

else()
    set(_name "onnxruntime-linux-x64-${ORT_VERSION}")
    _ort_fetch("https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_name}.tgz"
               "${_name}.tgz" "${_ort_dl}")
    set(ORT_INCLUDE_DIRS "${_ort_dl}/${_name}/include")
    list(APPEND ORT_RUNTIME_FILES "${_ort_dl}/${_name}/lib/libonnxruntime.so.${ORT_VERSION};libonnxruntime.so")
endif()

# Copies the ONNX Runtime libraries next to a target's binary after each build.
function(stemsplitter_copy_ort target)
    # Entries are "src;name" pairs, which CMake flattens into one list: walk it two at a time.
    list(LENGTH ORT_RUNTIME_FILES _n)
    math(EXPR _last "${_n} - 1")
    foreach(_i RANGE 0 ${_last} 2)
        math(EXPR _j "${_i} + 1")
        list(GET ORT_RUNTIME_FILES ${_i} _src)
        list(GET ORT_RUNTIME_FILES ${_j} _dst)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_src}" "$<TARGET_FILE_DIR:${target}>/${_dst}"
            VERBATIM)
    endforeach()
endfunction()
