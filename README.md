# Stem Splitter: AI stem separation inside FL Studio

A VST3 plugin that splits a song into **drums, bass, other and vocals** (or 6 stems with
**guitar and piano**) using Meta's HTDemucs, the same kind of model behind FL Studio's
built-in stem separator. Everything runs locally on your computer through ONNX Runtime.

- Drop a song on the plugin and press **Separate**
- Preview the stems inside the plugin with mute/solo, or in sync with FL's transport
- **Drag a stem (or all of them) straight onto FL's Playlist**
- Stems are saved as 32-bit float WAVs at the song's original sample rate and length, in
  `<song folder>/<song name> Stems/`
- An extra **Instrumental** stem (everything except vocals) is written too
- Separation runs on a background thread, so FL keeps playing, and you can cancel it at any time
- Optional GPU acceleration: DirectML on Windows (any DX12 GPU), CoreML on Mac

Separation is offline (it isn't real-time), like FL's own separator. A 3–4 minute song
takes roughly 1–3 minutes on a modern CPU, or well under a minute on a GPU.

---

## 1. Get the plugin

### Option A: let GitHub build it (no compilers needed)
1. Create a GitHub repository and push this folder to it.
2. Open the **Actions** tab. The **Build** workflow runs automatically and produces:
   - `StemSplitter-windows-x64-directml`: the plugin for Windows with GPU support (recommended; also works on CPU)
   - `StemSplitter-windows-x64`: the plugin for Windows, CPU only
   
   Both AI models are already **inside** these plugin downloads, so you can skip step 2. The
   plugin uses the 6-stem model by default; switch with **Model...**.
3. Download the artifacts you need from the finished run.

### Option B: build it yourself
You need **CMake 3.22+**, **Git**, and a C++ compiler:
- Windows: Visual Studio 2022 with the *Desktop development with C++* workload
- macOS: Xcode 14+

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release           # add -DSTEMSPLITTER_DIRECTML=ON for GPU on Windows
cmake --build build --config Release --parallel
```
JUCE and ONNX Runtime are downloaded automatically during the first configure.

The plugin ends up in `build/StemSplitter_artefacts/Release/VST3/Stem Splitter.vst3`. It's a
folder, and `onnxruntime.dll` is already inside it.

## 2. Get the AI model (one time)

The model weights aren't included in this repository. Convert Meta's pretrained model with
Python 3.10+ (or download the `model-*` artifact from GitHub Actions, as above):

```bash
python -m pip install -r tools/requirements.txt
python -m pip install --no-deps -r tools/requirements-nodeps.txt
python tools/export_model.py                  # 4 stems  -> htdemucs.onnx    (~170 MB)
python tools/export_model.py --six-stems      # 6 stems  -> htdemucs_6s.onnx (adds guitar + piano)
```

The script downloads the official weights, exports them, and **verifies** that the ONNX
model plus the plugin's spectrogram maths reproduce the original PyTorch output. It saves
the model where the plugin looks for it automatically:

| OS | Model folder |
|---|---|
| Windows | `%APPDATA%\StemSplitter\Models\` |
| macOS | `~/Library/Application Support/StemSplitter/Models/` |

If you put the `.onnx` file somewhere else, pick it in the plugin with **Model... → Choose model file...**.
When several models are present, the plugin uses `htdemucs_6s` first, then `htdemucs`. Your choice
in **Model...** is remembered. (Meta notes the 6-stem model's piano stem is its weakest; the other stems
are about as good as with the 4-stem model.)

## 3. Install in FL Studio

**Windows:** copy the whole `Stem Splitter.vst3` folder to `C:\Program Files\Common Files\VST3\`.
**macOS:** copy it to `~/Library/Audio/Plug-Ins/VST3/`.

In FL Studio, open **Options → Manage plugins → Find installed plugins**, then add **Stem Splitter**
to any Mixer insert slot. It's an effect: the track's own audio passes through, and the stem
preview plays on that insert.

## 4. Use it

1. Drag an audio file (WAV, MP3, FLAC, OGG or AIFF) from Explorer/Finder onto the plugin, or click
   the drop area to browse.
2. Pick the quality, then press **Separate**:
   - *Standard*: one pass
   - *High*: 2 time-shifted passes averaged together, about 2x slower
   - *Best*: 4 passes, about 4x slower
3. When it's done:
   - **Drag a stem by its name** onto the Playlist, the Channel rack or Edison
   - **Drag all to Playlist** drags every stem at once (Instrumental is left out, since it duplicates the others)
   - **Play** previews the stems. **M**/**S** mute and solo, and clicking a waveform jumps to that point
   - Turn on **Sync to FL transport** to hear the stems follow FL's song position (the song starts at bar 1)
   - **Open folder** shows the files

The plugin remembers the stems with your FL project and reloads them when you reopen it.

---

## How it works

HTDemucs is a hybrid model: a time-domain U-Net and a spectrogram U-Net joined by a
transformer. PyTorch's STFT can't be exported to ONNX, so `tools/export_model.py` exports
the network **without** its STFT/iSTFT (the approach from
[sevagh/demucs.onnx](https://github.com/sevagh/demucs.onnx)), and the plugin recreates, in C++,
everything Demucs does around the network:

| Step | Where |
|---|---|
| Decode audio, convert to stereo, resample to 44.1 kHz (windowed-sinc) | `Source/engine/AudioUtils.cpp` |
| Normalise, random time shift, 7.8 s segments with 25 % overlap and triangular cross-fades | `Source/engine/DemucsEngine.cpp` (`separate`) |
| Reflect padding and STFT (4096 / hop 1024, complex-as-channels) | `DemucsSpectrogram::forward` |
| Network inference | ONNX Runtime (`DemucsModel::run`) |
| iSTFT of the frequency branch, plus the time branch | `DemucsSpectrogram::inverse` |
| Resample back, write WAVs, build the preview | `Source/SeparationWorker.cpp` |

ONNX Runtime is loaded at runtime from inside the plugin bundle (`Source/engine/OrtLoader.cpp`)
instead of being linked. Windows would otherwise look for `onnxruntime.dll` next to `FL64.exe`
rather than next to the plugin, and loading by full path also avoids clashing with any other
copy of ONNX Runtime that's already loaded in FL.

### Tests

The tests don't need the real weights. They use a stand-in ONNX model with the same inputs and
outputs.

```bash
cmake -B build -DSTEMSPLITTER_BUILD_TESTS=ON && cmake --build build
python tests/run_tests.py build/StemSplitterCli_artefacts/Release/stemsplitter-cli
```

- The STFT and iSTFT match a NumPy re-implementation of `HTDemucs._spec`/`_ispec` (to about 1e-7).
- The full pipeline matches a NumPy port of Demucs' `apply_model` (to about 5e-7).
- The pipeline preserves length, sample rate and levels at 44.1 and 48 kHz, and for files shorter than one segment.
- `PluginHarness` checks the separation job, preview mixing, mute/solo, cancellation, saving and
  restoring projects, host sample-rate changes, and loading the actual `.vst3` through a VST3 host.

### Command-line tool

`stemsplitter-cli --model htdemucs.onnx song.wav` runs the same engine outside FL. It's useful for
checking a model, or for batch jobs.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No AI model found" | Run `tools/export_model.py`, or pick the `.onnx` file with **Model...** |
| "Couldn't find the ONNX Runtime library" | Copy the *whole* `Stem Splitter.vst3` folder, not only the file inside it |
| GPU box ticked but status says CPU | Use the `-directml` build on Windows. Older GPUs without DX12 fall back to CPU |
| Dropping a file does nothing | Only audio files are accepted. You can click the drop area to browse instead |
| macOS says the plugin is damaged | It's unsigned: run `xattr -dr com.apple.quarantine ~/Library/Audio/Plug-Ins/VST3/"Stem Splitter.vst3"` |

## Licence

This project's code is MIT-licensed. It builds on Demucs (MIT), demucs.onnx (MIT), ONNX Runtime
(MIT) and JUCE (AGPLv3/commercial). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
distributing binaries.
