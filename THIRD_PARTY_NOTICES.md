# Third-party software

| Component | Use | Licence |
|---|---|---|
| [Demucs v4 / HTDemucs](https://github.com/facebookresearch/demucs) (Meta) | The separation model and its pretrained weights; the segmenting, normalisation and spectrogram logic is ported from `demucs/apply.py` and `demucs/htdemucs.py` | MIT |
| [demucs.onnx](https://github.com/sevagh/demucs.onnx) (Sevag Hanssian) | The idea of exporting HTDemucs without its STFT, and the replacement `forward()` used in `tools/export_model.py` | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (Microsoft) | Runs the model; shipped as `onnxruntime.dll` / `.dylib` inside the plugin bundle | MIT |
| [DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML) (Microsoft) | Optional GPU build on Windows | Microsoft DirectML licence (redistributable) |
| [JUCE](https://juce.com) | Plugin framework, GUI, audio file reading/writing | AGPLv3 or a commercial JUCE licence |

**JUCE licensing:** building and using this plugin yourself is fine. If you *distribute*
the compiled plugin, you must either release it under the AGPLv3 (this repository's
MIT code is compatible) or hold a JUCE licence (the free "Personal"/"Starter" tier
covers small creators). See https://juce.com/get-juce.
