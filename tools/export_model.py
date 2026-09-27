#!/usr/bin/env python3
"""
Converts Meta's pretrained HTDemucs (Demucs v4) model to ONNX for the Stem Splitter plugin.

    python tools/export_model.py                # 4 stems: drums, bass, other, vocals
    python tools/export_model.py --six-stems    # 6 stems: + guitar, piano

The STFT/iSTFT can't be exported to ONNX, so we export the network *without* them
(the plugin does the STFT in C++). This follows sevagh/demucs.onnx (MIT): the
model's forward() is replaced by one that takes the waveform and its spectrogram
and returns the time- and frequency-branch outputs.

After exporting, the script checks that ONNX Runtime + the plugin's spectrogram
maths reproduce the original PyTorch model's output.

Setup (once):
    python -m pip install -r tools/requirements.txt
    python -m pip install --no-deps -r tools/requirements-nodeps.txt
"""
import argparse
import math
import os
import platform
import sys
import types
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). Run:\n"
             "  python -m pip install -r tools/requirements.txt\n"
             "  python -m pip install --no-deps -r tools/requirements-nodeps.txt")

# Demucs' checkpoints are pickles of Meta's own classes. PyTorch 2.6+ refuses those by
# default ("weights_only"), so allow them for these trusted, official files.
_torch_load = torch.load
def _load_trusted(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _load_trusted

try:
    from demucs.pretrained import get_model
    from demucs.htdemucs import HTDemucs
    from einops import rearrange
except ImportError as e:
    sys.exit(f"Missing dependency ({e}). Run:\n"
             "  python -m pip install -r tools/requirements.txt\n"
             "  python -m pip install --no-deps -r tools/requirements-nodeps.txt")


def default_model_folder() -> Path:
    """Where the plugin looks for models (matches ModelLocator.cpp)."""
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "StemSplitter" / "Models"
    if system == "Darwin":
        return Path.home() / "Library/Application Support/StemSplitter/Models"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "StemSplitter" / "Models"


# ----------------------------------------------------------------------------
# HTDemucs.forward without the STFT/iSTFT (from sevagh/demucs.onnx, MIT licence).
#   mix : [B, 2, T]              waveform (T = training length, 343980 samples)
#   x   : [B, 4, 2048, frames]   complex-as-channels spectrogram of `mix`
# returns
#   x   : [B, S, 4, 2048, frames] frequency-branch output (a CaC spectrogram)
#   xt  : [B, S, 2, T]           time-branch output
def core_forward(self, mix, x):
    training_length = int(self.segment * self.samplerate)
    B, C, Fq, T = x.shape

    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) / (1e-5 + std)

    xt = mix
    meant = xt.mean(dim=(1, 2), keepdim=True)
    stdt = xt.std(dim=(1, 2), keepdim=True)
    xt = (xt - meant) / (1e-5 + stdt)

    saved, saved_t, lengths, lengths_t = [], [], [], []
    for idx, encode in enumerate(self.encoder):
        lengths.append(x.shape[-1])
        inject = None
        if idx < len(self.tencoder):
            lengths_t.append(xt.shape[-1])
            tenc = self.tencoder[idx]
            xt = tenc(xt)
            if not tenc.empty:
                saved_t.append(xt)
            else:
                inject = xt
        x = encode(x, inject)
        if idx == 0 and self.freq_emb is not None:
            frs = torch.arange(x.shape[-2], device=x.device)
            emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
            x = x + self.freq_emb_scale * emb
        saved.append(x)

    if self.crosstransformer:
        if self.bottom_channels:
            b, c, f, t = x.shape
            x = rearrange(x, "b c f t-> b c (f t)")
            x = self.channel_upsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = self.channel_upsampler_t(xt)

        x, xt = self.crosstransformer(x, xt)

        if self.bottom_channels:
            x = rearrange(x, "b c f t-> b c (f t)")
            x = self.channel_downsampler(x)
            x = rearrange(x, "b c (f t)-> b c f t", f=f)
            xt = self.channel_downsampler_t(xt)

    for idx, decode in enumerate(self.decoder):
        skip = saved.pop(-1)
        x, pre = decode(x, skip, lengths.pop(-1))
        offset = self.depth - len(self.tdecoder)
        if idx >= offset:
            tdec = self.tdecoder[idx - offset]
            length_t = lengths_t.pop(-1)
            if tdec.empty:
                pre = pre[:, :, 0]
                xt, _ = tdec(pre, None, length_t)
            else:
                skip = saved_t.pop(-1)
                xt, _ = tdec(xt, skip, length_t)

    S = len(self.sources)
    x = x.view(B, S, -1, Fq, T)
    x = x * std[:, None] + mean[:, None]
    xt = xt.view(B, S, -1, training_length)
    xt = xt * stdt[:, None] + meant[:, None]
    return x, xt


# ----------------------------------------------------------------------------
# NumPy version of the plugin's C++ spectrogram code (HTDemucs._spec / _ispec)
NFFT, HL = 4096, 1024

def _window():
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)

def spec_cac(x):
    le = math.ceil(x.shape[-1] / HL)
    pad = HL // 2 * 3
    x = np.pad(x, ((0, 0), (pad, pad + le * HL - x.shape[-1])), mode="reflect")
    x = np.pad(x, ((0, 0), (NFFT // 2, NFFT // 2)), mode="reflect")
    frames = 1 + (x.shape[-1] - NFFT) // HL
    idx = np.arange(NFFT)[None, :] + HL * np.arange(frames)[:, None]
    z = np.fft.rfft(x[:, idx] * _window(), axis=-1) / np.sqrt(NFFT)
    z = z.transpose(0, 2, 1)[:, :-1, 2:2 + le]
    return np.stack([z.real, z.imag], axis=1).reshape(4, NFFT // 2, le).astype(np.float32)

def ispec_cac(m, length):
    z = m.reshape(2, 2, NFFT // 2, -1)
    z = z[:, 0] + 1j * z[:, 1]
    z = np.pad(z, ((0, 0), (0, 1), (2, 2)))
    pad = HL // 2 * 3
    le = HL * math.ceil(length / HL) + 2 * pad
    frames = z.shape[-1]
    w = _window()
    ola = np.zeros((2, NFFT + HL * (frames - 1)))
    env = np.zeros(NFFT + HL * (frames - 1))
    y = np.fft.irfft(z.transpose(0, 2, 1) * np.sqrt(NFFT), n=NFFT, axis=-1) * w
    for t in range(frames):
        ola[:, t * HL:t * HL + NFFT] += y[:, t]
        env[t * HL:t * HL + NFFT] += w * w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (ola / env)[:, NFFT // 2:NFFT // 2 + le]
    return out[:, pad:pad + length]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--six-stems", action="store_true", help="export htdemucs_6s (adds guitar and piano)")
    parser.add_argument("--out", type=Path, default=None, help=f"output folder (default: {default_model_folder()})")
    parser.add_argument("--skip-check", action="store_true", help="don't verify the exported model")
    args = parser.parse_args()

    name = "htdemucs_6s" if args.six_stems else "htdemucs"
    out_dir = args.out or default_model_folder()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{name}.onnx"

    print(f"Loading pretrained '{name}' (downloads ~80 MB from Meta the first time)...")
    model = get_model(name)
    if not isinstance(model, HTDemucs):
        if hasattr(model, "models") and len(model.models) == 1 and isinstance(model.models[0], HTDemucs):
            model = model.models[0]
        else:
            sys.exit(f"'{name}' is not a single HTDemucs model (bags of models aren't supported).")
    model.eval()
    model.cpu()

    segment = int(model.segment * model.samplerate)
    print(f"  sources: {', '.join(model.sources)}   segment: {segment} samples @ {model.samplerate} Hz")

    torch.manual_seed(0)
    mix = torch.randn(1, 2, segment) * 0.1
    with torch.no_grad():
        spec = model._magnitude(model._spec(mix))   # [1, 4, 2048, frames] (complex as channels)

    core = model
    original_forward = model.forward
    core.forward = types.MethodType(core_forward, core)

    print(f"Exporting to {out_file} ...")
    export_kwargs = dict(export_params=True, opset_version=17, do_constant_folding=True,
                         input_names=["mix", "spec"], output_names=["spec_out", "wave_out"])
    import inspect
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False   # the classic exporter is the one known to work for HTDemucs
    # nn.MultiheadAttention's inference "fast path" (aten::_native_multi_head_attention)
    # has no ONNX equivalent, so force the regular implementation. The fast path is
    # also only taken under no_grad, so export with autograd on.
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.export(core, (mix, spec), str(out_file), **export_kwargs)

    # Store the stem names so the plugin can label them
    import onnx
    onnx_model = onnx.load(str(out_file))
    for key, value in (("sources", ",".join(model.sources)), ("model", name), ("samplerate", str(model.samplerate))):
        entry = onnx_model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(onnx_model, str(out_file))
    size_mb = out_file.stat().st_size / 1e6
    print(f"  saved {size_mb:.0f} MB")

    if args.skip_check:
        return

    print("Checking the ONNX model against the original PyTorch model...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])

    # A more music-like test signal than noise
    t = np.arange(segment) / model.samplerate
    rng = np.random.default_rng(1)
    wave = np.stack([
        0.3 * np.sin(2 * np.pi * 110 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + 0.05 * rng.standard_normal(segment),
        0.3 * np.sin(2 * np.pi * 220 * t + 1) + 0.05 * rng.standard_normal(segment),
    ]).astype(np.float32)

    # 1) the full original model, STFT and all
    model.forward = original_forward
    with torch.no_grad():
        expected = model(torch.from_numpy(wave)[None])[0].numpy().astype(np.float64)   # [S, 2, T]

    # 2) plugin path: our spectrogram -> ONNX Runtime -> our inverse spectrogram + time branch
    spec_in = spec_cac(wave.astype(np.float64))[None]
    spec_out, wave_out = sess.run(None, {"mix": wave[None], "spec": spec_in})
    got = np.stack([ispec_cac(spec_out[0, s].astype(np.float64), segment) + wave_out[0, s]
                    for s in range(len(model.sources))])

    err = np.abs(got - expected).max() / (np.abs(expected).max() + 1e-12)
    print(f"  max relative difference: {err:.2e}")
    if err > 1e-3:
        sys.exit("  FAILED: the exported model doesn't match PyTorch. Please report this with your torch version "
                 f"({torch.__version__}).")
    print("  OK")
    print(f"\nDone. The plugin will find {out_file.name} automatically in:\n  {out_dir}")


if __name__ == "__main__":
    main()
