#!/usr/bin/env python3
"""
Engine tests that don't need the real (170 MB) Demucs weights.

1. Spectrogram test: the C++ STFT/iSTFT must match a NumPy re-implementation of
   HTDemucs._spec/_ispec (torch.stft/istft with normalized=True, center=True, reflect pad).
2. Pipeline test: a dummy ONNX model with the same inputs/outputs as HTDemucs, where
   stem 0 = istft(stft(mix)) (frequency branch) and stem 1 = mix (time branch).
   Both must come back equal to the input file, which exercises normalisation, the
   random shift, segmenting, cross-fading, padding and resampling end to end.

usage: python tests/run_tests.py path/to/stemsplitter-cli
"""
import os, sys, subprocess, tempfile, math
import numpy as np
from scipy.io import wavfile

CLI = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
SEG, NFFT, HL = 343980, 4096, 1024
failures = []

def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + "  " + detail)
    if not ok:
        failures.append(name)

# ---------------------------------------------------------------- reference
def win():
    n = np.arange(NFFT)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / NFFT)

def ref_spec(x):
    le = math.ceil(x.shape[-1] / HL)
    pad = HL // 2 * 3
    x = np.pad(x, ((0, 0), (pad, pad + le * HL - x.shape[-1])), mode="reflect")
    x = np.pad(x, ((0, 0), (NFFT // 2, NFFT // 2)), mode="reflect")          # torch.stft center=True
    frames = 1 + (x.shape[-1] - NFFT) // HL
    idx = np.arange(NFFT)[None, :] + HL * np.arange(frames)[:, None]
    z = np.fft.rfft(x[:, idx] * win(), axis=-1) / np.sqrt(NFFT)            # [C, T, F]
    z = z.transpose(0, 2, 1)[:, :-1, :]                                       # drop Nyquist
    assert z.shape[-1] == le + 4
    z = z[..., 2:2 + le]
    m = np.stack([z.real, z.imag], axis=1).reshape(4, NFFT // 2, le)          # CaC
    return m.astype(np.float32)

def ref_ispec(m, length):
    z = m.reshape(2, 2, NFFT // 2, -1)
    z = z[:, 0] + 1j * z[:, 1]
    z = np.pad(z, ((0, 0), (0, 1), (2, 2)))
    pad = HL // 2 * 3
    le = HL * math.ceil(length / HL) + 2 * pad
    frames = z.shape[-1]
    ola = np.zeros((2, NFFT + HL * (frames - 1)))
    env = np.zeros(NFFT + HL * (frames - 1))
    w = win()
    y = np.fft.irfft(z.transpose(0, 2, 1) * np.sqrt(NFFT), n=NFFT, axis=-1) * w
    for t in range(frames):
        ola[:, t * HL:t * HL + NFFT] += y[:, t]
        env[t * HL:t * HL + NFFT] += w * w
    x = (ola / env)[:, NFFT // 2:NFFT // 2 + le]
    return x[:, pad:pad + length]

def rel_err(a, b):
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))

# ---------------------------------------------------------------- 1. spectrogram
with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, SEG)).astype(np.float32)
    x[:, :1000] *= np.linspace(0, 1, 1000)   # something non-stationary at the edges
    x.tofile(f"{tmp}/in.f32")
    subprocess.run([CLI, "--spec-test", f"{tmp}/in.f32", f"{tmp}/spec.f32", f"{tmp}/wave.f32"], check=True)
    le = math.ceil(SEG / HL)
    spec = np.fromfile(f"{tmp}/spec.f32", np.float32).reshape(4, NFFT // 2, le)
    wave = np.fromfile(f"{tmp}/wave.f32", np.float32).reshape(2, SEG)

    ref = ref_spec(x.astype(np.float64))
    e = rel_err(spec, ref)
    check("stft matches HTDemucs._spec", e < 1e-5, f"max rel err {e:.2e}")

    refw = ref_ispec(spec.astype(np.float64), SEG)
    e = rel_err(wave, refw)
    check("istft matches HTDemucs._ispec", e < 1e-5, f"max rel err {e:.2e}")

# ---------------------------------------------------------------- 2. pipeline
def make_music(sr, seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    sig = np.zeros((2, t.size))
    for f in [55, 110, 220, 330, 440, 660, 1234, 3000, 7000]:
        env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.1, 2) * t + rng.uniform(0, 6))
        sig[0] += 0.05 * env * np.sin(2 * np.pi * f * t + rng.uniform(0, 6))
        sig[1] += 0.05 * env * np.sin(2 * np.pi * f * 1.003 * t + rng.uniform(0, 6))
    # band-limited noise (well below Nyquist, which Demucs' spectrogram drops)
    noise = rng.standard_normal((2, t.size))
    spec = np.fft.rfft(noise, axis=-1)
    freqs = np.fft.rfftfreq(t.size, 1 / sr)
    spec[:, freqs > 15000] = 0
    sig += 0.05 * np.fft.irfft(spec, n=t.size, axis=-1)
    sig += 0.01  # a little DC so the mean/std normalisation is exercised
    return sig.astype(np.float32)

with tempfile.TemporaryDirectory() as tmp:
    for sources in (4, 6):
        subprocess.run([sys.executable, f"{HERE}/make_dummy_model.py", f"{tmp}/dummy{sources}.onnx", str(sources)],
                       check=True, capture_output=True)

    cases = [("44.1k, 40 s", 44100, 40.0, 4), ("48k, 21.3 s (resampled)", 48000, 21.3, 4),
             ("44.1k, 3 s (shorter than one segment)", 44100, 3.0, 6)]

    for i, (label, sr, secs, sources) in enumerate(cases):
        audio = make_music(sr, secs, i)
        wav = f"{tmp}/song{i}.wav"
        wavfile.write(wav, sr, audio.T)
        out = f"{tmp}/out{i}"
        r = subprocess.run([CLI, "--model", f"{tmp}/dummy{sources}.onnx", "--out", out, wav],
                           capture_output=True, text=True)
        if r.returncode != 0:
            check(label, False, r.stderr + r.stdout)
            continue

        files = sorted(os.listdir(out))
        names = [f.split(" - ")[-1][:-4] for f in files]
        expected = ["Bass", "Drums", "Other", "Vocals"] + (["Guitar", "Piano"] if sources == 6 else [])
        check(f"{label}: stem names", sorted(names) == sorted(expected), str(names))

        def load(name):
            srr, d = wavfile.read(f"{out}/song{i} - {name}.wav")
            assert srr == sr, (srr, sr)
            return d.T.astype(np.float64)

        # Demucs drops the Nyquist bin, so the STFT path is only near-perfect; resampling adds a bit more
        tol = 1e-3 if sr == 44100 else 3e-3
        drums = load("Drums")   # stem 0: frequency branch identity
        bass = load("Bass")     # stem 1: time branch identity
        vocals = load("Vocals") # stem 3: zeros -> denormalised mean
        check(f"{label}: length preserved", drums.shape == audio.shape, f"{drums.shape} vs {audio.shape}")
        e = rel_err(drums, audio)
        check(f"{label}: STFT path reconstructs input", e < tol, f"max rel err {e:.2e}")
        e = rel_err(bass, audio)
        check(f"{label}: time path reconstructs input", e < tol, f"max rel err {e:.2e}")
        mean = audio.mean()
        e = float(np.max(np.abs(vocals - mean)))
        check(f"{label}: silent stem equals mix mean", e < 1e-4, f"max abs err {e:.2e}")

# ---------------------------------------------------------------- 3. exact apply_model reference
def ref_apply(mix, ws, wt, off, overlap=0.25):
    """NumPy port of demucs apply_model(shifts=1, split=True) + separate.py normalisation."""
    ref = mix.mean(0)
    m, sd = ref.mean(), ref.std(ddof=1)
    x = (mix - m) / sd
    L, ms = x.shape[1], 44100 // 2
    U = np.pad(x, ((0, 0), (ms, ms)))
    Ls = L + ms - off
    S = len(ws)
    out = np.zeros((S, 2, Ls)); sw = np.zeros(Ls)
    weight = np.concatenate([np.arange(1, SEG // 2 + 1), np.arange(SEG - SEG // 2, 0, -1)]).astype(np.float64)
    weight /= weight.max()
    stride = int((1 - overlap) * SEG)
    for o in range(0, Ls, stride):
        cl = min(SEG, Ls - o); delta = SEG - cl; start = off + o - delta // 2
        chunk = np.zeros((2, SEG))
        a, b = max(0, start), min(U.shape[1], start + SEG)
        chunk[:, a - start:b - start] = U[:, a:b]
        spec = ref_spec(chunk)
        for s in range(S):
            y = ref_ispec(spec * ws[s], SEG) + chunk * wt[s]
            out[s, :, o:o + cl] += weight[:cl] * y[:, delta // 2:delta // 2 + cl]
        sw[o:o + cl] += weight[:cl]
    out /= sw
    return out[..., ms - off:ms - off + L] * sd + m

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, f"{HERE}/make_dummy_model.py", f"{tmp}/d.onnx", "4"], check=True, capture_output=True)
    for secs, off in [(25.0, 1234), (9.0, 22050), (5.5, 0)]:
        audio = make_music(44100, secs, 7)
        wavfile.write(f"{tmp}/r.wav", 44100, audio.T)
        subprocess.run([CLI, "--model", f"{tmp}/d.onnx", "--out", f"{tmp}/o", "--shift-offset", str(off), f"{tmp}/r.wav"],
                       check=True, capture_output=True)
        ws = [1, 0, 0.5, 0]; wt = [0, 1, -0.25, 0]
        expected = ref_apply(audio.astype(np.float64), ws, wt, off)
        worst = 0.0
        for s, name in enumerate(["Drums", "Bass", "Other", "Vocals"]):
            got = wavfile.read(f"{tmp}/o/r - {name}.wav")[1].T.astype(np.float64)
            worst = max(worst, rel_err(got, expected[s]))
        check(f"matches NumPy port of demucs apply_model ({secs} s, shift {off})", worst < 2e-5, f"max rel err {worst:.2e}")

print()
print("ALL TESTS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
