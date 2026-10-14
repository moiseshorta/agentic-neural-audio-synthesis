"""Shared helpers: load ONLY the SAME-S decoder, decode latents, save wav + spectrogram.

The encoder in the TorchScript file is never called anywhere in this project.
"""
import os
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 44100            # Stable Audio family codecs run at 44.1 kHz
HOP = 4096            # samples per latent frame (patch 256 x stride 16)
DIM = 256             # latent channels
MODEL = os.path.join(os.path.dirname(__file__), "..", "neural-audio-codec", "same-s-cpu.ts")
DOCS = os.path.dirname(os.path.abspath(__file__))


def load_decoder(device="cpu"):
    core = torch.jit.load(MODEL, map_location=device).eval().core
    for p in core.parameters():
        p.requires_grad_(False)
    return core


def decode(core, z):
    """z: (256, T) or (B, 256, T) latents, T even -> (B, 2, T*4096) audio (torch)."""
    if z.dim() == 2:
        z = z[None]
    assert z.shape[-1] % 2 == 0, "SAME-S decoder needs an even number of latent frames"
    return core.decode(z)


def save_wav(path, y):
    """y: (2, N) or (N,) numpy float audio."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.T
    sf.write(path, y, SR, subtype="PCM_16")


def spectrogram_png(path, y, title="", f_max=11025, annotations=None, n_fft=2048, hop=256):
    """Mono log-mag spectrogram (+ waveform). annotations: list of (t_start, t_end, label)."""
    y = np.asarray(y)
    mono = y.mean(0) if y.ndim == 2 else y
    S = librosa.amplitude_to_db(np.abs(librosa.stft(mono, n_fft=n_fft, hop_length=hop)), ref=np.max, top_db=80)
    t = np.arange(len(mono)) / SR
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    ax[0].plot(t, mono, lw=0.4, color="k")
    ax[0].set_ylabel("amp")
    ax[0].set_title(title)
    ext = [0, len(mono) / SR, 0, SR / 2]
    ax[1].imshow(S, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=-80, vmax=0)
    ax[1].set_ylim(0, f_max)
    ax[1].set_ylabel("Hz")
    ax[1].set_xlabel("s")
    for a, b, lab in annotations or []:
        ax[1].axvline(a, color="cyan", lw=0.5, alpha=0.6)
        ax[1].text((a + b) / 2, f_max * 0.95, lab, color="cyan", ha="center", va="top", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
