"""Stage 0: what does the SAME-S decoder do with latents we invent?

Outputs -> 00_exploration/
"""
import os, json
import numpy as np, torch, librosa
import matplotlib.pyplot as plt
from samesdec import *

OUT = os.path.join(DOCS, "00_exploration")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
core = load_decoder()


def feats(y):
    """Crude perceptual descriptors of a stereo decoder output (numpy (2,N))."""
    mono = y.mean(0)
    rms = float(np.sqrt((mono ** 2).mean()) + 1e-9)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=512)) + 1e-9
    cent = float(librosa.feature.spectral_centroid(S=S, sr=SR).mean())
    flat = float(librosa.feature.spectral_flatness(S=S).mean())
    f0, vflag, vprob = librosa.pyin(mono, fmin=60, fmax=800, sr=SR, frame_length=4096, hop_length=512)
    vfrac = float(np.mean(vflag))
    f0m = float(np.nanmedian(f0)) if vfrac > 0 else 0.0
    side = float(np.sqrt(((y[0] - y[1]) ** 2).mean()) / (2 * rms))
    return dict(rms_db=20 * np.log10(rms), centroid=cent, flatness=flat, voiced=vfrac, f0=f0m, side_ratio=side)


# 1) which latent directions does the decoder listen to?
W = core.dec_project_in.weight.detach().numpy()  # (768, 256)
sv = np.linalg.svd(W, compute_uv=False)
col_norm = np.linalg.norm(W, axis=0)
np.savez(os.path.join(OUT, "dec_project_in_analysis.npz"), singular_values=sv, column_norms=col_norm)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].semilogy(sv); ax[0].set_title("dec_project_in singular values")
ax[1].bar(range(256), col_norm); ax[1].set_title("per-latent-dim input weight norm")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dec_project_in_analysis.png"), dpi=100); plt.close(fig)

# 2) probe latents
T = 8
probes = {"zeros": torch.zeros(256, T)}
for s in [0.25, 0.5, 0.75, 1.0]:
    probes[f"randn_x{s}"] = s * torch.randn(256, T)
v = torch.randn(256, 1)
for s in [0.5, 1.0]:
    probes[f"static_x{s}"] = (s * v).repeat(1, T)
table = {}
with torch.no_grad():
    for name, z in probes.items():
        y = decode(core, z)[0].numpy()
        save_wav(os.path.join(OUT, f"probe_{name}.wav"), y)
        spectrogram_png(os.path.join(OUT, f"probe_{name}.png"), y, title=name)
        table[name] = feats(y)
        print(name, {k: round(v, 3) for k, v in table[name].items()})

# 3) silence: optimise a latent whose decoded output has minimal energy
z = torch.zeros(256, 4, requires_grad=True)
opt = torch.optim.Adam([z], lr=0.05)
for i in range(300):
    y = decode(core, z)
    loss = y.pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
silence = z.detach().mean(1)
with torch.no_grad():
    y = decode(core, silence[:, None].repeat(1, 4))[0].numpy()
print("silence latent rms_db", 20 * np.log10(np.sqrt((y ** 2).mean())), "norm", silence.norm().item())
np.save(os.path.join(OUT, "silence_latent.npy"), silence.numpy())

# 4) single-dimension sweeps around the silence latent
rows = []
with torch.no_grad():
    for d in range(256):
        for sgn in (-3.0, 3.0):
            zz = silence[:, None].repeat(1, 4).clone()
            zz[d] += sgn
            y = decode(core, zz)[0].numpy()
            f = feats(y)
            rows.append([d, sgn] + [f[k] for k in ["rms_db", "centroid", "flatness", "voiced", "f0", "side_ratio"]])
rows = np.array(rows)
np.savetxt(os.path.join(OUT, "dim_sweep.csv"), rows, delimiter=",", fmt="%.4f",
           header="dim,offset,rms_db,centroid_hz,flatness,voiced_frac,f0_hz,side_ratio")
json.dump(table, open(os.path.join(OUT, "probe_features.json"), "w"), indent=1)
print("top voiced dims:", rows[np.argsort(-rows[:, 5])[:15]][:, [0, 1, 2, 5, 6]].round(2).tolist())
print("top loud dims:", rows[np.argsort(-rows[:, 2])[:10]][:, [0, 1, 2, 3]].round(1).tolist())
