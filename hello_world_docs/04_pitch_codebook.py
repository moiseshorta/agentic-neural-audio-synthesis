"""Stage 4: pitch-aware phone vocabulary.

Spectral losses cannot slide harmonics to a new F0 (iteration 3 stayed at the codebook's 120 Hz),
so every voiced phone gets its own latent at 5 pitch levels:
(8 voiced phones x 5 F0 + /s/ + /h/) x 6 restarts, static latent per candidate, 4 frames.
Optimised in MPS batches of <= 10 targets (~240 latent frames): bigger batches overflow unified
memory into swap and run ~10x slower. Also asks: is pitch a consistent *direction* in latent space?
Outputs -> 04_pitch_codebook/
"""
import os, json, time
import numpy as np, torch, librosa
import matplotlib.pyplot as plt
from samesdec import *
from phonetics import tracks, V
from optim_core import DEV, make_targets, spec_loss
from analysis import lpc_formants, bark

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="04_pitch_codebook")
ap.add_argument("--f0s", default="95,115,140,170,205")
ap.add_argument("--formant_scale", type=float, default=1.0, help="1.15 ~ adult female vocal tract")
args = ap.parse_args()
OUT = os.path.join(DOCS, args.out)
os.makedirs(OUT, exist_ok=True)
VOICED = ["a", "e", "i", "o", "u", "l", "m", "n"]
F0S = [int(f) for f in args.f0s.split(",")]
KEYS = [f"{p}@{f}" for p in VOICED for f in F0S] + ["s", "h"]
R, T, STEPS, CHUNK = 6, 4, 400, 10
N = T * HOP
torch.manual_seed(1)
core = load_decoder(DEV)
sil = torch.tensor(np.load(os.path.join(DOCS, "00_exploration", "silence_latent.npy")), device=DEV)
old = dict(np.load(os.path.join(DOCS, "02_phoneme_codebook", "codebook.npz")))


def stationary(key):
    p, f = (key.split("@") + ["120"])[:2]
    f = float(f)
    return [(p, 1000 * N / SR + 50, 0, [(0.0, f), (1.0, f * 0.98)])]


def fit(keys, seed0):
    targets = make_targets(lambda ft: [tracks(ft, utt=stationary(k), seed=seed0 + i, f_ap=4500 * args.formant_scale,
                                              formant_scale=args.formant_scale) for i, k in enumerate(keys)], N)
    idx = torch.arange(len(keys), device=DEV).repeat_interleave(R)
    v0 = sil[None] + 0.4 * torch.randn(len(idx), 256, device=DEV)
    for i, k in enumerate(keys):  # restart 0 of every key warm-starts from the 120 Hz phone latent
        v0[i * R] = torch.tensor(old[k.split("@")[0]], device=DEV) + 0.05 * torch.randn(256, device=DEV)
    v = v0.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.02)
    t0 = time.time()
    for step in range(STEPS):
        prog = min(1.0, step / (0.6 * STEPS))
        y = decode(core, v[:, :, None].repeat(1, 1, T))
        loss, pf = spec_loss(y, targets, idx, floor_db=-30 - 40 * prog, w_sc=1.0 - prog, w_harm=0.5)
        opt.zero_grad(); loss.sum().backward(); opt.step()
        if step % 100 == 0 or step == STEPS - 1:
            best = pf.mean(1).view(len(keys), R).min(1).values.cpu().numpy()
            print(f"  {step:4d} mean-best {best.mean():.2f} dB  {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        y = decode(core, v[:, :, None].repeat(1, 1, T))
        _, pf = spec_loss(y, targets, idx)
    err = pf.mean(1).view(len(keys), R).cpu().numpy()
    pick = [i * R + int(err[i].argmin()) for i in range(len(keys))]
    out = (v[pick].detach().cpu().numpy(), err, y[pick].mean(1).cpu().numpy())
    del targets, v, y, opt
    torch.mps.empty_cache()
    return out


book, report = {}, {}
for c0 in range(0, len(KEYS), CHUNK):
    keys = KEYS[c0:c0 + CHUNK]
    print("chunk", keys, flush=True)
    vb, err, monos = fit(keys, c0)
    for i, k in enumerate(keys):
        book[k] = vb[i]
        mono = monos[i]
        f0, vf, _ = librosa.pyin(mono, fmin=60, fmax=500, sr=SR, frame_length=2048)
        r = dict(err_db=round(float(err[i].min()), 2), restart=int(err[i].argmin()), voiced_pct=round(float(100 * vf.mean())))
        if "@" in k:
            ft = float(k.split("@")[1])
            r["f0"] = round(float(np.nanmedian(f0)), 1) if vf.any() else 0.0
            r["f0_err_cents"] = round(float(1200 * np.log2(r["f0"] / ft))) if r["f0"] else None
        p = k.split("@")[0]
        if p in V:
            y11 = librosa.resample(mono, orig_sr=SR, target_sr=11025)
            fm = lpc_formants(y11[len(y11) // 4: 3 * len(y11) // 4])
            r["F"] = [round(float(x)) for x in fm]
            if len(fm) >= 2:
                r["heard"] = min(V, key=lambda q: np.sum((bark(np.array(V[q][:2]) * args.formant_scale) - bark(fm[:2])) ** 2))
        report[k] = r
        print(k, r, flush=True)
np.savez(os.path.join(OUT, "pitch_codebook.npz"), **book)
json.dump(report, open(os.path.join(OUT, "pitch_codebook_report.json"), "w"), indent=1)

# pitch ladders: each voiced phone "sung" across its 5 F0 levels (one latent sequence, 20 frames)
with torch.no_grad():
    for p in VOICED:
        zz = torch.tensor(np.stack([book[f"{p}@{f}"] for f in F0S for _ in range(T)], 1), device=DEV)
        yy = decode(core, zz)[0].cpu().numpy()
        save_wav(os.path.join(OUT, f"ladder_{p}.wav"), yy / np.abs(yy).max() * 0.89)
        spectrogram_png(os.path.join(OUT, f"ladder_{p}.png"), yy, title=f"/{p}/ at F0 = {F0S} Hz", f_max=5000, n_fft=4096)

# is pitch a direction? regress latent on log F0 per phone, compare slopes across phones
lf = np.log(F0S)
S = np.stack([np.polyfit(lf - lf.mean(), np.stack([book[f"{p}@{f}"] for f in F0S]), 1)[0] for p in VOICED])
Sn = S / np.linalg.norm(S, axis=1, keepdims=True)
np.save(os.path.join(OUT, "pitch_direction.npy"), S.mean(0))
allZ = np.stack([book[k] for k in KEYS if "@" in k])
c = allZ - allZ.mean(0)
P2 = c @ np.linalg.svd(c, full_matrices=False)[2][:2].T
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
ax[0].imshow(Sn @ Sn.T, cmap="coolwarm", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(VOICED)), VOICED); ax[0].set_yticks(range(len(VOICED)), VOICED)
ax[0].set_title("cosine sim. of per-phone d(latent)/d(log F0)")
sc = ax[1].scatter(P2[:, 0], P2[:, 1], c=np.tile(F0S, len(VOICED)), cmap="viridis", s=40)
for n_, k in enumerate([k for k in KEYS if "@" in k]):
    if k.endswith(f"@{F0S[0]}"):
        ax[1].text(P2[n_, 0], P2[n_, 1], k.split("@")[0], fontsize=12)
plt.colorbar(sc, ax=ax[1], label="F0 Hz"); ax[1].set_title("voiced phone latents, PCA (colour = F0)")
E = np.array([[report[f"{p}@{f}"]["err_db"] for f in F0S] for p in VOICED])
im = ax[2].imshow(E, cmap="magma_r"); plt.colorbar(im, ax=ax[2], label="dB")
ax[2].set_xticks(range(len(F0S)), F0S); ax[2].set_yticks(range(len(VOICED)), VOICED); ax[2].set_title("fit error per (phone, F0)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "pitch_geometry.png"), dpi=100); plt.close(fig)
print("mean |cos| between phone pitch slopes:", float(np.abs(Sn @ Sn.T)[~np.eye(len(VOICED), dtype=bool)].mean()))
