"""Stage 2: a latent *vocabulary* of sustained Spanish phones, found by gradient descent through the decoder.

For each phone a single static latent vector (repeated over 4 frames = 0.37 s) is optimised so the
decoded sound matches that phone's analytic spectrum (F0 120 Hz). All phones x restarts run as one
MPS batch. Outputs -> 02_phoneme_codebook/
"""
import os, json, time
import numpy as np, torch
import matplotlib.pyplot as plt
from samesdec import *
from phonetics import tracks, V
from optim_core import DEV, make_targets, spec_loss
from analysis import lpc_formants, bark
import librosa

OUT = os.path.join(DOCS, "02_phoneme_codebook")
os.makedirs(OUT, exist_ok=True)
PHONES = ["a", "e", "i", "o", "u", "l", "m", "n", "s", "h"]
R, T, STEPS = 12, 4, 300
N = T * HOP
torch.manual_seed(0)
core = load_decoder(DEV)


def stationary(ph):
    return [(ph, 1000 * N / SR + 50, 0, [(0.0, 120), (1.0, 118)])]


targets = make_targets(lambda ft: [tracks(ft, utt=stationary(p), seed=i) for i, p in enumerate(PHONES)], N)
idx = torch.arange(len(PHONES), device=DEV).repeat_interleave(R)
sil = torch.tensor(np.load(os.path.join(DOCS, "00_exploration", "silence_latent.npy")), device=DEV)
v = (sil[None] + 0.4 * torch.randn(len(idx), 256, device=DEV)).requires_grad_(True)
opt = torch.optim.Adam([v], lr=0.02)
t0 = time.time()
for step in range(STEPS):
    prog = min(1.0, step / (0.6 * STEPS))
    y = decode(core, v[:, :, None].repeat(1, 1, T))
    loss, _ = spec_loss(y, targets, idx, floor_db=-30 - 40 * prog, w_sc=1.0 - prog)
    opt.zero_grad(); loss.sum().backward(); opt.step()
    if step % 50 == 0 or step == STEPS - 1:
        with torch.no_grad():
            _, pf = spec_loss(y, targets, idx)
        best = pf.mean(1).view(len(PHONES), R).min(1).values.cpu().numpy()
        print(f"{step:4d} " + " ".join(f"{p}:{b:4.1f}" for p, b in zip(PHONES, best)) + f"  {time.time() - t0:.0f}s", flush=True)

# ---- select best restart per phone, save codebook + audio + analysis
with torch.no_grad():
    y = decode(core, v[:, :, None].repeat(1, 1, T))
    _, pf = spec_loss(y, targets, idx)
err = pf.mean(1).view(len(PHONES), R).cpu().numpy()
book, report = {}, {}
for i, p in enumerate(PHONES):
    j = i * R + int(err[i].argmin())
    book[p] = v[j].detach().cpu().numpy()
    yy = y[j].cpu().numpy()
    save_wav(os.path.join(OUT, f"phone_{p}.wav"), yy / np.abs(yy).max() * 0.89)
    spectrogram_png(os.path.join(OUT, f"phone_{p}.png"), yy, title=f"/{p}/  logmel err {err[i].min():.1f} dB", f_max=11025)
    mono = yy.mean(0)
    f0, vf, _ = librosa.pyin(mono, fmin=60, fmax=600, sr=SR, frame_length=2048)
    r = dict(err_db=float(err[i].min()), rms_db=float(20 * np.log10(np.sqrt((mono ** 2).mean()))),
             voiced_pct=float(100 * vf.mean()), f0=float(np.nanmedian(f0)) if vf.any() else 0.0,
             latent_norm=float(np.linalg.norm(book[p])))
    if p in V:
        y11 = librosa.resample(mono, orig_sr=SR, target_sr=11025)
        fm = lpc_formants(y11[len(y11) // 4: 3 * len(y11) // 4])
        r["F"] = [round(float(x)) for x in fm]
        if len(fm) >= 2:
            r["heard"] = min(V, key=lambda k: np.sum((bark(np.array(V[k][:2])) - bark(fm[:2])) ** 2))
    report[p] = r
    print(p, r)
np.savez(os.path.join(OUT, "codebook.npz"), **book)
json.dump(report, open(os.path.join(OUT, "codebook_report.json"), "w"), indent=1)

# latent-space geometry of the vocabulary
Z = np.stack([book[p] for p in PHONES])
Zc = Z - Z.mean(0)
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
P2 = Zc @ Vt[:2].T
zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].imshow(zn @ zn.T, cmap="viridis"); ax[0].set_xticks(range(len(PHONES)), PHONES); ax[0].set_yticks(range(len(PHONES)), PHONES)
ax[0].set_title("cosine similarity of phone latents")
ax[1].scatter(P2[:, 0], P2[:, 1])
for k, p in enumerate(PHONES):
    ax[1].text(P2[k, 0], P2[k, 1], f" /{p}/", fontsize=13)
ax[1].set_title("phone latents, PCA plane")
ax[2].imshow(err, aspect="auto", cmap="magma_r"); ax[2].set_yticks(range(len(PHONES)), PHONES)
ax[2].set_xlabel("restart"); ax[2].set_title("final log-mel error per restart (dB)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "codebook_geometry.png"), dpi=100); plt.close(fig)
