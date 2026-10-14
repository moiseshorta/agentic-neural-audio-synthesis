"""Final takes + latent-path analysis + progression montage. Outputs -> final/"""
import os, argparse, csv
import numpy as np, torch
import matplotlib.pyplot as plt
from samesdec import *
import phonetics as P_
from phonetics import timeline
from analysis import analyze

ap = argparse.ArgumentParser()
ap.add_argument("--takes", nargs="+", required=True, help="name=latents.npy[:cand]=UTT=book_dir=f0_scale=formant_scale")
ap.add_argument("--progression", nargs="*", default=[], help="latents.npy[:cand] in listening order")
args = ap.parse_args()
OUT = os.path.join(DOCS, "final")
os.makedirs(OUT, exist_ok=True)
core = load_decoder("mps")


def load(spec):
    path, _, cand = spec.partition(":")
    z = np.load(os.path.join(DOCS, path))
    return z[int(cand)] if cand else z


def dec(z):
    with torch.no_grad():
        return decode(core, torch.tensor(z, dtype=torch.float32, device="mps"))[0].cpu().numpy()


for spec in args.takes:
    name, lat, uttname, book, f0s, fms = spec.split("=")
    utt = getattr(P_, uttname)
    tl = timeline(utt)
    z = load(lat)
    y = dec(z)
    save_wav(os.path.join(OUT, f"hola_mundo_{name}.wav"), y / np.abs(y).max() * 0.89)
    np.save(os.path.join(OUT, f"latents_{name}.npy"), z)
    spectrogram_png(os.path.join(OUT, f"spectrogram_{name}.png"), y, title=f"'hola mundo, cómo estás hoy?' - {name} (SAME-S decoder only)",
                    annotations=[(a, b, ph) for a, b, ph, g, f in tl])
    m = analyze(y, os.path.join(OUT, f"perceptual_{name}"), utt=utt, f0_scale=float(f0s), formant_scale=float(fms), title=name)

    # the sentence as a path through the phone vocabulary's latent space
    pb = dict(np.load(os.path.join(DOCS, book, "pitch_codebook.npz")))
    keys = sorted(pb)
    B = np.stack([pb[k] for k in keys])
    mu = B.mean(0)
    Vt = np.linalg.svd(B - mu, full_matrices=False)[2][:2]
    pB, pZ = (B - mu) @ Vt.T, (z.T - mu) @ Vt.T
    rows = []
    for k in range(z.shape[1]):
        a0, a1 = k * HOP / SR, (k + 1) * HOP / SR
        phones = [ph for a, b, ph, g, f in tl if b > a0 and a < a1]
        d = np.linalg.norm(B - z[:, k], axis=1)
        rows.append([k, f"{a0:.3f}", "-".join(phones), keys[d.argmin()], f"{d.min():.2f}", f"{np.linalg.norm(z[:, k]):.2f}"])
    with open(os.path.join(OUT, f"latent_path_{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t_start_s", "phones_in_frame", "nearest_vocab_entry", "distance", "frame_norm"])
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(11, 8))
    for (x, y_), k in zip(pB, keys):
        ax.scatter(x, y_, s=18, color="lightgray")
        if "@" not in k or k.endswith(f"@{sorted({int(q.split('@')[1]) for q in keys if '@' in q})[1]}"):
            ax.text(x, y_, k.split("@")[0], color="gray", fontsize=11)
    ax.plot(pZ[:, 0], pZ[:, 1], "-", color="tab:blue", lw=1, alpha=0.6)
    sc = ax.scatter(pZ[:, 0], pZ[:, 1], c=np.arange(len(pZ)), cmap="plasma", s=40, zorder=3)
    for k in range(len(pZ)):
        ax.text(pZ[k, 0], pZ[k, 1], " " + rows[k][2], fontsize=7)
    plt.colorbar(sc, label="latent frame (92.9 ms each)")
    ax.set_title(f"{name}: the sentence as a path through the phone-vocabulary latent plane (PCA of {book})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"latent_path_{name}.png"), dpi=100); plt.close(fig)
    print(name, {k: round(v, 2) for k, v in m.items() if isinstance(v, float)})

if args.progression:  # every take decoded again, 0.6 s apart, peak-normalised each
    gap = np.zeros((2, int(0.6 * SR)))
    parts = []
    for spec in args.progression:
        y = dec(load(spec))
        parts += [y / np.abs(y).max() * 0.89, gap]
    save_wav(os.path.join(OUT, "progression_montage.wav"), np.concatenate(parts, 1))
    with open(os.path.join(OUT, "progression_montage.txt"), "w") as fh:
        fh.write("\n".join(args.progression) + "\n")
