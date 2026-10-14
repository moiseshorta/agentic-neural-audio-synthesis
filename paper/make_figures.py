"""Paper schematic (Fig. 2): (a) the ANAS loop, (b) the pitch-direction diagnostic from the case study, (c) local latent surgery.
Run: python paper/make_figures.py  -> paper/fig_schematic.pdf (+ .png preview)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "hello_world_docs")
FOREST, MOSS, INK, GREY = "#2d5a3c", "#8ca98e", "#1a1a1a", "#8a8a8a"
VOICED = ["a", "e", "i", "o", "u", "l", "m", "n"]


def atlas(book="15_pitch_codebook_female", latents="final/latents_female.npy"):
    """Unit x pitch vocabulary and a decoded sentence, projected on the vocabulary's first two principal components."""
    pb = np.load(os.path.join(DOCS, book, "pitch_codebook.npz"))
    keys = sorted(pb.files)
    B = np.stack([pb[k] for k in keys])
    mu = B.mean(0)
    V = np.linalg.svd(B - mu, full_matrices=False)[2][:2]
    P = dict(zip(keys, (B - mu) @ V.T))
    f0s = sorted({int(k.split("@")[1]) for k in keys if "@" in k})
    ladders = {p: np.stack([P[f"{p}@{f}"] for f in f0s]) for p in VOICED}
    path = (np.load(os.path.join(DOCS, latents)).T - mu) @ V.T
    return ladders, {k: P[k] for k in ("s", "h")}, path, f0s


def draw_atlas(ax, ink=INK, faint=GREY, accent=FOREST, path_color=MOSS, fs=7):
    ladders, single, path, f0s = atlas()
    ax.plot(path[:, 0], path[:, 1], "-", color=path_color, lw=0.8, alpha=0.7, zorder=1)
    for p, L in ladders.items():
        ax.plot(L[:, 0], L[:, 1], "-", color=accent, lw=1.1, zorder=2)
        ax.scatter(L[:, 0], L[:, 1], s=np.linspace(4, 22, len(L)), color=accent, zorder=3, linewidths=0)
        ax.text(L[-1, 0], L[-1, 1], f" /{p}/", color=ink, fontsize=fs, va="center", zorder=4)
    for k, (x, y) in single.items():
        ax.scatter([x], [y], s=14, color=faint, zorder=3, linewidths=0)
        ax.text(x, y, f" /{k}/", color=faint, fontsize=fs, va="center")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return f0s


def pitch_cos(book):
    """|cos| between per-phone pitch directions: slope of the latent regressed on log F0 (as in 04_pitch_codebook.py)."""
    pb = np.load(os.path.join(DOCS, book, "pitch_codebook.npz"))
    f0s = sorted({int(k.split("@")[1]) for k in pb.files if "@" in k})
    lf = np.log(f0s)
    S = np.stack([np.polyfit(lf - lf.mean(), np.stack([pb[f"{p}@{f}"] for f in f0s]), 1)[0] for p in VOICED])
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    return np.abs(S @ S.T)


def draw_pitch_cos(ax):
    from matplotlib.colors import LinearSegmentedColormap
    male, female = pitch_cos("04_pitch_codebook"), pitch_cos("15_pitch_codebook_female")
    n = len(VOICED)
    M = np.where(np.tri(n, k=-1, dtype=bool), male, female)
    np.fill_diagonal(M, np.nan)
    ax.imshow(M, cmap=LinearSegmentedColormap.from_list("f", ["white", FOREST]), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "1" if i == j else f"{M[i, j]:.2f}"[1:], ha="center", va="center", fontsize=5,
                    color=GREY if i == j else INK)
    ax.set_xticks(range(n), [f"/{p}/" for p in VOICED], fontsize=6)
    ax.set_yticks(range(n), [f"/{p}/" for p in VOICED], fontsize=6)
    ax.tick_params(length=0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    off = ~np.eye(n, dtype=bool)
    ax.set_xlabel(f"|cos|, below diagonal: male (mean {male[off].mean():.2f}), above: female ({female[off].mean():.2f})",
                  fontsize=5.6, labelpad=3)


def box(ax, x, y, w, h, text, fc="white", ec=INK, tc=INK, fs=7, lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.15)


def arrow(ax, a, b, text="", color=INK, ls="-", rad=0.0, fs=6, off=(0, 0)):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, ls=ls, shrinkA=0, shrinkB=0,
                                                  connectionstyle=f"arc3,rad={rad}", mutation_scale=7))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, fontsize=fs, ha="center", va="center", color=color)


def draw_loop(ax):
    ax.set_xlim(0, 1); ax.set_ylim(-0.22, 1); ax.axis("off")
    fs = 6.3
    box(ax, 0.00, 0.80, 0.28, 0.15, "request $q$\n“make it say …”", fc="#f4f4f4", ec="#f4f4f4", fs=fs)
    box(ax, 0.00, 0.36, 0.28, 0.32, "LLM agent\n+ ANAS skill", fc=FOREST, ec=FOREST, tc="white", fs=fs)
    box(ax, 0.36, 0.76, 0.32, 0.19, "analytic target\n$\\mathbb{E}|X_t(f)|^2$ (P2)", fs=fs)
    box(ax, 0.36, 0.40, 0.32, 0.26, "latents $Z$\natlas · search ·\nsurgery (P3–P6)", fs=fs)
    box(ax, 0.36, 0.04, 0.32, 0.22, "frozen decoder $D$\nprobes (P0–P1)", fc="#eef3ef", ec=FOREST, fs=fs)
    box(ax, 0.76, 0.40, 0.24, 0.26, "loss +\nguards (P5)", fs=fs)
    box(ax, 0.76, 0.04, 0.24, 0.22, "audio $\\hat y$\nproxies (P7)", fs=fs)
    arrow(ax, (0.14, 0.80), (0.14, 0.68))
    arrow(ax, (0.28, 0.62), (0.36, 0.85))
    arrow(ax, (0.28, 0.53), (0.36, 0.53))
    arrow(ax, (0.52, 0.40), (0.52, 0.26))
    arrow(ax, (0.68, 0.15), (0.76, 0.15))
    arrow(ax, (0.88, 0.26), (0.88, 0.40))
    arrow(ax, (0.68, 0.86), (0.88, 0.66), rad=-0.3)
    arrow(ax, (0.76, 0.53), (0.68, 0.53), "$\\nabla_Z$", color=FOREST, off=(0, 0.05))
    arrow(ax, (0.86, 0.04), (0.10, 0.36), color=GREY, ls="--", rad=-0.45)
    ax.text(0.47, -0.205, "measurements, never listening", fontsize=5.8, color=GREY, ha="center")


def draw_surgery(ax):
    ax.set_xlim(-1.05, 6.1); ax.set_ylim(-0.25, 4.45); ax.axis("off")
    for k in range(6):
        free = k in (2, 3)
        ax.add_patch(Rectangle((k + 0.06, 3.3), 0.88, 0.75, fc=FOREST if free else "white", ec=INK, lw=0.7))
        ax.text(k + 0.5, 3.675, f"$z_{{s+{k}}}$" if k else "$z_s$", ha="center", va="center", fontsize=6.5,
                color="white" if free else INK)
    def bracket(a, b, y, c):
        ax.plot([a + 0.1, a + 0.1, b + 0.9, b + 0.9], [y + 0.18, y, y, y + 0.18], color=c, lw=0.9)
    for a in (0, 2, 4):
        bracket(a, a + 1, 2.75, INK)
    for a in (1, 3):
        bracket(a, a + 1, 2.2, GREY)
    ax.text(-1.0, 2.82, "pass 1", fontsize=5.5, va="center", color=INK)
    ax.text(-1.0, 2.27, "pass 2", fontsize=5.5, va="center", color=GREY)
    shade = {0: "#f2f2f2", 1: "#dbe6de", 2: FOREST, 3: FOREST, 4: "#dbe6de", 5: "#f2f2f2"}
    for k in range(6):
        ax.add_patch(Rectangle((k + 0.06, 1.15), 0.88, 0.62, fc=shade[k], ec=INK, lw=0.7))
    ax.text(2.5 + 0.5, 1.46, "exact", ha="center", va="center", fontsize=6.5, color="white")
    ax.text(2.7, 0.82, "decoded output of the 6-frame window", ha="center", fontsize=5.6, color=INK)
    ax.text(2.7, 0.42, "$s{+}2, s{+}3$ equal the full decode ($|\\Delta|\\leq10^{-6}$)", ha="center", fontsize=5.6, color=FOREST)
    ax.text(2.7, 0.02, "re-search them: 48 restarts, splice if better", ha="center", fontsize=5.6, color=INK)


if __name__ == "__main__":
    fig = plt.figure(figsize=(7.1, 2.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 0.95, 1.1], wspace=0.14, left=0.005, right=0.995, bottom=0.14, top=0.9)
    axes = [fig.add_subplot(g) for g in gs]
    draw_loop(axes[0])
    draw_pitch_cos(axes[1])
    draw_surgery(axes[2])
    for ax, t in zip(axes, ["(a) the ANAS loop", "(b) pitch is not a shared direction", "(c) local latent surgery"]):
        ax.set_title(t, fontsize=7.5, loc="left", pad=3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig_schematic.{ext}"), dpi=200)
