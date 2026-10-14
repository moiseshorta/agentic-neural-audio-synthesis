"""Optimise SAME-S latents (decoder only) so the decoded audio says "hola mundo, cómo estás hoy?".

Latents are free torch parameters; the only audio generator is core.decode().
B random restarts are optimised together in one batched decoder call on MPS.
Objective: decoded multi-resolution mel spectrogram vs the analytic phonetic target
(phonetics.py), with a coarse-to-fine curriculum:
  early: Stevens-law compressed power (power^0.3) + log-mel with a high floor (-35 dB)
  late : log-mel with a deep floor (-75 dB) so silences and weak consonants matter.
"""
import os, argparse, json, time
import numpy as np, torch, librosa
import matplotlib.pyplot as plt
from samesdec import *
from phonetics import UTT, timeline, tracks, target_power
from analysis import analyze

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="02_curriculum")
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--batch", type=int, default=8, help="parallel random restarts")
ap.add_argument("--init_noise", type=float, default=0.3)
ap.add_argument("--f0_scale", type=float, default=1.0)
ap.add_argument("--formant_scale", type=float, default=1.0)
ap.add_argument("--reg", type=float, default=1e-3, help="L2 pull of latents toward the silence latent")
ap.add_argument("--init", default="silence", help="silence | path/to/latents.npy")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

OUT = os.path.join(DOCS, args.tag)
os.makedirs(OUT, exist_ok=True)
dev = torch.device("mps")
core = load_decoder(dev)
torch.manual_seed(args.seed)

tl = timeline()
T = int(np.ceil(tl[-1][1] * SR / HOP)); T += T % 2
N = T * HOP
B = args.batch

RES = [(2048, 256, 128), (512, 128, 64)]  # (n_fft, hop, n_mels): pitch/formant detail + temporal detail
targets = []
for n_fft, hop, n_mels in RES:
    ft = np.arange(1 + N // hop) * hop / SR
    tr = tracks(ft, f0_scale=args.f0_scale, formant_scale=args.formant_scale, seed=args.seed)
    fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmax=16000)
    M = target_power(tr, n_fft) @ fb.T
    Mt = torch.tensor(M.T, dtype=torch.float32, device=dev)
    targets.append(dict(n_fft=n_fft, hop=hop, M=M, peak=float(M.max()), Mt=Mt, Mc=Mt ** 0.3,
                        fb=torch.tensor(fb, device=dev), win=torch.hann_window(n_fft, device=dev)))


def losses(y, floor_db, w_comp):
    """Per-candidate loss (B,) and the -75 dB log-mel distance (B,) used to rank candidates."""
    tot = torch.zeros(y.shape[0], device=dev)
    rank = torch.zeros(y.shape[0], device=dev)
    for tg in targets:
        X = torch.stft(y.reshape(-1, y.shape[-1]), tg["n_fft"], tg["hop"], window=tg["win"], center=True, return_complex=True)
        M = (tg["fb"] @ (X.real ** 2 + X.imag ** 2)).reshape(y.shape[0], 2, -1, X.shape[-1])
        fl = tg["peak"] * 10 ** (floor_db / 10)
        logd = 10 * (torch.log10(M + fl) - torch.log10(tg["Mt"] + fl)).abs().mean((1, 2, 3))
        comp = (M.clamp_min(0) ** 0.3 - tg["Mc"]).abs().mean((1, 2, 3)) / tg["Mc"].mean()
        tot = tot + logd + w_comp * 10 * comp
        with torch.no_grad():
            fl75 = tg["peak"] * 10 ** -7.5
            rank += 10 * (torch.log10(M + fl75) - torch.log10(tg["Mt"] + fl75)).abs().mean((1, 2, 3))
    return tot / len(targets), rank / len(targets)


sil = torch.tensor(np.load(os.path.join(DOCS, "00_exploration", "silence_latent.npy")), device=dev)
if args.init == "silence":
    z0 = sil[None, :, None].repeat(B, 1, T) + args.init_noise * torch.randn(B, 256, T, device=dev)
else:
    zi = torch.tensor(np.load(args.init), dtype=torch.float32, device=dev)
    zi = zi if zi.dim() == 3 else zi[None]
    z0 = zi[:1].repeat(B, 1, 1) + args.init_noise * torch.randn(B, 256, T, device=dev) * (torch.arange(B, device=dev) > 0)[:, None, None]
z = z0.clone().requires_grad_(True)
opt = torch.optim.Adam([z], lr=args.lr)
warm = 30
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * s / args.steps))))
hist = []
t0 = time.time()
for step in range(args.steps):
    prog = min(1.0, step / (0.6 * args.steps))
    floor_db = -35 - 40 * prog
    w_comp = 1.0 - 0.7 * prog
    y = decode(core, z)
    loss_b, rank_b = losses(y, floor_db, w_comp)
    reg_b = ((z - sil[None, :, None]) ** 2).mean((1, 2))
    loss = (loss_b + args.reg * reg_b).sum()  # sum -> candidates get independent gradients
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    hist.append(rank_b.detach().cpu().numpy())
    if step % 50 == 0 or step == args.steps - 1:
        r = rank_b.detach().cpu().numpy()
        print(f"{step:4d} floor {floor_db:5.1f}  -75dB-logmel best {r.min():5.2f} mean {r.mean():5.2f}  "
              f"z-std {z.detach().std().item():.2f}  {time.time() - t0:.0f}s", flush=True)

# ---- pick best restart and save everything
hist = np.array(hist)
best = int(hist[-1].argmin())
zall = z.detach().cpu().numpy()
np.save(os.path.join(OUT, "latents_all_restarts.npy"), zall)
zf = zall[best]
np.save(os.path.join(OUT, "latents.npy"), zf)
with torch.no_grad():
    y = decode(core, z[best:best + 1])[0].cpu().numpy()
peak = np.abs(y).max()
save_wav(os.path.join(OUT, "hola_mundo.wav"), y / peak * 0.89)
ann = [(a, b, ph) for a, b, ph, g, f in tl]
spectrogram_png(os.path.join(OUT, "spectrogram.png"), y, title=f"{args.tag} restart {best} (decoded)", f_max=11025, annotations=ann)

tg = targets[0]
Xo = librosa.feature.melspectrogram(y=y.mean(0), sr=SR, n_fft=tg["n_fft"], hop_length=tg["hop"], n_mels=128, fmax=16000, power=2)
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a_, M_, ttl in [(ax[0], tg["M"].T, "analytic phonetic target (never rendered to audio)"), (ax[1], Xo, "decoded audio")]:
    a_.imshow(10 * np.log10(M_ + M_.max() * 10 ** -7.5), origin="lower", aspect="auto", cmap="magma",
              extent=[0, N / SR, 0, 128], vmax=10 * np.log10(M_.max()), vmin=10 * np.log10(M_.max()) - 75)
    a_.set_title(ttl); a_.set_ylabel("mel bin")
    for a, b, ph in ann:
        a_.axvline(a, color="cyan", lw=0.4); a_.text((a + b) / 2, 122, ph, color="cyan", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "target_vs_decoded_mel.png"), dpi=100); plt.close(fig)

# latent analysis: heatmap, per-frame norm, frame-to-frame cosine similarity, restart losses
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5), gridspec_kw={"width_ratios": [3, 1.3, 1.3, 1.3]})
ax[0].imshow(zf, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3); ax[0].set_title("latents (256 x T)"); ax[0].set_xlabel("frame (92.9 ms)")
ax[1].plot(np.linalg.norm(zf, axis=0), "o-"); ax[1].axhline(sil.norm().item(), color="gray", ls="--"); ax[1].set_title("frame L2 norm (dashed = silence)")
zn = zf / np.linalg.norm(zf, axis=0, keepdims=True)
ax[2].imshow(zn.T @ zn, cmap="viridis", vmin=-1, vmax=1); ax[2].set_title("frame cosine similarity")
ax[3].plot(hist); ax[3].set_title("-75 dB log-mel dist per restart"); ax[3].set_xlabel("step"); ax[3].set_ylim(0, None)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "latent_analysis.png"), dpi=100); plt.close(fig)
np.savetxt(os.path.join(OUT, "loss_curve.csv"), hist, delimiter=",", fmt="%.4f", header="logmel_db per restart")

m = analyze(y, os.path.join(OUT, "perceptual"), f0_scale=args.f0_scale, formant_scale=args.formant_scale, title=args.tag)
m.update(final_logmel_db=float(hist[-1, best]), restart_final_db=hist[-1].round(2).tolist(), best_restart=best,
         frames=T, seconds=N / SR, args=vars(args),
         latent_mean=float(zf.mean()), latent_std=float(zf.std()), frame_norm_mean=float(np.linalg.norm(zf, axis=0).mean()))
json.dump(m, open(os.path.join(OUT, "perceptual_metrics.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in m.items() if k not in ("vowels", "args", "sibilants")}, indent=1))
print("vowels:", [(r["ph"], r["heard"], r["F"]) for r in m["vowels"]])
print("sibilants:", m["sibilants"])
