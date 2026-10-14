"""Stage 3: full utterance. Codebook-initialised population, segment crossover, noise-robust latents.

- init: each latent frame = time-overlap-weighted mix of the phone codebook latents (02_phoneme_codebook)
- population of P candidates optimised together in one MPS batch
- every K steps: per-latent-frame scores -> splice the best frames of all candidates into an elite
  (the decoder only looks ~1 frame left/right, so splicing is nearly lossless), refill the worst half
- noise-robust objective: decode(z + sigma*eps). The decoder was trained with noise on its latents
  ("softnorm decode (noise disabled)"), so robustness pulls latents toward its training manifold.
"""
import os, argparse, json, time
import numpy as np, torch, librosa
import matplotlib.pyplot as plt
from samesdec import *
import phonetics as P_
from phonetics import timeline, tracks
from optim_core import DEV, make_targets, spec_loss, frame_scores, periodicity_frames, periodicity_loss, silence_frames, silence_loss, noise_frames, aperiodicity_loss
from analysis import analyze

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="03_utterance")
ap.add_argument("--utt", default="UTT", help="name of the utterance table in phonetics.py")
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--pop", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.007)
ap.add_argument("--sigma", type=float, default=0.05, help="latent noise for the robust objective")
ap.add_argument("--xover", type=int, default=150, help="crossover period (steps)")
ap.add_argument("--init", default="codebook", help="codebook | pitchbook | path/to/latents.npy[:cand]")
ap.add_argument("--book", default="04_pitch_codebook", help="pitch vocabulary folder for --init pitchbook")
ap.add_argument("--f0_scale", type=float, default=1.0)
ap.add_argument("--formant_scale", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--floor0", type=float, default=-30, help="starting loss floor (dB); -70 = no coarse-to-fine curriculum")
ap.add_argument("--w_per", type=float, default=0.0, help="weight of the periodicity (autocorrelation) hinge loss")
ap.add_argument("--w_sil", type=float, default=0.0, help="weight of the silence guard (dB above room tone in pauses)")
ap.add_argument("--pull", type=float, default=0.0, help="re-pitch: fraction of the way back to the planned F0")
ap.add_argument("--w_aper", type=float, default=0.0, help="weight of the fricative aperiodicity guard (patch-period buzz)")
ap.add_argument("--w_burst", type=float, default=1.0, help="loss weight of stop-burst frames (closures get sqrt)")
ap.add_argument("--repitch", action="store_true", help="adopt the init latents' own decoded F0 into the target")
ap.add_argument("--sweep", default="", help="per-candidate 'sigma:w_harm,...' (A/B inside one batch, disables crossover)")
ap.add_argument("--onset", type=float, default=35, help="voice onset ramp after pauses (ms), 0=off")
ap.add_argument("--offset", type=float, default=80, help="breathy offset ramp before pauses (ms), 0=off")
ap.add_argument("--f_ap", type=float, default=4500, help="aperiodicity crossover (Hz), 0=perfectly periodic voice")
ap.add_argument("--room", type=float, default=0, help="room-tone level re speech (dB), 0=off (digital silence)")
ap.add_argument("--w_harm", type=float, default=0.0, help="weight of the 60-2500 Hz harmonic fine-structure loss")
args = ap.parse_args()

OUT = os.path.join(DOCS, args.tag)
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(args.seed)
core = load_decoder(DEV)
utt = getattr(P_, args.utt)
tl = timeline(utt)
T = int(np.ceil(tl[-1][1] * SR / HOP)); T += T % 2
N = T * HOP
f0_curve = None
if args.repitch:  # analysis-by-synthesis: measure the pitch the decoder actually makes, let the target follow it
    from analysis import decoded_f0_curve
    with torch.no_grad():
        path, _, cand = args.init.partition(":")
        zi = np.load(path)
        zi = zi[int(cand)] if cand else zi
        y_init = decode(core, torch.tensor(zi, dtype=torch.float32, device=DEV))[0].cpu().numpy()
    f0_curve = decoded_f0_curve(y_init, utt=utt, pull=args.pull, f0_scale=args.f0_scale)
    np.save(os.path.join(OUT, "f0_curve.npy"), np.stack(f0_curve))
targets = make_targets(lambda ft: [tracks(ft, utt=utt, f0_scale=args.f0_scale, formant_scale=args.formant_scale, seed=args.seed,
                                          onset_ms=args.onset, offset_ms=args.offset, f_ap=args.f_ap or None,
                                          room_db=args.room or None, f0_curve=f0_curve)], N)
if args.w_burst != 1:  # emphasise stop bursts (x w_burst) and closures (x sqrt) in the spectral loss
    for tg in targets:
        ft = np.arange(tg["At"].shape[-1]) * tg["hop"] / SR
        w = np.ones_like(ft)
        for a, b, ph, g, f in tl:
            m = (ft >= a - 0.005) & (ft < b + 0.005)
            w[m] = np.maximum(w[m], args.w_burst if ph.endswith("rel") else np.sqrt(args.w_burst) if ph.endswith("cl") else 1)
        tg["w"] = torch.tensor(w, dtype=torch.float32, device=DEV)
pt = np.arange(0, N / SR, 0.01)  # periodicity analysis every 10 ms on voiced target frames
per_frames = periodicity_frames(tracks(pt, utt=utt, f0_scale=args.f0_scale, jitter=0, onset_ms=args.onset,
                                       offset_ms=args.offset, f0_curve=f0_curve), pt)
nz_frames = noise_frames(tracks(pt, utt=utt, jitter=0, onset_ms=args.onset, offset_ms=args.offset), pt)
sil_starts = silence_frames(tracks(pt, utt=utt, jitter=0, onset_ms=args.onset, offset_ms=args.offset), pt)
sil_thr = -18 + (args.room or -60) + 8  # dBFS: room tone level (re the -18 dBFS speech reference) + 8 dB
idx = torch.zeros(args.pop, dtype=torch.long, device=DEV)
sil = np.load(os.path.join(DOCS, "00_exploration", "silence_latent.npy"))

if args.init == "codebook":
    book = dict(np.load(os.path.join(DOCS, "02_phoneme_codebook", "codebook.npz")))
    book["sil"] = sil
    alias = {"kcl": "sil", "tcl": "sil", "dcl": "m", "drel": "n", "krel": "h", "trel": "s"}
    z_init = np.zeros((256, T))
    for k in range(T):
        a0, a1 = k * HOP / SR, (k + 1) * HOP / SR
        w = {}
        for a, b, ph, g, f in tl:
            ov = max(0.0, min(b, a1) - max(a, a0))
            if ov > 0:
                key = alias.get(ph, ph)
                w[key] = w.get(key, 0) + ov
        if not w:
            w = {"sil": 1.0}
        z_init[:, k] = sum(book[p] * x for p, x in w.items()) / sum(w.values())
elif args.init == "pitchbook":  # pitch-aware vocabulary: interpolate in log F0 between the two nearest levels
    pb = dict(np.load(os.path.join(DOCS, args.book, "pitch_codebook.npz")))
    F0S = sorted({int(k.split("@")[1]) for k in pb if "@" in k})
    alias = {"kcl": "sil", "tcl": "sil", "dcl": "m", "drel": "n", "krel": "h", "trel": "s"}
    ft = np.arange(T * 64) * (HOP / 64) / SR
    f0t = tracks(ft, utt=utt, jitter=0)["f0"] * args.f0_scale

    def latent(ph, f0):
        if ph == "sil":
            return sil
        if ph in ("s", "h"):
            return pb[ph]
        lf = np.clip(np.log(f0), np.log(F0S[0]), np.log(F0S[-1]))
        j = int(np.clip(np.searchsorted(np.log(F0S), lf) - 1, 0, len(F0S) - 2))
        w = (lf - np.log(F0S[j])) / (np.log(F0S[j + 1]) - np.log(F0S[j]))
        return (1 - w) * pb[f"{ph}@{F0S[j]}"] + w * pb[f"{ph}@{F0S[j + 1]}"]

    z_init = np.zeros((256, T))
    for k in range(T):
        a0, a1 = k * HOP / SR, (k + 1) * HOP / SR
        acc, wsum = np.zeros(256), 0.0
        for a, b, ph, g, f in tl:
            ov = max(0.0, min(b, a1) - max(a, a0))
            if ov > 0:
                f0 = float(np.interp((max(a, a0) + min(b, a1)) / 2, ft, f0t))
                acc += ov * latent(alias.get(ph, ph), f0); wsum += ov
        z_init[:, k] = acc / wsum if wsum else sil
else:
    path, _, cand = args.init.partition(":")  # 'latents_population.npy:2' picks one candidate
    z_init = np.load(path)
    z_init = z_init[int(cand)] if cand else z_init
if args.sweep:
    cfg = [tuple(map(float, (c + ":0").split(":")[:3])) for c in args.sweep.split(",")]  # sigma:w_harm[:w_per]
    args.pop, args.xover = len(cfg), 0
    idx = torch.zeros(args.pop, dtype=torch.long, device=DEV)
    sig_c = torch.tensor([c[0] for c in cfg], device=DEV)
    wh_c = torch.tensor([c[1] for c in cfg], device=DEV)
    wp_c = torch.tensor([c[2] for c in cfg], device=DEV)
else:
    sig_c = torch.full((args.pop,), args.sigma, device=DEV)
    wh_c = torch.full((args.pop,), args.w_harm, device=DEV)
    wp_c = torch.full((args.pop,), args.w_per, device=DEV)
z0 = torch.tensor(z_init, dtype=torch.float32, device=DEV)[None].repeat(args.pop, 1, 1)
if not args.sweep:
    z0[1:] += 0.1 * torch.randn_like(z0[1:])
z = z0.clone().requires_grad_(True)


def new_opt(lr):
    return torch.optim.Adam([z], lr=lr)


def lr_at(s):
    return args.lr * min(1, (s + 1) / 30) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * s / args.steps)))


opt = new_opt(args.lr)
hist, t0 = [], time.time()
for step in range(args.steps):
    prog = min(1.0, step / (0.5 * args.steps))
    floor_db = args.floor0 + (-70 - args.floor0) * prog
    w_sc = (1.0 - prog) * (args.floor0 > -70)
    sig = sig_c * (1 - 0.5 * step / args.steps)
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    y = decode(core, z + sig[:, None, None] * torch.randn_like(z))
    loss, pf = spec_loss(y, targets, idx, floor_db=floor_db, w_sc=w_sc, w_harm=wh_c)
    if wp_c.any():
        lp, r_now = periodicity_loss(y, per_frames)
        loss = loss + wp_c * 100 * lp  # 100 x (hinge) ~ dB scale of the spectral terms
    if args.w_sil:
        loss = loss + args.w_sil * silence_loss(y, sil_starts, sil_thr)
    if args.w_aper and len(nz_frames["c"]):
        loss = loss + args.w_aper * 100 * aperiodicity_loss(y, nz_frames)
    opt.zero_grad(); loss.sum().backward(); opt.step()
    hist.append(pf.mean(1).detach())  # stays on GPU: no per-step MPS sync

    if args.xover and step > 0 and step % args.xover == 0 and step < 0.85 * args.steps:
        with torch.no_grad():
            _, pf0 = spec_loss(decode(core, z), targets, idx)
            sc = frame_scores(pf0, T)
            sc = torch.nn.functional.avg_pool1d(sc[:, None], 3, 1, 1, count_include_pad=False)[:, 0]  # neighbour-aware
            pick = sc.argmin(0)  # best candidate per latent frame
            elite = z[pick, :, torch.arange(T, device=DEV)].T  # (256, T)
            _, pfe = spec_loss(decode(core, elite[None]), targets, idx[:1])
            tot = pf0.mean(1)
            order = tot.argsort()
            src = elite if pfe.mean() < tot[order[0]] else z.data[order[0]].clone()
            z.data[order[args.pop // 2:]] = src[None] + 0.03 * torch.randn_like(z.data[order[args.pop // 2:]])
            z.data[order[-1]] = src
            msg = f"xover: elite {pfe.mean().item():.2f} vs best {tot[order[0]].item():.2f} (frames from {pick.unique().numel()} candidates)"
        opt = new_opt(lr_at(step))
        print(msg, flush=True)
    if step % 50 == 0 or step == args.steps - 1:
        h = hist[-1].cpu().numpy()
        rr = f" r(median) {r_now.median(-1).values.cpu().numpy().round(3)}" if wp_c.any() else ""
        print(f"{step:4d} floor {floor_db:5.1f} sig {sig.max().item():.3f}  -70dB logmel best {h.min():5.2f} mean {h.mean():5.2f}{rr}  {time.time() - t0:.0f}s", flush=True)

# ---- final: noise-free ranking, save the best
with torch.no_grad():
    yall = decode(core, z)
    _, pf = spec_loss(yall, targets, idx)
tot = pf.mean(1).cpu().numpy()
best = int(tot.argmin())
zall = z.detach().cpu().numpy()
np.save(os.path.join(OUT, "latents_population.npy"), zall)
zf = zall[best]
np.save(os.path.join(OUT, "latents.npy"), zf)
y = yall[best].cpu().numpy()
save_wav(os.path.join(OUT, "hola_mundo.wav"), y / np.abs(y).max() * 0.89)
ann = [(a, b, ph) for a, b, ph, g, f in tl]
spectrogram_png(os.path.join(OUT, "spectrogram.png"), y, title=f"{args.tag} (decoded, candidate {best})", f_max=11025, annotations=ann)

tg = targets[0]
Mo = librosa.feature.melspectrogram(y=y.mean(0), sr=SR, n_fft=tg["n_fft"], hop_length=tg["hop"], n_mels=128, fmax=16000, power=1)
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a_, M_, ttl in [(ax[0], tg["A"][0], "analytic phonetic target (never rendered to audio)"), (ax[1], Mo, "decoded audio")]:
    D = 20 * np.log10(M_ + M_.max() * 10 ** -3.5)
    a_.imshow(D, origin="lower", aspect="auto", cmap="magma", extent=[0, N / SR, 0, 128], vmax=D.max(), vmin=D.max() - 70)
    a_.set_title(ttl); a_.set_ylabel("mel bin")
    for a, b, ph in ann:
        a_.axvline(a, color="cyan", lw=0.4); a_.text((a + b) / 2, 122, ph, color="cyan", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "target_vs_decoded_mel.png"), dpi=100); plt.close(fig)

hist = torch.stack(hist).cpu().numpy()
fig, ax = plt.subplots(1, 4, figsize=(20, 4.5), gridspec_kw={"width_ratios": [3, 1.3, 1.3, 1.3]})
ax[0].imshow(zf, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3); ax[0].set_title("latents (256 x T)"); ax[0].set_xlabel("frame (92.9 ms)")
ax[1].plot(np.linalg.norm(zf, axis=0), "o-", label="optimised"); ax[1].plot(np.linalg.norm(z_init, axis=0), ".--", label="codebook init")
ax[1].axhline(np.linalg.norm(sil), color="gray", ls=":"); ax[1].legend(); ax[1].set_title("frame L2 norm")
zn = zf / np.linalg.norm(zf, axis=0, keepdims=True)
ax[2].imshow(zn.T @ zn, cmap="viridis", vmin=-1, vmax=1); ax[2].set_title("frame cosine similarity")
ax[3].plot(hist); ax[3].set_title("-70 dB log-mel error per candidate"); ax[3].set_xlabel("step")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "latent_analysis.png"), dpi=100); plt.close(fig)
np.savetxt(os.path.join(OUT, "loss_curve.csv"), hist, delimiter=",", fmt="%.4f", header="logmel_db per candidate")

if args.sweep:  # keep every A/B candidate audible + measured
    sweep_rows = []
    for c, (sg, wh, wp) in enumerate(cfg):
        yc = yall[c].cpu().numpy()
        name = f"cand{c}_sigma{sg:g}_wharm{wh:g}_wper{cfg[c][2]:g}"
        save_wav(os.path.join(OUT, name + ".wav"), yc / np.abs(yc).max() * 0.89)
        mc = analyze(yc, os.path.join(OUT, name), utt=utt, title=name, f0_scale=args.f0_scale, formant_scale=args.formant_scale)
        sweep_rows.append(dict(candidate=c, sigma=sg, w_harm=wh, w_per=wp, logmel_db=float(tot[c]), **{k: v for k, v in mc.items() if k not in ("vowels", "sibilants")}))
        print(name, json.dumps({k: round(v, 2) for k, v in sweep_rows[-1].items() if isinstance(v, float)}))
    json.dump(sweep_rows, open(os.path.join(OUT, "sweep_summary.json"), "w"), indent=1)
m = analyze(y, os.path.join(OUT, "perceptual"), utt=utt, f0_scale=args.f0_scale, formant_scale=args.formant_scale, title=args.tag)
m.update(final_logmel_db=float(tot[best]), population_final_db=tot.round(2).tolist(), frames=T, seconds=N / SR, args=vars(args),
         side_ratio=float(np.sqrt(((y[0] - y[1]) ** 2).mean() / ((y[0] + y[1]) ** 2).mean())),
         latent_std=float(zf.std()), frame_norm_mean=float(np.linalg.norm(zf, axis=0).mean()),
         init_to_final_dist=float(np.linalg.norm(zf - z_init) / np.linalg.norm(z_init)))
json.dump(m, open(os.path.join(OUT, "perceptual_metrics.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in m.items() if k not in ("vowels", "args", "sibilants")}, indent=1))
print("vowels:", [(r["ph"], r["heard"], r["F"]) for r in m["vowels"]])
print("sibilants:", m["sibilants"])
