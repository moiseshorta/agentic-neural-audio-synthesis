"""Stage 12: local latent surgery for consonants the global optimisation cannot reach (/k/ burst, /t/ closure).

Decoder dependency analysis (chunks of 2 frames, second pass shifted by one): decoding a 6-frame
window that starts at an even frame s reproduces the full-context output of frames s+2, s+3 exactly
(measured max diff ~1e-6) and of s+1, s+4 except their outer 256-sample patch (the final k=3 conv).
So frames s+2, s+3 can be re-searched with many random restarts in one cheap MPS batch, scored
only on that exact region, and the winner spliced back.
"""
import os, argparse, json, time
import numpy as np, torch
from samesdec import *
import phonetics as P_
from phonetics import timeline, tracks
from optim_core import DEV, make_targets, spec_loss, periodicity_frames, periodicity_loss, noise_frames, aperiodicity_loss

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="latents.npy[:cand]")
ap.add_argument("--f0_curve", required=True)
ap.add_argument("--tag", default="12_local_surgery")
ap.add_argument("--utt", default="UTT2")
ap.add_argument("--sites", default="10,16,18", help="even window starts; frames s+2,s+3 are re-searched")
ap.add_argument("--B", type=int, default=48)
ap.add_argument("--win", type=int, default=6, help="window length (even); scored region = frames s+1..s+win-2")
ap.add_argument("--free", default="2,3", help="offsets inside the window that are re-searched")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--formant_scale", type=float, default=1.0)
ap.add_argument("--f_ap", type=float, default=4500)
args = ap.parse_args()
OUT = os.path.join(DOCS, args.tag)
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
core = load_decoder(DEV)
utt = getattr(P_, args.utt)
path, _, cand = args.init.partition(":")
Z = np.load(path)
Z = Z[int(cand)] if cand else Z
f0c = np.load(args.f0_curve)
kw = dict(utt=utt, onset_ms=35, offset_ms=80, f_ap=args.f_ap, room_db=-50, f0_curve=f0c, formant_scale=args.formant_scale)
tl = timeline(utt)
report = []
for s in map(int, args.sites.split(",")):
    W = args.win
    fr = [int(f) for f in args.free.split(",")]
    t0 = (s + 1) * HOP / SR  # scored region = frames s+1..s+W-2
    targets = make_targets(lambda ft: [tracks(ft + t0, **kw)], (W - 2) * HOP)
    for tg in targets:  # same burst/closure emphasis as the global run
        ft = t0 + np.arange(tg["At"].shape[-1]) * tg["hop"] / SR
        w = np.ones_like(ft)
        for a, b, ph, g, f in tl:
            m = (ft >= a - 0.005) & (ft < b + 0.005)
            w[m] = np.maximum(w[m], 5 if ph.endswith("rel") else 2.2 if ph.endswith("cl") else 1)
        tg["w"] = torch.tensor(w, dtype=torch.float32, device=DEV)
    pt = t0 + np.arange(0, (W - 2) * HOP / SR, 0.01)
    pf = periodicity_frames(tracks(pt, utt=utt, jitter=0, onset_ms=35, offset_ms=80, f0_curve=f0c), pt)
    pf["c"] = pf["c"] - int(round(t0 * SR))  # region coordinates
    nf = noise_frames(tracks(pt, utt=utt, jitter=0, onset_ms=35, offset_ms=80), pt)
    nf["c"] = nf["c"] - int(round(t0 * SR))
    win = torch.tensor(Z[:, s:s + W], dtype=torch.float32, device=DEV)[None].repeat(args.B, 1, 1)
    free0 = win[:, :, fr[0]:fr[-1] + 1].clone()
    k = args.B // 3  # restarts: small, medium and large perturbations of the current frames
    free0[1:k] += 0.2 * torch.randn_like(free0[1:k])
    free0[k:2 * k] += 0.5 * torch.randn_like(free0[k:2 * k])
    free0[2 * k:] = free0[2 * k:] * 0.5 + 0.6 * torch.randn_like(free0[2 * k:])
    free = free0.clone().requires_grad_(True)
    opt = torch.optim.Adam([free], lr=0.01)
    idx = torch.zeros(args.B, dtype=torch.long, device=DEV)

    def region(zf, noise=0.0):
        b = zf.shape[0]
        zz = torch.cat([win[:b, :, :fr[0]], zf, win[:b, :, fr[-1] + 1:]], 2)
        return decode(core, zz + noise * torch.randn_like(zz))[:, :, HOP:(W - 1) * HOP]

    def score(y):
        l, _ = spec_loss(y, targets, idx[:y.shape[0]], w_harm=3.0)
        lp, _ = periodicity_loss(y, pf) if len(pf["c"]) else (torch.zeros_like(l), None)
        la = aperiodicity_loss(y, nf) if len(nf["c"]) else 0.0
        return l + 300 * lp + 100 * la

    with torch.no_grad():
        before = score(region(win[:1, :, fr[0]:fr[-1] + 1]))[0].item()
    for step in range(args.steps):
        loss = score(region(free, 0.07))
        opt.zero_grad(); loss.sum().backward(); opt.step()
    with torch.no_grad():
        final = score(region(free))
    j = int(final.argmin())
    after = final[j].item()
    if after < before:
        Z[:, s + fr[0]:s + fr[-1] + 1] = free[j].detach().cpu().numpy()
    covered = [ph for a, b, ph, g, f in tl if b > t0 and a < t0 + (W - 2) * HOP / SR]
    report.append(dict(window_start=s, frames=[s + f for f in fr], phones=covered, loss_before=round(before, 2),
                       loss_after=round(after, 2), winner=j, winner_group=["small", "medium", "random"][min(j // k, 2)],
                       accepted=after < before))
    print(report[-1], flush=True)
    del targets, free, opt
    torch.mps.empty_cache()

np.save(os.path.join(OUT, "latents.npy"), Z)
json.dump(report, open(os.path.join(OUT, "surgery_report.json"), "w"), indent=1)
with torch.no_grad():
    y = decode(core, torch.tensor(Z, dtype=torch.float32, device=DEV))[0].cpu().numpy()
save_wav(os.path.join(OUT, "hola_mundo.wav"), y / np.abs(y).max() * 0.89)
spectrogram_png(os.path.join(OUT, "spectrogram.png"), y, title=args.tag, annotations=[(a, b, ph) for a, b, ph, g, f in tl])
