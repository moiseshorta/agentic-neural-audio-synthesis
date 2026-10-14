"""Perceptual / phonetic analysis of a decoded utterance against the phonetic plan."""
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from samesdec import SR
from phonetics import timeline, tracks, V, UTT

VOWELS = set(V)


def bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def lpc_formants(x, sr=11025, order=12):
    x = np.append(x[0], x[1:] - 0.97 * x[:-1]) * np.hamming(len(x))
    a = librosa.lpc(x, order=order)
    r = np.roots(a)
    r = r[np.imag(r) > 0]
    f = np.angle(r) * sr / (2 * np.pi)
    bw = -sr / np.pi * np.log(np.abs(r))
    keep = (f > 150) & (bw < 500)
    return np.sort(f[keep])[:3]


def analyze(y, out_prefix, utt=UTT, f0_scale=1.0, formant_scale=1.0, title=""):
    mono = y.mean(0) if y.ndim == 2 else y
    tl = timeline(utt)
    hop = 256
    f0, vflag, _ = librosa.pyin(mono, fmin=60, fmax=600, sr=SR, frame_length=2048, hop_length=hop)
    t = librosa.times_like(f0, sr=SR, hop_length=hop)
    tr = tracks(t, utt, f0_scale=f0_scale, formant_scale=formant_scale, jitter=0)
    tv = tr["av"] > -20
    both = tv & vflag
    cents = 1200 * np.log2(f0[both] / tr["f0"][both]) if both.any() else np.array([np.nan])
    m = {
        "f0_rmse_cents": float(np.sqrt(np.nanmean(cents ** 2))),
        "f0_octave_errors_pct": float(100 * np.mean(np.abs(cents) > 600)),
        "voicing_recall_pct": float(100 * vflag[tv].mean()),
        "voicing_false_alarm_pct": float(100 * vflag[~tv & (tr["af"] < -60)].mean()),
    }
    # vowels: LPC formants on the middle half of each vowel segment
    y11 = librosa.resample(mono, orig_sr=SR, target_sr=11025)
    vrows, correct = [], 0
    protos = {k: np.array(v[:2]) * formant_scale for k, v in V.items()}
    for a, b, ph, g, _ in tl:
        if ph not in VOWELS:
            continue
        c, h = (a + b) / 2, (b - a) / 4
        seg = y11[int((c - h) * 11025): int((c + h) * 11025)]
        fm = lpc_formants(seg)
        if len(fm) < 2:
            vrows.append(dict(ph=ph, t=round(c, 3), F=[float(x) for x in fm], heard="?"))
            continue
        heard = min(protos, key=lambda k: np.sum((bark(protos[k]) - bark(fm[:2])) ** 2))
        correct += heard == ph
        vrows.append(dict(ph=ph, t=round(c, 3), F=[round(float(x)) for x in fm], heard=heard))
    # harmonics-to-noise ratio (autocorrelation peak at the target period) on vowel nuclei
    for r, (a, b, ph, g, _) in zip(vrows, [x for x in tl if x[2] in VOWELS]):
        c, h = (a + b) / 2, (b - a) / 4
        seg = mono[int((c - h) * SR): int((c + h) * SR)]
        seg = seg - seg.mean()
        f0t = float(np.interp(c, t, tr["f0"]))
        lo, hi = int(SR / (f0t * 1.25)), int(SR / (f0t * 0.8))
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        ac = ac / (ac[0] + 1e-12) / (1 - np.arange(len(ac)) / len(ac))  # unbiased
        rmax = float(np.clip(ac[lo:hi].max(), 1e-4, 0.9999))
        r["hnr_db"] = round(10 * np.log10(rmax / (1 - rmax)), 1)
    m["hnr_db_median"] = float(np.median([r["hnr_db"] for r in vrows]))
    m["vowel_id_pct"] = 100 * correct / max(1, len(vrows))
    m["vowels"] = vrows
    # sibilants: centroid and high-band energy share
    S = np.abs(librosa.stft(mono, n_fft=1024, hop_length=hop)) ** 2
    fr = librosa.fft_frequencies(sr=SR, n_fft=1024)
    srow = []
    for a, b, ph, g, _ in tl:
        if ph == "s":
            cols = (t >= a + 0.015) & (t < b - 0.015)
            Ss = S[:, cols[: S.shape[1]]].mean(1)
            srow.append(dict(t=round((a + b) / 2, 3), centroid=round(float((fr * Ss).sum() / Ss.sum())),
                             hf_share=round(float(Ss[fr > 3000].sum() / Ss.sum()), 3)))
    m["sibilants"] = srow
    # loudness structure: silence vs speech
    e = 10 * np.log10(S.sum(0) + 1e-12)
    sil = tr["av"][: len(e)] < -80
    sil &= tr["af"][: len(e)] < -80
    m["speech_minus_silence_db"] = float(np.median(e[~sil]) - np.median(e[sil]))
    # F0 roughness: frame-to-frame relative F0 change inside voiced runs (natural speech ~0.5-2 %)
    ok = vflag[1:] & vflag[:-1]
    m["f0_roughness_pct"] = float(100 * np.median(np.abs(np.diff(np.log(f0)))[ok])) if ok.any() else float("nan")
    # warble at the latent frame rate (SR/4096 = 10.77 Hz) in the speech amplitude envelope
    env = librosa.feature.rms(y=mono, frame_length=1024, hop_length=220)[0]
    env = env[np.interp(np.arange(len(env)) * 220 / SR, t[: len(sil)], sil.astype(float)) < 0.5]
    E = np.abs(np.fft.rfft((env - env.mean()) * np.hanning(len(env)))) ** 2
    fm = np.fft.rfftfreq(len(env), 220 / SR)
    band = (fm > 2) & (fm < 30)
    m["frame_rate_warble_db"] = float(10 * np.log10(E[np.abs(fm - SR / 4096) < 0.8].mean() / E[band].mean()))
    json.dump(m, open(out_prefix + "_metrics.json", "w"), indent=1)

    # figure: output spectrogram + F0 overlay + vowel chart
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.05, 0.38, 0.7, 0.55])
    D = librosa.amplitude_to_db(np.sqrt(S), ref=np.max, top_db=80)
    ax.imshow(D, origin="lower", aspect="auto", extent=[0, len(mono) / SR, 0, SR / 2], cmap="magma")
    ax.set_ylim(0, 8000)
    for a, b, ph, g, _ in tl:
        ax.axvline(a, color="cyan", lw=0.4, alpha=0.5)
        ax.text((a + b) / 2, 7700, ph, color="cyan", ha="center", fontsize=9)
    ax.set_title(f"{title}  f0 rmse {m['f0_rmse_cents']:.0f}c  vowel-id {m['vowel_id_pct']:.0f}%  voicing {m['voicing_recall_pct']:.0f}%  HNR {m['hnr_db_median']:.0f} dB")
    ax2 = fig.add_axes([0.05, 0.06, 0.7, 0.26], sharex=ax)
    ax2.plot(t, np.where(tv, tr["f0"], np.nan), "c-", lw=3, alpha=0.5, label="target F0")
    ax2.plot(t, f0, "k.", ms=2, label="pyin F0 (decoded)")
    ax2.set_ylim(50, 400); ax2.set_ylabel("Hz"); ax2.legend(loc="upper left")
    ax3 = fig.add_axes([0.8, 0.38, 0.18, 0.55])
    for k, p in protos.items():
        ax3.plot(p[1], p[0], "o", ms=18, color="lightgray"); ax3.text(p[1], p[0], k, ha="center", va="center")
    for r in vrows:
        if len(r["F"]) >= 2:
            ax3.plot(r["F"][1], r["F"][0], "x", color="green" if r["heard"] == r["ph"] else "red")
            ax3.text(r["F"][1], r["F"][0], r["ph"], fontsize=7)
    ax3.set_xlim(2800, 500); ax3.set_ylim(1000, 150); ax3.set_xlabel("F2"); ax3.set_ylabel("F1")
    ax3.set_title("vowel chart (x = decoded)")
    fig.savefig(out_prefix + "_analysis.png", dpi=100)
    plt.close(fig)

    # waveform zooms (40 ms) at vowel nuclei: natural voice = glottal pulses + decaying formant ringing
    vt = [x for x in tl if x[2] in VOWELS]
    fig, ax = plt.subplots(2, (len(vt) + 1) // 2, figsize=(18, 5), sharey=True)
    for a_, (a, b, ph, g, _) in zip(ax.flat, vt):
        c = int((a + b) / 2 * SR)
        a_.plot(np.arange(-882, 882) / SR * 1000, mono[c - 882: c + 882], lw=0.6, color="k")
        a_.set_title(f"/{ph}/ @ {(a + b) / 2:.2f}s", fontsize=9)
    for a_ in ax.flat[len(vt):]:
        a_.axis("off")
    fig.tight_layout(); fig.savefig(out_prefix + "_waveforms.png", dpi=90); plt.close(fig)
    return m


def decoded_f0_curve(y, utt=UTT, max_dev_st=3.0, hop=256, pull=0.0, f0_scale=1.0):
    """F0 contour the decoder actually produced, refined on a harmonic comb, merged with the plan.

    Where the plan is voiced and the decoded pitch is within max_dev_st semitones of it, the decoded
    value wins; elsewhere the planned value is kept. Returns (times, f0)."""
    mono = y.mean(0) if y.ndim == 2 else y
    f0, vflag, _ = librosa.pyin(mono, fmin=60, fmax=500, sr=SR, frame_length=2048, hop_length=hop)
    t = librosa.times_like(f0, sr=SR, hop_length=hop)
    plan = tracks(t, utt, jitter=0, f0_scale=f0_scale)["f0"]
    X = np.abs(librosa.stft(mono, n_fft=4096, hop_length=hop))
    fr = np.fft.rfftfreq(4096, 1 / SR)
    logX = np.log(X + 1e-6)
    out = plan.copy()
    for j in np.where(vflag & ~np.isnan(f0))[0]:
        if abs(12 * np.log2(f0[j] / plan[j])) > max_dev_st:
            continue
        cand = f0[j] * np.linspace(0.97, 1.03, 61)  # refine to 0.1 %: harmonic comb up to 3 kHz
        k = np.arange(1, 40)
        fk = cand[:, None] * k[None]
        score = np.where(fk < 3000, np.interp(fk, fr, logX[:, min(j, X.shape[1] - 1)]), 0).sum(1)
        out[j] = cand[score.argmax()]
    ok = out != plan
    sm = out.copy()
    for j in np.where(ok)[0]:  # light median smoothing inside decoded-voiced runs
        w = out[max(0, j - 2): j + 3][ok[max(0, j - 2): j + 3]]
        sm[j] = np.median(w)
    sm = sm * (plan / sm) ** pull  # walk part of the way back toward the planned contour (continuation)
    return t, sm
