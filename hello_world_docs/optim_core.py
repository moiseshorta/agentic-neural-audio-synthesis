"""Batched spectral-matching loss for decoder-only latent optimisation (MPS)."""
import numpy as np
import torch
import librosa
from samesdec import SR, HOP
from phonetics import target_power

DEV = torch.device("mps")
RES = [(2048, 256, 128), (512, 128, 64)]  # (n_fft, hop, n_mels): harmonics/formants + timing
HARM_MAX = 4000  # Hz, top of the linear-frequency harmonic loss band (iterations <= 5 used 2500)


def make_targets(track_list, N):
    """track_list(frame_t) -> list of P track dicts. Returns per-resolution magnitude-mel targets (P, mels, frames)."""
    out = []
    for n_fft, hop, n_mels in RES:
        ft = np.arange(1 + N // hop) * hop / SR
        fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmax=16000)
        mags = [np.sqrt(target_power(tr, n_fft)).T for tr in track_list(ft)]  # (bins, frames) each
        A = np.stack([fb @ m for m in mags])  # magnitude mel
        At = torch.tensor(A, dtype=torch.float32, device=DEV)
        band = (np.fft.rfftfreq(n_fft, 1 / SR) >= 60) & (np.fft.rfftfreq(n_fft, 1 / SR) <= HARM_MAX)  # resolved harmonics
        L = torch.tensor(np.stack([m[band] for m in mags]), dtype=torch.float32, device=DEV)
        out.append(dict(n_fft=n_fft, hop=hop, A=A, At=At, peak=At.amax((1, 2)), L=L, Lpeak=L.amax((1, 2)),
                        band=torch.tensor(band, device=DEV),
                        fb=torch.tensor(fb, device=DEV), win=torch.hann_window(n_fft, device=DEV)))
    return out


def mag_mel(y, tg):
    """y: (B, 2, N) -> (B, 2, mels, frames) mel of STFT magnitude, and (B, 2, band bins, frames) linear magnitude."""
    X = torch.stft(y.reshape(-1, y.shape[-1]), tg["n_fft"], tg["hop"], window=tg["win"], center=True, return_complex=True)
    mag = torch.sqrt(X.real ** 2 + X.imag ** 2 + 1e-12)
    B = y.shape[0]
    return (tg["fb"] @ mag).reshape(B, 2, tg["fb"].shape[0], -1), mag[:, tg["band"]].reshape(B, 2, -1, mag.shape[-1])


def spec_loss(y, targets, idx, floor_db=-70.0, w_sc=0.0, w_harm=0.0):
    """Per-candidate loss (B,), plus per-frame dB error at -70 dB floor on the finest-frequency resolution.

    idx: (B,) long tensor, which target each candidate is matched to.
    Loss = mean |20 log10 (M + fl) - 20 log10 (A + fl)|  (+ w_sc * spectral convergence), averaged over resolutions.
    """
    tot = 0.0
    for r, tg in enumerate(targets):
        M, Lm = mag_mel(y, tg)
        A = tg["At"][idx][:, None]  # (B,1,mels,frames)
        if r == 0 and (torch.is_tensor(w_harm) or w_harm):  # harmonic fine structure on a linear axis
            Lt = tg["L"][idx][:, None]
            flh = (tg["Lpeak"][idx] * 10 ** (min(floor_db, -55) / 20))[:, None, None, None]
            tot = tot + w_harm * 20 * (torch.log10(Lm + flh) - torch.log10(Lt + flh)).abs().mean((1, 2, 3))
        fl = (tg["peak"][idx] * 10 ** (floor_db / 20))[:, None, None, None]
        d = 20 * (torch.log10(M + fl) - torch.log10(A + fl)).abs()
        if "w" in tg:  # per-frame emphasis (e.g. stop bursts are short but carry the consonant); (F,) or (P, F)
            wt = tg["w"][idx] if tg["w"].dim() == 2 else tg["w"][None]
            tot = tot + (d.mean((1, 2)) * wt).sum(-1) / wt.sum(-1)
        else:
            tot = tot + d.mean((1, 2, 3))
        if w_sc:
            tot = tot + w_sc * 10 * ((M - A).pow(2).sum((1, 2, 3)).sqrt() / (A.pow(2).sum((1, 2, 3)).sqrt() * np.sqrt(2)))
        if r == 0:
            with torch.no_grad():
                fl70 = (tg["peak"][idx] * 10 ** (-70 / 20))[:, None, None, None]
                per_frame = (20 * (torch.log10(M + fl70) - torch.log10(A + fl70)).abs()).mean((1, 2))  # (B, frames)
    return tot / len(targets), per_frame


def frame_scores(per_frame, T, hop=RES[0][1]):
    """Average per-STFT-frame error into per-latent-frame scores (B, T)."""
    B, F = per_frame.shape
    k = torch.clamp(torch.arange(F, device=per_frame.device) * hop // HOP, max=T - 1)
    s = torch.zeros(B, T, device=per_frame.device).index_add_(1, k, per_frame)
    return s / torch.bincount(k, minlength=T).clamp_min(1)


def periodicity_frames(tr, frame_t, min_av=-6.0, W=1024, guard_s=0.03):
    """Voiced analysis frames (centre sample, period in samples) from a parameter track dict.

    Only where voicing is established: av > min_av at the frame and within +-guard_s (no onsets/offsets)."""
    dt = frame_t[1] - frame_t[0]
    g = int(round(guard_s / dt))
    ok = tr["av"] > min_av
    keep = np.array([ok[max(0, j - g): j + g + 1].all() for j in range(len(ok))])
    c = np.round(frame_t[keep] * SR).astype(int)
    p = SR / tr["f0"][keep]
    return dict(c=torch.tensor(c, device=DEV), p=torch.tensor(p, dtype=torch.float32, device=DEV), W=W)


def _lag_corr(x, n, lag):
    """Normalised correlation between x[n] and x[n + lag] (fractional lag, linear interp). n: (J, W), lag: (J,)."""
    N = x.shape[-1]
    sh = n.float() + lag[:, None]
    i0 = sh.floor().long()
    fr = sh - i0
    a = x[:, n.clamp(0, N - 2)]
    i0 = i0.clamp(0, N - 2)
    b = x[:, i0] * (1 - fr) + x[:, i0 + 1] * fr
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)  # (B, J)


def periodicity_loss(y, pf, r_target=0.97, patch=256):
    """Hinge on normalised autocorrelation at the target period, plus two cheat guards.

    r(P) >= r_target: periodic voice (0.97 ~ HNR 15 dB, natural; 1.0 would sound buzzy).
    r(P/2) <= 0.8: no octave-up solution (a period-P/2 signal also has r(P) = 1).
    r(256) <= 0.85 unless P ~ 256: the decoder can emit identical 256-sample patches = a 172.27 Hz
    buzz that is trivially periodic; seen in iteration 9."""
    x = y.mean(1)  # (B, N) mono
    W = pf["W"]
    n = pf["c"][:, None] - W // 2 + torch.arange(W, device=DEV)[None]  # (J, W)
    r = _lag_corr(x, n, pf["p"])
    r_half = _lag_corr(x, n, pf["p"] / 2)
    r_patch = _lag_corr(x, n, torch.full_like(pf["p"], float(patch)))
    far = (pf["p"] - patch).abs() > 25
    hinge = torch.relu(r_target - r) + torch.relu(r_half - 0.8) + far * torch.relu(r_patch - 0.85)
    return hinge.mean(-1), r.detach()


def silence_frames(tr, frame_t, guard_s=0.03):
    """10 ms frames that are deep silence in the plan (no voicing/noise within +-guard_s)."""
    dt = frame_t[1] - frame_t[0]
    g = int(round(guard_s / dt))
    quiet = (tr["av"] < -80) & (tr["af"] < -80)
    keep = np.array([quiet[max(0, j - g): j + g + 1].all() for j in range(len(quiet))])
    return torch.tensor(np.round(frame_t[keep] * SR).astype(int), device=DEV)


def silence_loss(y, starts, thr_dbfs, L=441):
    """Mean dB by which decoded 10 ms RMS exceeds thr_dbfs inside planned silences (clicks, buzz)."""
    x = y.mean(1)
    n = (starts[:, None] + torch.arange(L, device=DEV)[None]).clamp(0, x.shape[-1] - 1)
    rms_db = 10 * torch.log10(x[:, n].pow(2).mean(-1) + 1e-12)
    return torch.relu(rms_db - thr_dbfs).mean(-1)


def noise_frames(tr, frame_t, W=1024, guard_s=0.012):
    """Centres of frames that should be pure noise (fricatives, breath): noise on, voicing off, +-guard_s."""
    dt = frame_t[1] - frame_t[0]
    g = int(round(guard_s / dt))
    ok = (tr["af"] > -40) & (tr["av"] < -60)
    keep = np.array([ok[max(0, j - g): j + g + 1].all() for j in range(len(ok))])
    return dict(c=torch.tensor(np.round(frame_t[keep] * SR).astype(int), device=DEV), W=W)


def aperiodicity_loss(y, nf, lags=(256, 512), tol=0.1):
    """Fricatives must not repeat at the decoder's patch period (256 samples = 172.27 Hz) or its multiple."""
    x = y.mean(1)
    n = nf["c"][:, None] - nf["W"] // 2 + torch.arange(nf["W"], device=DEV)[None]
    tot = 0.0
    for lag in lags:
        r = _lag_corr(x, n, torch.full((n.shape[0],), float(lag), device=DEV))
        tot = tot + torch.relu(r.abs() - tol).mean(-1)
    return tot
