"""Hand-written Spanish phonetics -> analytic *spectral* target for latent optimisation.

No waveform is ever synthesised here. Each target frame is the *expected STFT power*
of a source-filter model, evaluated in closed form in the frequency domain:
  harmonics of F0 (glottal tilt) x cascade formant resonators (x nasal zero)  +  shaped noise.
The decoder is the only thing in this project that produces audio.
"""
import numpy as np
from samesdec import SR

# Spanish vowel formants F1..F5 (Hz), adult male (Quilis-style values)
V = {
    "a": (700, 1250, 2500, 3500, 4500),
    "e": (450, 1850, 2550, 3500, 4500),
    "i": (290, 2250, 3000, 3600, 4600),
    "o": (480, 880, 2450, 3400, 4500),
    "u": (320, 780, 2350, 3400, 4500),
}
BW_V = (80, 90, 130, 200, 260)
HIGHER_POLES = [(5500 + 1000 * k, 300 + 60 * k) for k in range(20)]  # Klatt-style higher-pole correction

# phone -> F (formants), B (bandwidths), av (voicing dB), af (noise dB), noise (shape), nz (nasal zero), tilt
PH = {k: dict(F=f, B=BW_V, av=0.0, af=-32.0, noise="asp") for k, f in V.items()}
PH.update({
    "l":    dict(F=(360, 1150, 2700, 3400, 4500), B=(80, 130, 160, 220, 300), av=-6.0, af=-40.0, noise="asp"),
    "m":    dict(F=(270, 1100, 2200, 3300, 4500), B=(60, 220, 320, 400, 500), av=-9.0, af=-60.0, noise="asp", nz=(900, 180), tilt=6.0),
    "n":    dict(F=(270, 1500, 2500, 3400, 4500), B=(60, 220, 320, 400, 500), av=-9.0, af=-60.0, noise="asp", nz=(1800, 250), tilt=6.0),
    "dcl":  dict(F=(250, 1500, 2500, 3400, 4500), B=(80, 300, 400, 500, 600), av=-24.0, af=-80.0, noise="asp", tilt=18.0),
    "drel": dict(F=(300, 1600, 2600, 3400, 4500), B=(100, 200, 300, 400, 500), av=-14.0, af=-16.0, noise=(4000, 4000)),
    "kcl":  dict(F=(450, 1200, 2400, 3400, 4500), B=BW_V, av=-90.0, af=-90.0, noise="asp"),
    "krel": dict(F=(480, 1100, 2450, 3400, 4500), B=(150, 200, 300, 400, 500), av=-90.0, af=-12.0, noise=(1700, 1300)),
    "tcl":  dict(F=(450, 1700, 2600, 3400, 4500), B=BW_V, av=-90.0, af=-90.0, noise="asp"),
    "trel": dict(F=(500, 1700, 2600, 3400, 4500), B=(150, 200, 300, 400, 500), av=-90.0, af=-10.0, noise=(5000, 5000)),
    "s":    dict(F=(450, 1700, 2600, 3400, 4500), B=BW_V, av=-90.0, af=-8.0, noise=(6500, 3500)),
    "sil":  dict(F=(500, 1500, 2500, 3500, 4500), B=BW_V, av=-90.0, af=-90.0, noise="asp"),
    "h":    dict(F=(600, 1300, 2500, 3500, 4500), B=(150, 200, 300, 400, 500), av=-90.0, af=-20.0, noise="asp"),  # breath
})

# "hola mundo, cómo estás hoy?"  ->  [ˈo.la ˈmun̪.d̪o | ˈko.mo es.ˈta.s ˈoi̯]   (h is silent in Spanish)
# (phone, duration ms, stress gain dB, [(pos in segment 0..1, F0 Hz), ...])
UTT = [
    ("sil", 260, 0, []),
    ("o", 115, 2, [(0.0, 122), (0.6, 140)]),
    ("l", 55, 0, [(0.5, 128)]),
    ("a", 125, 0, [(0.2, 120), (1.0, 108)]),
    ("m", 70, 0, [(0.5, 112)]),
    ("u", 85, 2, [(0.5, 134)]),
    ("n", 60, 0, [(0.5, 126)]),
    ("dcl", 45, 0, [(0.5, 120)]),
    ("drel", 12, 0, []),
    ("o", 140, -1, [(0.2, 117), (1.0, 124)]),
    ("sil", 210, 0, []),
    ("kcl", 55, 0, []),
    ("krel", 28, 0, []),
    ("o", 100, 3, [(0.3, 152), (1.0, 140)]),
    ("m", 62, 0, [(0.5, 132)]),
    ("o", 80, 0, [(0.5, 126)]),
    ("e", 72, -1, [(0.5, 121)]),
    ("s", 95, 0, []),
    ("tcl", 48, 0, []),
    ("trel", 16, 0, []),
    ("a", 120, 2, [(0.2, 138), (1.0, 128)]),
    ("s", 85, 0, []),
    ("o", 105, 1, [(0.2, 126), (1.0, 132)]),
    ("i", 150, 0, [(0.4, 150), (1.0, 178)]),
    ("sil", 260, 0, []),
]


def timeline(utt=UTT):
    t, out = 0.0, []
    for ph, d, g, f0 in utt:
        out.append((t, t + d / 1000, ph, g, f0))
        t += d / 1000
    return out


def _smooth(x, sigma_frames):
    if sigma_frames <= 0:
        return x
    k = np.arange(-int(4 * sigma_frames), int(4 * sigma_frames) + 1)
    k = np.exp(-0.5 * (k / sigma_frames) ** 2)
    k /= k.sum()
    pad = len(k) // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def tracks(frame_t, utt=UTT, f0_scale=1.0, formant_scale=1.0, seed=0, coartic_ms=18, jitter=0.012,
           onset_ms=0, offset_ms=0, f_ap=None, room_db=None, f0_curve=None):
    """Per-frame parameter tracks at times frame_t (s)."""
    tl = timeline(utt)
    n = len(frame_t)
    hop_s = frame_t[1] - frame_t[0]
    F = np.zeros((n, 5)); B = np.zeros((n, 5)); av = np.full(n, -90.0); af = np.full(n, -90.0)
    tilt = np.zeros(n); nzf = np.full(n, 0.0); seg = np.zeros(n, int)
    noise_c = np.zeros(n); noise_w = np.zeros(n); asp = np.zeros(n, bool)
    f0_pts = []
    for i, (a, b, ph, g, f0s) in enumerate(tl):
        m = (frame_t >= a) & (frame_t < b)
        p = PH[ph]
        F[m] = p["F"]; B[m] = p["B"]; av[m] = p["av"] + g; af[m] = p["af"] + g
        tilt[m] = p.get("tilt", 0.0); seg[m] = i
        if "nz" in p:
            nzf[m] = p["nz"][0]
        if p["noise"] == "asp":
            asp[m] = True
        else:
            noise_c[m], noise_w[m] = p["noise"]
        f0_pts += [(a + pos * (b - a), hz) for pos, hz in f0s]
    if (frame_t >= tl[-1][1]).any():  # trailing frames beyond the utterance = silence
        m = frame_t >= tl[-1][1]
        F[m] = PH["sil"]["F"]; B[m] = PH["sil"]["B"]; asp[m] = True; seg[m] = len(tl) - 1
    # natural voice onsets after a pause, and breathy decaying offsets into a pause
    for i, (a, b, ph, g, f0s) in enumerate(tl):
        if PH[ph]["av"] < -30:
            continue
        if onset_ms and i > 0 and tl[i - 1][2] == "sil":
            m = (frame_t >= a) & (frame_t < a + onset_ms / 1000)
            av[m] += 20 * np.log10(np.maximum(np.sin(np.pi / 2 * (frame_t[m] - a) / (onset_ms / 1000)) ** 2, 1e-4))
        if offset_ms and i + 1 < len(tl) and tl[i + 1][2] == "sil":
            m = (frame_t >= b - offset_ms / 1000) & (frame_t < b)
            av[m] += 20 * np.log10(np.maximum(np.sin(np.pi / 2 * (b - frame_t[m]) / (offset_ms / 1000)) ** 2, 1e-4))
            af[m] = np.maximum(af[m], av[m] - 14)  # breath takes over as voicing dies
    # coarticulation: smooth formants in log-frequency, amplitudes lightly
    sig = coartic_ms / 1000 / hop_s
    F = np.exp(_smooth(np.log(F), sig)) * formant_scale
    B = _smooth(B, sig)
    av = _smooth(av, 2.5 / 1000 / hop_s)
    af = _smooth(af, 1.5 / 1000 / hop_s)
    tilt = _smooth(tilt, sig / 2)
    nz_on = _smooth((nzf > 0).astype(float), sig / 2)
    nzf = np.where(nzf > 0, nzf, 1000.0)
    nzf = _smooth(nzf, sig / 2) * formant_scale
    # F0: smooth interpolation through anchors + slow "micro-prosody" jitter
    pt, ph = np.array(f0_pts).T
    f0 = np.exp(np.interp(frame_t, pt, np.log(ph)))
    rng = np.random.default_rng(seed)
    wob = _smooth(rng.standard_normal(n), 12 / 1000 / hop_s)
    f0 = f0 * (1 + jitter * wob / (wob.std() + 1e-9)) * f0_scale
    if f0_curve is not None:  # (times, f0) measured on a decoded take: overrides the planned contour
        f0 = np.interp(frame_t, f0_curve[0], f0_curve[1])
    return dict(t=frame_t, F=F, B=B, av=av, af=af, tilt=tilt, nzf=nzf, nz_on=nz_on, f0=f0,
                noise_c=noise_c, noise_w=noise_w, asp=asp, seg=seg, fs=formant_scale, f_ap=f_ap, room_db=room_db)


def _resonator_db(f, Fi, Bi):
    """|H| of a 2-pole resonator, unity gain at DC, in dB. f: (K,), Fi,Bi: scalars or (n,1)."""
    num = Fi ** 2 + (Bi / 2) ** 2
    den = np.sqrt(((f - Fi) ** 2 + (Bi / 2) ** 2) * ((f + Fi) ** 2 + (Bi / 2) ** 2))
    return 20 * np.log10(num / den)


def _window_kernel(n_fft, oversample=64, lobes=4):
    """|W(df)|^2 for a Hann window, df in Hz on a fine grid within +-lobes bins."""
    w = np.hanning(n_fft + 1)[:-1]  # periodic Hann (torch.hann_window default)
    grid = np.linspace(-lobes, lobes, 2 * lobes * oversample + 1)  # in bins
    n = np.arange(n_fft)
    Wd = np.abs(np.exp(-2j * np.pi * np.outer(grid / n_fft, n)) @ w) ** 2
    return grid * SR / n_fft, Wd, (w ** 2).sum()


def target_power(tr, n_fft, rms_db=-18.0):
    """Expected one-sided STFT power (frames, n_fft//2+1) for the parameter tracks."""
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    kdf, kW, w2 = _window_kernel(n_fft)
    n = len(tr["t"])
    P = np.zeros((n, len(freqs)))
    ref = 10 ** (rms_db / 20)
    if tr.get("room_db") is not None:  # pink above 100 Hz, rolled off below 60 Hz and above 10 kHz
        fr = np.maximum(freqs, 1)
        room = 1 / np.maximum(fr, 100) / (1 + (60 / fr) ** 4) / (1 + (fr / 10000) ** 4)
        P += (ref * 10 ** (tr["room_db"] / 20)) ** 2 * room / room.mean() * w2
    for j in range(n):
        Fj, Bj = tr["F"][j], tr["B"][j]
        # vocal-tract filter (dB) on the bin grid
        def vt_db(f):
            h = sum(_resonator_db(f, Fj[i], Bj[i]) for i in range(5))
            h = h + sum(_resonator_db(f, fh * tr["fs"], bh) for fh, bh in HIGHER_POLES)  # keep 5-10 kHz alive
            if tr["nz_on"][j] > 0.01:  # nasal antiformant (inverse resonator), cross-faded in
                h = h - tr["nz_on"][j] * _resonator_db(f, tr["nzf"][j], 150.0)
            return h
        # --- voiced part: harmonics with -6 dB/oct net source+radiation tilt + extra tilt
        if tr["av"][j] > -80:
            f0 = tr["f0"][j]
            fk = f0 * np.arange(1, int((SR / 2 - 200) // f0))
            src_db = lambda f: -10 * np.log10(1 + (f / 180.0) ** 2) - tr["tilt"][j] * np.log2(np.maximum(f, 100) / 100)
            amp = 10 ** ((src_db(fk) + vt_db(fk)) / 20)
            gain = np.sqrt(2) * ref * 10 ** (tr["av"][j] / 20) / np.sqrt((amp ** 2).sum())  # rms -> ref*av
            amp *= gain
            ap_k = 0.0 if tr["f_ap"] is None else 1 / (1 + (tr["f_ap"] / fk) ** 3)
            # spread each harmonic with the window kernel: power (a/2)^2 |W|^2
            d = freqs[None, :] - fk[:, None]
            hit = np.abs(d) <= kdf[-1]
            pw = np.zeros_like(d)
            pw[hit] = np.interp(d[hit], kdf, kW)
            P[j] += (((1 - ap_k) * amp)[:, None] ** 2 / 4 * pw).sum(0)
            if tr["f_ap"] is not None:  # aperiodic share of the same envelope: PSD S(f) = a(f)^2 / (2 f0)
                S = gain ** 2 * 10 ** ((src_db(freqs) + vt_db(freqs)) / 10) / (2 * f0)
                P[j] += 1 / (1 + (tr["f_ap"] / np.maximum(freqs, 1)) ** 3) * S * (SR / 2) * w2
        # --- noise part (aspiration shaped by vocal tract, or fricative/burst band)
        if tr["af"][j] > -80:
            if tr["asp"][j]:
                shape_db = vt_db(freqs) - 10 * np.log10(1 + (1500 / np.maximum(freqs, 1)) ** 4)  # hp ~1.5k
            else:
                c, wdt = tr["noise_c"][j], tr["noise_w"][j]
                shape_db = _resonator_db(freqs, c, wdt) - 10 * np.log10(1 + (0.7 * c / np.maximum(freqs, 1)) ** 8)
            s = 10 ** (shape_db / 10)
            s /= s.mean()
            var = (ref * 10 ** (tr["af"][j] / 20)) ** 2
            P[j] += var * s * w2
    return P


# same sentence, preceded by a short inhale (real close-mic recordings almost always have one)
UTT_BREATH = [("sil", 120, 0, []), ("h", 300, -9, []), ("sil", 140, 0, [])] + UTT[1:]


# prosody v2: Spanish pre-nuclear accents peak *after* the stressed vowel (delayed peak),
# continuation rise at the comma, higher register for the question, final rise on "hoy?"
UTT2 = [
    ("sil", 260, 0, []),
    ("o", 120, 2, [(0.0, 112), (1.0, 132)]),
    ("l", 50, 1, [(0.5, 138)]),
    ("a", 125, 0, [(0.15, 134), (1.0, 106)]),
    ("m", 70, -1, [(0.5, 104)]),
    ("u", 90, 2, [(0.2, 112), (1.0, 128)]),
    ("n", 55, 0, [(0.5, 130)]),
    ("dcl", 40, 0, [(0.5, 124)]),
    ("drel", 12, 0, []),
    ("o", 150, -1, [(0.1, 116), (0.6, 112), (1.0, 122)]),
    ("sil", 220, 0, []),
    ("kcl", 55, 0, []),
    ("krel", 30, 0, []),
    ("o", 105, 3, [(0.0, 140), (1.0, 156)]),
    ("m", 60, 0, [(0.5, 158)]),
    ("o", 75, 0, [(0.3, 146), (1.0, 132)]),
    ("e", 70, -1, [(0.5, 126)]),
    ("s", 95, 0, []),
    ("tcl", 45, 0, []),
    ("trel", 18, 0, []),
    ("a", 125, 2, [(0.1, 128), (0.7, 140), (1.0, 136)]),
    ("s", 80, 0, []),
    ("o", 110, 1, [(0.1, 124), (1.0, 130)]),
    ("i", 160, -1, [(0.3, 145), (1.0, 182)]),
    ("sil", 260, 0, []),
]
UTT2_BREATH = [("sil", 120, 0, []), ("h", 300, -9, []), ("sil", 140, 0, [])] + UTT2[1:]
