# "hola mundo, cómo estás hoy?" — decoder-only speech from SAME-S latents

Rules followed: audio comes **only** from `core.decode()` of `neural-audio-codec/same-s-cpu.ts`
(the encoder is never called). Latents are created with numpy/torch. No TTS, no recordings, no
waveform synthesised outside the decoder: the phonetic target is an *analytic expected-STFT-power*
description (formant tables, F0 contour, durations) that is only used inside the loss.

Environment: conda env `latent-alife-synthesis` (torch 2.10, numpy 2.2.6, librosa 0.11), Apple M4 Max, MPS.

## The decoder (what we learned by reading it)
- Stable Audio 3 "SAME-S" autoencoder, 44.1 kHz stereo, 256-dim latents, **4096 samples per latent frame (92.9 ms)**.
- Decoder: latent → 768-d token + 16 learned query tokens → 2× chunked transformers (chunks of 2 frames,
  second pass shifted by 1 frame) → conv → 16 patches × 256 samples. Receptive field ≈ ±1 frame.
- Latents are "softnormed" (÷ running_std = 0.0657): unit-ish scale. Needs an even number of frames.

## Iterations

| # | folder | idea | result |
|---|--------|------|--------|
| 0 | `00_exploration/` | probe zeros / random / static latents, per-dim sweeps, find a silence latent | zeros ≠ silence (−34 dB); random N(0,1) → bursts of noise; a *static* latent → pitched harmonic buzz; silence latent found (−65 dB, norm 8.06); single dims barely matter (distributed code); `dec_project_in` uses all 256 dims |
| 1 | `01_baseline/` | Adam lr 0.05 on log-power-mel L1 vs analytic target, init = silence | **diverged** (16 → 46 dB error): latents exploded (std 13), broadband noise |
| 2 | `02_curriculum_test/` | lr 0.01, 8 restarts in one MPS batch, floor curriculum −35→−75 dB | "hola mundo" envelope appears, second phrase collapses **exactly** onto the silence latent: power-spectrum gradients ∝ signal amplitude vanish at silence (silence = attractor). HNR −6 dB (noise, not voice) |
| 2b | `02_phoneme_codebook/` | loss on mel of **magnitude** (no vanishing grad); one static latent per phone, 10 phones × 12 restarts in one batch | all 5 vowels identified from LPC formants (a 725/1249/2517 Hz vs 700/1250/2500 target). Latent PCA plane is phonetic: open/back vowels vs /i u/, nasals+/l/ cluster, voiceless /s h/ apart |
| 3 | `03_codebook_init/` | utterance init = time-weighted mix of phone latents; population of 8 + frame-wise crossover; noise-robust objective decode(z+σε) | error 15.3 → **5.7 dB**; vowel ID 90 %, pauses −95 dBFS (40 dB below speech), /s/ centroid 7.7–8.2 kHz, voicing recall 93 %. Phrase-1 F0 follows target closely; **phrase-2 F0 stuck at the codebook's 120 Hz** (spectral losses can't slide harmonics). HNR median 6 dB (natural ≈ 15–25) |
| 4 | `04_pitch_codebook/` | pitch-aware vocabulary: 8 voiced phones × F0 {95,115,140,170,205} + /s/ /h/, harmonic fine-structure loss 60–2500 Hz | all 40 voiced entries: right vowel, F0 within ±50 cents (95–205 Hz). `ladder_*.wav` = each phone "sung" up the 5 pitches. **Pitch is not a shared latent direction**: per-phone d(latent)/d(logF0) are near-orthogonal (mean abs cos 0.12). HNR of sustained vowels 10–21 dB at F0 ≥ 140 Hz (lower at 95–115). Found a faint 172.27 Hz tone (= 44100/256, the decoder's patch rate) in /s/ and silences |
| 5 | `05_pitchbook_prosody2/` | init from pitch vocabulary (log-F0 interpolation per frame), prosody v2 (Spanish delayed peaks, question register, final rise), voice onsets/breathy offsets, aperiodicity above ~4.5 kHz, room tone −68 dBFS, harmonic loss | error **4.70 dB**, F0 error 192 → **92 cents**, final "hoy?" rises 140 → 172 Hz, stereo centred (side/mid 0.07). Vowel ID 80 % (coarticulated /o/ before /e/, final glide), HNR still 5 dB. 3/8 candidates diverged and were never recycled |
| 6 | `06_sweep_sigma_harm/` | A/B inside one batch: per-candidate latent-noise σ ∈ {0, .05, .1} × harmonic-loss weight ∈ {0.5, 1.5, 3}, band widened to 4 kHz, warm start from 5 | **σ = 0 diverges** from a good solution (→ 9–19 dB); σ ≥ 0.05 with strong harmonic weight is stable (4.64 dB). The latent noise is a landscape smoother: the decoder's loss surface is rough at small scales. `frame_spectra_target_vs_decoded.png`: harmonics aligned only below ~1.2 kHz; in phrase 2 the decoder's F0 sits a few % off the plan so *every* harmonic is misaligned |
| 7 | `07_repitch/` | analysis-by-synthesis re-pitch: measure decoded F0 (pyin + 0.1 % harmonic-comb refinement), let the target adopt it (±3 st), rebuild targets | 4.52 dB, HNR 6.0 — small gain: converged basin |
| 8 | `08_periodicity/` | **periodicity loss**: differentiable normalised autocorrelation at the period on voiced frames, hinge at r = 0.97 (≈ HNR 15 dB, natural, not buzzy); sweep weight {0, 1, 3, 10} | **HNR 6 → 15.4 dB**, voicing recall 98 %, vowel ID 100 % (w = 3), clean harmonic lines to ~5 kHz. Cost: 4.5 → 5.7–6.1 dB vs the analytic target. New artifact: a −13 dBFS click at 2.29 s (hinge on the decaying final /i/ pushes a latent frame that also spans the tail) |
| 9 | `09_silence_guard/` | periodicity only where voicing is established (av > −6 dB, ±30 ms), **silence guard** (10 ms RMS in planned pauses ≤ room tone + 8 dB), re-pitch with 30 % pull back toward the planned contour | click gone (tails ≤ −58 dBFS), but the pull destabilised pitch (octave errors 25–29 %) and "estás" /a/ collapsed into a **172.27 Hz buzz = identical consecutive 256-sample patches**, a degenerate, trivially periodic decoder mode |
| 10 | `10_guards/` | periodicity loss gains two cheat guards: r(P/2) ≤ 0.8 (no octave-up), r(256 samples) ≤ 0.85 unless P ≈ 256 (no patch buzz); no pull | all 4 candidates stable, HNR 13.6–14 dB, no octave errors, vowel ID 90 %. Per-frame error audit: **/k/ burst −16 dB, /t/ closure +16 dB** (no gap: "estás" ≈ "essás") |
| 11 | `11_bursts/` | frame weights from the plan: bursts ×5, closures ×√5 | bursts unchanged (local minimum); 2/4 candidates diverged |
| 12 | `12_local_surgery/` | **local latent surgery**: a 6-frame window starting at an even frame reproduces the full-context output of its 2 middle frames exactly (max diff 1e-6, measured) → 48 restarts of just those 2 frames per MPS batch, score on the exact region, splice the winner | /k/ burst −16 → −3.6 dB, /t/ closure +16 → +0.2 dB (real silent gap), /s/ +6 → −0.7 dB. **HNR 15.8 dB, voicing false-alarm 28 → 5.8 %**, vowel ID 90 % |
| 13 | `13_surgery_glide/` | surgery on the o→i glide of "hoy" + offset | window loss 24 → 14; current best male take |
| 14 | `14_surgery_accent/` | surgery on "cómo" toward the *planned* accent peak (158 Hz) | **rejected**: voicing 74 %, vowel ID 70 %, octave garbage. The decoder's attractor there is ~139 Hz flat |
| 15 | `15_pitch_codebook_female/` | voice B: female vocabulary, F0 165–320 Hz, formants ×1.15 (decoder harmonics were cleaner at F0 ≥ 140 Hz) | all vowels identified, F0 mostly within ±50 cents; pitch again phone-specific (mean abs cos 0.09) |
| 16 | `16_female_base/` | female voice, F0 contour ×1.75, **inhale breath before "hola"** (`UTT2_BREATH`), pitch-vocabulary init, recycled crossover, silence guard | already better than the male at the same stage: 4.44 dB, F0 error 62 cents, **vowel ID 100 %**, HNR 7.8 (male 5.1) |
| 17 | `17_female_periodicity/` | re-pitch + periodicity/guards polish sweep | σ = 0.1: 4.61 dB, **HNR 16.5 dB**, voicing 99 %, vowel ID 90 %, F0 error 76 cents. Consonant audit: same failure pattern as the male (/k/ −11 dB, /t/ closure +18.6 dB) |
| 18 | `18_female_surgery/` | local surgery at 5 windows (/d/, /k/, /t/, 2nd /s/, glide) | all accepted: /k/ −11 → −3.5 dB, /t/ closure +18.6 → +1.0 dB, /s/ +7 → −1.5 dB |
| 19 | `19_female_surgery_d/` | surgery generalised to 8-frame windows / any frame pair, for the /d/ burst that straddles frames 11–12 | /d/ stays weak (−10.6 dB); accepted as-is: post-nasal Spanish /d/ in "mundo" is naturally weakly released |
| — | (audit) | lag-256 autocorrelation inside /s/: **0.4–0.5** in both voices (real fricative noise ≈ 0): the patch-rate buzz hides inside the hiss | → fricative aperiodicity guard: \|r(256)\|, \|r(512)\| ≤ 0.1 on frames planned as pure noise |
| 20 | `20_female_polish/` | final global polish: all terms + aperiodicity guard + burst weights, lr 0.002 | error 4.6 → **3.46 dB**, **HNR 17.9 dB**, /s/ lag-256 corr 0.51 → **0.04**, consonants preserved |
| 21 | `21_male_polish/` | same polish for the male take | error **3.84 dB**, HNR 17.65 dB, **vowel ID 100 %**, /s/ lag-256 corr ≈ 0 |

## Final takes (`final/`)

| | male (`hola_mundo_male.wav`) | female (`hola_mundo_female.wav`) |
|---|---|---|
| source latents | `21_male_polish` cand 0 | `20_female_polish` cand 1 |
| frames / length | 28 × 92.9 ms = 2.60 s | 30 = 2.79 s (incl. inhale) |
| log-mel distance to the phonetic plan | 3.84 dB | 3.46 dB |
| vowel identification (LPC F1/F2, nearest Spanish vowel) | 100 % | 90 % |
| HNR, median over vowels | 17.7 dB | 17.9 dB |
| F0 vs planned contour | 92 cents rms (phrase 2 flatter than planned) | 69 cents rms |
| voicing recall / false alarm | 97 % / 12 % | 98 % / 5 % |
| pauses vs speech | −47 dB (room tone ≈ −68 dBFS) | −40 dB |
| 10.77 Hz latent-frame warble | −7.9 dB | −9.1 dB |

Also in `final/`: `perceptual_*_analysis.png` (spectrogram, F0 vs plan, vowel chart), `perceptual_*_waveforms.png`
(40 ms glottal-pulse zooms), `latent_path_*.png/.csv` (each latent frame's nearest phone@F0 vocabulary entry and
the sentence drawn as a path in the vocabulary's PCA plane), and `progression_montage.wav`: the takes from
iterations 1 → 2 → 3 → 5 → 8 → 10 → 13 → 21 (male) → 16 → 20 (female), 0.6 s apart, so the improvement can be heard in one file.

## What mattered (in order of impact)
1. Mel of **magnitude**, not power — power-spectrum gradients vanish at silence and every restart collapsed onto it.
2. A **phone vocabulary** found by optimising latents through the decoder, then a **pitch-aware** one (spectral losses cannot slide harmonics to a new F0).
3. **Latent noise during optimisation** (σ 0.05–0.1): without it the optimiser leaves good solutions — the loss surface is rough at small scales.
4. **Periodicity loss** (autocorrelation hinge at r = 0.97) + guards against the decoder's cheats (octave-up, identical-patch 172 Hz buzz): HNR 6 → 16–18 dB.
5. **Local latent surgery** on 2-frame windows that the decoder reproduces exactly: stop bursts and closure gaps.
6. Silence guard, fricative aperiodicity guard, re-pitching the target to the decoder's own F0.


## Performance notes (MPS)
- fp32 only: fp16 gradients → NaN, bf16 gradients uncorrelated with fp32 (cos 0.009); speed gain was only ~18 %.
- fwd+loss+bwd for 8 candidates × 2.6 s audio = 265 ms. Removing the per-step `.cpu()` sync in the loop took a
  step from ~880 ms to ~250 ms.
- Unified-memory limit: backward needs ~6 MB per latent frame. 1000 frames/step (252 codebook candidates)
  pushed the machine into 13.8 GB of swap at ~0% CPU; batches of ~200–300 frames/step run at full speed.
