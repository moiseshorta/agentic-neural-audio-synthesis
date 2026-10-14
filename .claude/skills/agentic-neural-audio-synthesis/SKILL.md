---
name: agentic-neural-audio-synthesis
description: Disentangle the latent space of any frozen neural audio codec or autoencoder decoder (e.g. SAME-S, Stable Audio, EnCodec, DAC, RAVE) by experiment, then use that map to make the decoder produce a requested sound (speech, a note, a beat, a texture) by writing and optimising its latents directly, with no encoder, no training data, no TTS and no reference audio. Use when asked to "use only the decoder", "make the decoder say/play X", "map/untangle/explore the latent space of a codec", "create latents for the decoder", "decoder-only synthesis", "agentic neural audio synthesis", or to drive a pretrained audio decoder without a trained prior.
---

# Agentic Neural Audio Synthesis

You play a frozen audio decoder like an instrument, but first you have to find its controls. Map the latent space by
experiment: where silence, noise and tone live, which factors are shared directions and which are entangled, how far
each frame reaches, and how the decoder cheats. Then write the target from domain knowledge (phonetics, instrument
acoustics), optimise the latents through the decoder, measure the result, and change the method based on what failed.
You cannot hear, so every judgement must come from a measurement or a spectrogram. Keep the map you build (findings,
atlas files, measured constants) in the log. It is reusable for every later sound on the same codec.

Reference implementation (speech, SAME-S decoder): `hello_world_docs/` in this repository. Losses are in
`optim_core.py`, the analytic target in `phonetics.py`, the proxies in `analysis.py`, the sentence optimiser in
`03_utterance.py` and the surgery in `12_local_surgery.py`. Reuse them before writing new code.

## Hard rules (keep unless the user changes them)

- Audio comes **only** from `decoder(latents)`. Never call the encoder. Never read an audio file. Never synthesise a waveform outside the decoder, not even as a target.
- The target is a *description* (expected spectrum, envelopes, F0) used only inside the loss.
- Save every iteration in a numbered folder (`NN_short_name/`): wav, spectrogram PNG, latents `.npy`, metrics `.json`. Keep a `LOG.md` table with columns `# | folder | idea | result`, including dead ends.
- Grep your own scripts before reporting: show that there is no encoder call and no audio read.

## Procedure

### 0. Read the decoder before touching it
- Load the model and read its source. A TorchScript `.code` / debug pickle often holds the original Python with comments. Write down the sample rate, channels, latent dimension, samples per latent frame (hop), latent normalisation (e.g. divided by a running std), frame-count constraints (even T?) and the architecture (chunked attention, conv kernel).
- Work out the **receptive field in latent frames**. You will rely on it for crossover and surgery. If the source is unreadable, measure it: perturb one latent frame and see which output samples change.
- Identify the latent type. Continuous (VAE / autoencoder, e.g. Stable Audio, RAVE): optimise the latents directly. Quantised (RVQ codecs such as EnCodec or DAC): optimise the continuous embedding that the decoder consumes (the sum of codebook vectors), and optionally snap to the nearest codes at the end and re-measure.
- Probe: zeros, random N(0, s) at several s, one latent repeated over time. Then find a **silence latent** by minimising output energy. Zeros are often not silent.
- Check that every latent dimension is used (input projection SVD and column norms). Codes are usually distributed, so single-dimension sweeps barely matter.

### 1. Write the target as physics
- Express the sound as parameter tracks per analysis frame: F0, partial amplitudes, resonances (formants or body modes), noise level and band, and envelopes.
- Convert the tracks to the **expected STFT power** in closed form: each partial is spread by the window kernel |W(f − f_k)|², plus shaped noise power × Σw². Average attacks through the window so that onsets leak into earlier frames the way a real STFT does.
- Add realism terms: coarticulation or legato smoothing, onset and offset ramps, aperiodic energy above a crossover frequency, room tone in silences, micro-jitter on F0, breaths before phrases.
- Plot target vs decoded mel for every run.

### 2. Loss: magnitude, not power
- Multi-resolution (e.g. 2048/256 and 512/128) **log-mel of STFT magnitude**, L1 in dB with a floor relative to the peak. Use a coarse-to-fine floor curriculum (−30 → −70 dB).
- Never use log-*power* mel. Its gradients scale with amplitude and vanish at silence, so restarts collapse onto the silence latent.
- Add a linear-frequency magnitude term over the resolved partials (about 60–4000 Hz) for harmonic fine structure.
- Return a **per-candidate** loss of shape (B,) and call `.sum().backward()`, so batch members get independent gradients.

### 3. Vocabulary first, sequence second
- Optimise one **static** latent per unit (phone, instrument articulation) against its stationary target, with 6–12 restarts per unit in one batch.
- Spectral losses cannot slide partials to a new pitch, so build **unit × pitch** entries (about 5 levels across the range). Do not assume pitch is a shared latent direction; measure it (|cos| between per-unit d z / d log F0 was about 0.1).
- Initialise a sequence frame by frame as the time-overlap-weighted mix of the entries it spans, interpolated in log F0.
- Plot the vocabulary's PCA plane. It shows what the decoder considers similar.

### 4. Search
- Population of about 8 sequences in one batch: Adam, warm-up plus cosine schedule, lr 0.004–0.007 (0.002 for polishing).
- **Latent noise** during optimisation: decode(z + σε) with σ 0.05–0.1, annealed by half. This is required. Starting from a good solution (4.7 dB), every σ = 0 run left it (9–19 dB). σ = 0.1, or σ = 0.05 with a strong harmonic weight, held it (4.6–4.7 dB). Sweep σ on your codec in one batch.
- Frame-wise crossover every ~150 steps. Score each latent frame (neighbour-averaged), splice the best candidate per frame into an elite, and refill the worst half. This works because the receptive field is about ±1 frame.
- A/B inside one batch: give each candidate its own hyper-parameters (`--sweep sigma:w_harm:w_per`) instead of running sequential experiments.
- **Analysis-by-synthesis re-pitch**: measure the decoded F0 (pYIN plus harmonic-comb refinement) and let the target adopt it within ±3 semitones. Do not drag the decoder far from the pitch it prefers.

### 5. Periodicity and the decoder's cheats
- Periodicity loss on voiced frames: normalised autocorrelation of 1024-sample windows at the planned period P, hinge `relu(0.97 − r(P))`. 0.97 ≈ 15 dB HNR, natural; 1.0 is buzzy. Apply it only where voicing is established (±30 ms away from onsets and offsets).
- Expect **specification gaming** and guard against it in the loss:
  - octave-up: `relu(r(P/2) − 0.8)`
  - repeated decoder patches (a buzz at SR / patch_size, here 44100/256 = 172.27 Hz): `relu(r(patch) − 0.85)` unless P ≈ patch
  - the same patch periodicity hidden in noise: on frames planned as pure noise, `relu(|r(patch)| − 0.1)` and `relu(|r(2·patch)| − 0.1)`
  - clicks or buzz in pauses: a silence guard on 10 ms RMS, kept at or below room tone + 8 dB.
- After every new loss term, look for what got worse.

### 6. Local latent surgery for short events
- Short events (stop bursts, closures, transients, drum hits) sit in local minima that global weighting cannot move (×5 burst weights changed nothing).
- Use the decoder's chunk structure: find the window W and offset such that decoding only the window reproduces the full-context output of some inner frames. **Measure it** (max |diff| about 1e-6). With 2-frame chunks and a 1-frame shifted pass, a 6-frame window starting at an even frame is exact for frames s+2 and s+3.
- Re-search only those frames: 48 restarts in one batch (small, medium and fresh-random perturbations), latent noise 0.07, about 300–400 steps. Score on the exact region with the same loss plus emphasis on the event, and splice the winner back only if it beats the original.
- Generalise to 8-frame windows for events that straddle two frames.

### 7. Measure like a listener you are not
Build proxies and report them every iteration:
- speech: vowel identity from LPC formants (nearest vowel in Bark space), F0 error vs plan (cents) and octave errors, voicing recall and false alarms, median HNR from autocorrelation, pause depth, sibilant centroid, lag-patch autocorrelation in fricatives, and per-frame spectral error at each consonant
- music: onset timing error vs the grid, per-event spectral error, pitch per note, decay times, and the energy share per instrument
- always: the latent-frame-rate warble (SR/hop Hz modulation), stereo side/mid ratio, and target-vs-decoded spectrogram plots
- Audit per frame, then fix the worst region locally (step 6) rather than re-weighting globally.

### 8. Know when to stop
- If forcing the plan breaks the take (octave garbage, voicing collapse), the decoder has an attractor there. Reject the branch, log it, and keep the previous take.
- If the output stays rough at a given register, try another register or voice. Here, female F0 (≥ 140 Hz) gave cleaner harmonics.
- Finish with a low-lr global polish (all terms plus guards) from the best take, then write final takes and a progression montage that has one take per main iteration.

## Engineering notes
- Use fp32. fp16 gradients were NaN and bf16 gradients were uncorrelated with fp32.
- Never sync to the CPU inside the optimisation loop. Keep loss history on the device; removing `.cpu()` cut step time by 3.5×.
- Keep batches under about 300 latent frames per step on 36 GB unified memory. Larger batches swap and run about 10× slower.
- Free memory between surgery sites (`torch.mps.empty_cache()`).
- Run long jobs in the background, write logs to a file, and grep progress.

## Report
End with: final takes (paths), a metrics table, what mattered (ranked), remaining artefacts, and the grep evidence that only the decoder produced audio. State plainly that you cannot hear the result and that the user should listen.
