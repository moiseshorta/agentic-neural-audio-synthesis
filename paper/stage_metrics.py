"""Re-measure every stage's best take with the final analysis.py, so Table 2 uses one metric definition.
Run: ~/miniconda3/envs/latent-alife-synthesis/bin/python paper/stage_metrics.py  -> paper/stage_metrics.json"""
import os, sys, json, tempfile
import soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "hello_world_docs")
sys.path.insert(0, DOCS)
import phonetics as P
from analysis import analyze

M, F = ("UTT2", 1.0, 1.0), ("UTT2_BREATH", 1.75, 1.15)
STAGES = {"01_baseline": ("UTT", 1, 1), "02_curriculum_test": ("UTT", 1, 1), "03_codebook_init": ("UTT", 1, 1),
          **{d: M for d in ["05_pitchbook_prosody2", "06_sweep_sigma_harm", "07_repitch", "08_periodicity", "09_silence_guard",
                            "10_guards", "11_bursts", "12_local_surgery", "13_surgery_glide", "14_surgery_accent"]},
          **{d: F for d in ["16_female_base", "17_female_periodicity", "18_female_surgery", "19_female_surgery_d"]},
          "final/hola_mundo_female.wav": F, "final/hola_mundo_male.wav": M}
KEPT = {"08_periodicity": "cand2_sigma0.1_wharm3_wper3.wav", "10_guards": "cand1_sigma0.07_wharm3_wper3.wav",
        "11_bursts": "cand3_sigma0.1_wharm3_wper3.wav", "17_female_periodicity": "cand1_sigma0.1_wharm3_wper3.wav"}
out = {}
with tempfile.TemporaryDirectory() as tmp:
    for d, (utt, f0s, fms) in STAGES.items():
        wav = KEPT.get(d, "hola_mundo.wav")  # sweeps: the candidate the agent carried forward
        y, _ = sf.read(os.path.join(DOCS, d if d.endswith(".wav") else f"{d}/{wav}"), always_2d=True)
        m = analyze(y.T, os.path.join(tmp, "x"), utt=getattr(P, utt), f0_scale=f0s, formant_scale=fms, title=d)
        out[d] = {k: round(v, 2) for k, v in m.items() if isinstance(v, float)}
        print(d, out[d], flush=True)
json.dump(out, open(os.path.join(HERE, "stage_metrics.json"), "w"), indent=1)
