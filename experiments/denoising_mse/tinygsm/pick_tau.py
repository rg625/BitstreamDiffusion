#!/usr/bin/env python
"""Best common global score temperature for the S/M switching study: the tau (run for both models, seeds 0 and 1)
with the highest mean GSM8K accuracy averaged over CoBit-S and CoBit-M (tau = 1 is the base runs). Prints tau."""
import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "results_tinygsm/diagnostics/raw"
acc = {}
for f in RAW.glob("*_seed[01].json"):
    m = re.fullmatch(r"([SM])_(base|stemp[\d.]+)_seed([01])", f.stem)
    if not m:
        continue
    tau = 1.0 if m.group(2) == "base" else float(m.group(2)[5:])
    acc.setdefault((m.group(1), tau), []).append(json.loads(f.read_text())["accuracy"])
taus = sorted(t for (mm, t), v in acc.items() if mm == "S" and len(v) == 2 and len(acc.get(("M", t), [])) == 2)
score = {t: 0.5 * (sum(acc[("S", t)]) / 2 + sum(acc[("M", t)]) / 2) for t in taus}
for t in taus:
    print(f"# tau {t:g}: S {50 * sum(acc[('S', t)]):.2f}  M {50 * sum(acc[('M', t)]):.2f}  mean {100 * score[t]:.2f}",
          flush=True)
print(f"{max(score, key=score.get):g}")
