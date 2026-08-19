#!/usr/bin/env python3
"""
verify.py
---------
Quantitative checks that the stochastic simulations behave as the physics
requires. Run: `python verify.py` (exits non-zero if any check fails).

Checks
======
1. Shared-driver limit (c = 1) reproduces Paper B:
   - intrinsic coherence near the QPO is ~1 (Paper B's "fully coherent" regime);
   - the measured cross-spectral phase equals the analytic interference
     prediction across the QPO, band-averaged, to within a small tolerance.
2. The delta_nu phase feature is genuinely interference-driven, not a delay:
   its amplitude at the QPO grows monotonically with the frequency offset,
   and vanishes when the two bands are identical.
3. Partial incoherence (c = 0.5) dilutes coherence at the QPO, as expected
   for realistic stochastic signals (Paper B, Sect. 5 caveat).
"""
import sys
import numpy as np

from interference_sims import (run_experiment, REF)

FAIL = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAIL.append(name)


def near_qpo(freq, lo=0.7, hi=1.4):
    return (freq >= lo) & (freq <= hi)


print("== 1. Shared-driver limit (c=1): measured == analytic, coherence ~1 ==")
res, cfg = run_experiment("dnu", c=1.0, seed=11)
for r in res:
    f = r["prod"]["cs"].freq
    mask = near_qpo(f)
    dphase = np.abs(r["prod"]["lag"][mask] - r["an_phase"][mask])
    med_coh = np.nanmedian(r["prod"]["coh"][mask])
    ok_phase = np.nanmedian(dphase) < 0.06
    ok_coh = med_coh > 0.9
    check(f"dnu={r['val']:.2f}: |measured-analytic| small", ok_phase,
          f"median dphi={np.nanmedian(dphase):.3f} rad")
    check(f"dnu={r['val']:.2f}: coherence ~1 near QPO", ok_coh,
          f"median coh={med_coh:.3f}")


print("\n== 2. dnu phase feature grows with offset and vanishes at 0 ==")
amps = []
for r in res:
    f = r["prod"]["cs"].freq
    mask = near_qpo(f, 0.85, 1.25)
    amps.append(np.nanmax(np.abs(r["an_phase"][mask])))
amps = np.array(amps)
vals = np.array([r["val"] for r in res])
check("zero offset -> ~zero phase", amps[vals == 0][0] < 0.02,
      f"amp(0)={amps[vals==0][0]:.4f} rad")
check("phase amplitude monotonic in dnu", np.all(np.diff(amps) > 0),
      "amps=" + ", ".join(f"{a:.2f}" for a in amps))


print("\n== 3. Partial incoherence (c=0.5) lowers coherence at the QPO ==")
res1, _ = run_experiment("dnu", c=1.0, seed=11)
res_half, _ = run_experiment("dnu", c=0.5, seed=11)
def coh_at_qpo(results):
    r = results[-1]
    f = r["prod"]["cs"].freq
    return np.nanmedian(r["prod"]["coh"][near_qpo(f)])
c1, ch = coh_at_qpo(res1), coh_at_qpo(res_half)
check("c=1 coherence > c=0.5 coherence at QPO", c1 > ch + 0.2,
      f"coh(c=1)={c1:.2f} vs coh(c=0.5)={ch:.2f}")


print()
if FAIL:
    print(f"FAILED {len(FAIL)} check(s): " + "; ".join(FAIL))
    sys.exit(1)
print("All checks passed.")
