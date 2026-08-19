# signal_interference

Does Paper B's *"phase lags without delays"* interference survive when the
signals are realistic, stochastic, and only partially coherent — instead of the
deterministic sine waves used in the paper?

This folder is a *self-contained* workflow that answers that question by
combining the two papers:

- **Paper B** (*Something for nothing: phase lags without delays*) models each
  energy band as a *hybrid* signal — an aperiodic component (a zero-centered
  Lorentzian, the broadband noise / BBN) plus a damped oscillator (a
  Lorentzian at ν₀, the QPO), sharing a common damping/width Δ. It shows
  *analytically* that if two bands differ only in QPO/BBN centroid frequency
  (δν), width (δΔ), or modulation fraction (δm), the interference between the BBN
  and QPO parts imprints a structured, non-zero cross-spectral phase lag — with
  no physical time delay anywhere. But Paper B's signals are deterministic sine
  shots, *"fully coherent by construction"*, and it flags (Section 5) that *"a
  fully stochastic treatment, in which the QPO is modeled as a driven damped
  oscillator rather than a deterministic sinusoid, is left for future work."*

- **Paper A** (*Complex lags from simple physics*) is exactly that machinery:
  generate broadband noise and convolve it with a damped-harmonic-oscillator
  (DHO) transfer function to make a *stochastic* QPO, then measure the
  cross-spectrum, phase lag, and coherence with `stingray`.

This workflow is the stochastic treatment Paper B left for future work. It builds
Paper B's hybrid bands with Paper A's noise-⊛-transfer-function method and checks
whether the interference lags persist.

## The model

The workflow passes a single white driver `n(t)` (flat power spectrum) through
two linear filters — both from Paper A's toolkit — to build each band's
variability:

```
aperiodic (BBN):  H_BBN(f; Δ)      = 1 / (Δ + 2i f)                     # Lorentzian at 0, FWHM Δ
oscillator (QPO): H_DHO(f; ν, ζ)   = ω₀²/(ω₀² − ω² + 2i ζ ω₀ ω)         # Paper A's DHO, ζ = Δ/(2ν)

v_i(t) = (1 − m_i)·â_i(t) + m_i·q̂_i(t; φ_i)          # Paper B's hybrid mixture (unit-variance parts)
rate_i(t) = mean_i · (1 + frac_rms · v_i/std(v_i))   # Poisson-sampled light curve
```

Both bands are filtered from the *same* driver, so this reproduces Paper B's
"fully coherent" regime. A knob `c ∈ [0, 1]` lets each band mix in an independent
driver (`n_i = √c·n_shared + √(1−c)·n_indep`) to make the bands *partially
incoherent* — the realistic case Paper B could not treat.

Because both components of both bands are linear filters of the driver, the
expected cross-spectral phase is exactly `arg(H̄₁ H̄₂*)`, computed in
`analytic_products()` and overlaid on every measured phase panel as the analytic
interference prediction. In the c = 1 limit the measurement must converge to it;
where coherence drops, the measurement scatters away from it.

Reference band (Paper B): ν = 1 Hz, Δ = 0.2 (Q = 5), m = 0.5, φ = 0.
Sampling matches the papers: dt = 1/512 s, 1024 s total, 16 s segments.

## What it produces

Four inter-band offset experiments (Paper B's Figs. 5–8), each a four-panel
figure styled after Paper B — ν·|CS(ν)| magnitude / phase / time-lag / intrinsic
coherence — with Paper B's per-experiment colormaps (Greens for δν, Blues for
δΔ, Oranges for φ, diverging red–blue for δm) and a grey dotted line at ν₀. The
top three panels follow Paper B's layout; in the phase and time-lag panels the
measured curve is solid and the analytic prediction is dashed. The coherence
panel is this work's addition, since Paper B is coherent by construction:

| File | Sweep | Paper B analog |
|------|-------|------------------|
| `figures/dnu.png`    | Frequency offset δν | Fig. 5 — sharp phase dip at ν₀ |
| `figures/dDelta.png` | Width offset δΔ     | Fig. 6 — phase sign-change at ν₀ plus broad low-frequency phase |
| `figures/phi.png`    | QPO phase offset φ  | Fig. 7 — smooth peak (a *genuine* lag, the control) |
| `figures/dm.png`     | Modulation offset δm | Fig. 8 — antisymmetric phase, flips sign with δm |

To get the partially coherent versions (`figures/*_c0.5.png`), add `-c 0.5`.

## Findings

1. **The interference survives.** With fully stochastic, broadband-driven,
   Poisson-sampled signals, the measured phase lags reproduce Paper B's analytic
   predictions for all four offsets in the coherent limit: the phase equals
   `arg(H̄₁ H̄₂*)` to within 0.04 rad, with coherence ≈ 1. A frequency, width, or
   modulation-fraction difference between bands produces a structured, non-zero
   phase lag with no physical delay, exactly as Paper B argues, and the effect is
   not an artifact of using pure sinusoids.
2. **Coherence is the discriminator.** Making the bands partially incoherent
   (c < 1) leaves the phase structure near ν₀ (where coherence is highest) but
   dilutes it elsewhere and adds noise to the estimate — precisely Paper B's
   Section 5 caveat that partial incoherence *"will dilute the cross-spectral
   phase ... reducing the amplitude of the features"* and lowers the
   signal-to-noise ratio of any phase measurement. Even at c = 1, a large δm
   drives a genuine high-frequency coherence drop where the two bands' BBN/QPO
   mixes diverge.

To check these findings quantitatively, run `verify.py`.

## Usage

```bash
pip install -r requirements.txt

python interference_sims.py all                 # all four figures, coherent (c=1)
python interference_sims.py all -c 0.5          # partially-coherent versions
python interference_sims.py dnu --ext pdf       # one experiment, PDF output
python interference_sims.py dm --no-poisson     # ideal noise-free limit
python verify.py                                # quantitative checks (exit 0 = pass)
```

Key options: `-c` / `--coherence` (shared driver fraction), `--frac_rms` (total
fractional rms), `--mean` (count rate; a higher value means less Poisson noise),
`--no-poisson`, and `--seed`.

## Files

- `interference_sims.py` — transfer functions, band builder, analytic overlay,
  experiment runner, plotting, and CLI.
- `verify.py` — quantitative correctness checks.
- `requirements.txt` and `figures/` — dependencies and outputs.

## Relation to the source repos

This folder is self-contained; nothing here imports from `QPO_sims/`. The DHO
transfer function and the broadband-⊛-transfer-function method follow Paper A's
`qpo_sims.py` (`dho_filter` and the stingray cross-spectral products); the hybrid
BBN+QPO band and the δν/δΔ/δm/φ parameterization follow Paper B.
