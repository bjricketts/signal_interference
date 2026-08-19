#!/usr/bin/env python3
"""
interference_sims.py
--------------------
Stochastic test of the "phase lags without delays" interference of QPO Paper B,
using the broadband-noise + transfer-function machinery of QPO Paper A.

Background
==========
Paper B ("Something for nothing: phase lags without delays") models each energy
band as a *hybrid* signal: a decaying aperiodic component (a zero-centred
Lorentzian, the broadband noise / BBN) plus a damped oscillatory component (a
Lorentzian at nu0, the QPO), both sharing a common damping envelope Delta.  It
shows analytically that if two bands differ *only* in their QPO/BBN centroid
frequency (delta_nu), width (delta_Delta), or modulation fraction (delta_m),
the interference between the BBN and QPO parts imprints a structured, non-zero
phase lag on the cross-spectrum -- with no physical time delay anywhere.

Crucially, Paper B builds those signals from *deterministic sine waves* and
notes (Sect. 5): the signals are "fully coherent by construction ... A fully
stochastic treatment, in which the QPO is modeled as a driven damped oscillator
rather than a deterministic sinusoid, is left for future work."

This module IS that stochastic treatment.  It reuses Paper A's approach --
broadband noise convolved (in the Fourier domain) with a damped-harmonic-
oscillator (DHO) transfer function -- to build Paper B's hybrid bands from a
common stochastic driver, Poisson-samples them, and measures the cross-spectrum
with stingray exactly as Paper A does.  We then ask: do Paper B's interference
phase lags survive for realistic, stochastic, finite-coherence signals?

Model (per band i)
==================
Shared white driver n(t) (flat power spectrum).  Optionally each band mixes in
an independent driver to control inter-band coherence:

    n_i(t) = sqrt(c) * n_shared(t) + sqrt(1 - c) * n_indep_i(t)

Two linear filters of that driver (both from Paper A's toolkit):

    aperiodic (BBN):  H_BBN(f; Delta)     = 1 / (Delta + 2i f)        -> Lorentzian at 0, FWHM Delta
    oscillator (QPO): H_DHO(f; f0, zeta)  = w0^2/(w0^2 - w^2 + 2i zeta w0 w)
                                            (Paper A's DHO), w = 2pi f, w0 = 2pi f0,
                                            zeta = Delta/(2 f0)  ->  FWHM ~ Delta, Q = f0/Delta

The band variability is the Paper-B mixture (each component normalised to unit
variance, then combined by modulation fraction m and an optional QPO phase phi):

    v_i(t) = (1 - m_i) * a_hat_i(t) + m_i * q_hat_i(t; phi_i)

and the observed light curve is a Poisson draw around

    rate_i(t) = mean_i * (1 + frac_rms * v_i(t) / std(v_i)).

Because both the BBN and QPO parts of every band are linear filters of the same
driver, the cross-spectral *phase* is exactly arg(Hbar_1 Hbar_2^*) with

    Hbar_i(f) = (1 - m_i) H_BBN(f;Delta_i)/sigma_a,i
              + m_i e^{-i phi_i} H_DHO(f;nu_i,zeta_i)/sigma_q,i,

which is the analytic prediction we overlay on the measured phase.  In the
shared-driver limit (c = 1) the intrinsic coherence is unity and the measured
phase must converge to this prediction (Paper B's regime).  For c < 1 the bands
are only partially coherent and we can watch the features dilute.

Run `python interference_sims.py --help`.
"""
from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np

# Coherence ~1 in the shared-driver limit makes stingray's phase-error formula
# divide by zero; those bins simply have negligible error. Silence the noise.
warnings.filterwarnings("ignore", message="divide by zero")
np.seterr(divide="ignore", invalid="ignore")
import matplotlib.pyplot as plt

from stingray import Lightcurve, AveragedCrossspectrum, AveragedPowerspectrum


# =========================================================================
# Transfer functions (Paper A machinery) evaluated on an rfft frequency grid
# =========================================================================

def H_bbn(freq: np.ndarray, Delta: float) -> np.ndarray:
    """Zero-centred Lorentzian filter (the aperiodic / broadband component).

    H(f) = 1 / (Delta + 2i f).  |H|^2 is a Lorentzian centred at 0 with
    FWHM = Delta.  This is exactly Paper B's X_BBN (up to a constant).
    """
    return 1.0 / (Delta + 2j * freq)


def H_dho(freq: np.ndarray, f0: float, zeta: float, phi: float = 0.0) -> np.ndarray:
    """Paper A's damped-harmonic-oscillator transfer function (the QPO).

    H(f) = w0^2 / (w0^2 - w^2 + 2i zeta w0 w),  w = 2 pi f,  w0 = 2 pi f0.
    Q = 1/(2 zeta); the power-spectral peak has FWHM ~ f0/Q = 2 zeta f0.
    An optional constant phase offset `phi` (radians) is applied to the
    positive-frequency content -- Paper B's oscillation phase lag.
    """
    w = 2 * np.pi * freq
    w0 = 2 * np.pi * f0
    H = w0**2 / (w0**2 - w**2 + 2j * zeta * w0 * w)
    if phi != 0.0:
        ph = np.exp(-1j * phi * (freq > 0))  # constant offset on f>0, unity at DC
        H = H * ph
    return H


def zeta_for(Delta: float, nu: float) -> float:
    """DHO damping ratio giving a peak FWHM of Delta at centroid nu."""
    return Delta / (2.0 * nu)


# =========================================================================
# Band construction
# =========================================================================

@dataclass
class BandSpec:
    """Parameters of one energy band's hybrid signal."""
    nu: float = 1.0       # QPO centroid frequency [Hz]
    Delta: float = 0.2    # FWHM / damping of BOTH components [Hz]  (Q = nu/Delta)
    m: float = 0.5        # modulation fraction: variance split QPO vs BBN
    phi: float = 0.0      # constant QPO phase offset [rad]
    mean: float = 1.0e4   # mean count rate
    frac_rms: float = 0.3 # total fractional rms of the variability


def _apply(x: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Convolve a real signal with a frequency-domain filter H (rfft grid)."""
    X = np.fft.rfft(x - x.mean())
    return np.fft.irfft(H * X, n=len(x))


def make_driver(N: int, rng: np.random.Generator) -> np.ndarray:
    """A white (flat-spectrum) stochastic driver, zero mean, unit variance."""
    return rng.standard_normal(N)


def band_variability(driver: np.ndarray, dt: float, spec: BandSpec):
    """Return (v, a_hat, q_hat) -- the unit-variance band variability and its
    normalised aperiodic and QPO parts -- from a given driver realisation."""
    freq = np.fft.rfftfreq(len(driver), d=dt)
    zeta = zeta_for(spec.Delta, spec.nu)
    a = _apply(driver, H_bbn(freq, spec.Delta))
    q = _apply(driver, H_dho(freq, spec.nu, zeta, spec.phi))
    a_hat = a / a.std()
    q_hat = q / q.std()
    v = (1.0 - spec.m) * a_hat + spec.m * q_hat
    return v, a_hat, q_hat


def make_band(driver: np.ndarray, t: np.ndarray, dt: float, spec: BandSpec,
              rng: np.random.Generator, poisson: bool = True) -> Lightcurve:
    """Build one Poisson-sampled light curve for a band on a given driver."""
    v, _, _ = band_variability(driver, dt, spec)
    rate = spec.mean * (1.0 + spec.frac_rms * v / v.std())
    rate = np.clip(rate, 0.0, None)
    counts = rng.poisson(rate) if poisson else rate
    return Lightcurve(t, counts, dt=dt, skip_checks=True)


def mixed_drivers(N: int, c: float, seed_shared: int, seed1: int, seed2: int):
    """Two per-band drivers sharing a fraction `c` of a common realisation.

    c = 1 -> identical drivers (fully coherent, Paper B's regime).
    c = 0 -> independent drivers (uncorrelated bands).
    """
    n_s = make_driver(N, np.random.default_rng(seed_shared))
    if c >= 1.0:
        return n_s, n_s
    n1 = make_driver(N, np.random.default_rng(seed1))
    n2 = make_driver(N, np.random.default_rng(seed2))
    d1 = np.sqrt(c) * n_s + np.sqrt(1.0 - c) * n1
    d2 = np.sqrt(c) * n_s + np.sqrt(1.0 - c) * n2
    return d1, d2


# =========================================================================
# Analytic (linear-response) prediction
# =========================================================================

def effective_filter(freq: np.ndarray, spec: BandSpec) -> np.ndarray:
    """Effective complex transfer function Hbar_i(f) of a band's variability,
    with each component normalised to unit variance exactly as the simulation
    does (sigma from the flat-driver power on this grid)."""
    zeta = zeta_for(spec.Delta, spec.nu)
    hb = H_bbn(freq, spec.Delta)
    hq = H_dho(freq, spec.nu, zeta, spec.phi)
    # White driver has flat power, so component variance ~ sum |H|^2 on the grid.
    sig_a = np.sqrt(np.sum(np.abs(hb) ** 2))
    sig_q = np.sqrt(np.sum(np.abs(hq) ** 2))
    return (1.0 - spec.m) * hb / sig_a + spec.m * hq / sig_q


def analytic_products(freq: np.ndarray, spec1: BandSpec, spec2: BandSpec):
    """Analytic cross-spectrum, phase lag and |CS| for the shared-driver limit.

    CS(f) = Hbar_1(f) Hbar_2^*(f) (times the driver power, a positive real
    scalar that does not affect phase).  Coherence is unity by construction.
    Reference band is band 1; CS is defined as band2 x conj(band1) to match
    the stingray call AveragedCrossspectrum(lc2, lc1, ...) used below.
    """
    H1 = effective_filter(freq, spec1)
    H2 = effective_filter(freq, spec2)
    # Matches stingray's phase convention for AveragedCrossspectrum(lc2, lc1)
    # (verified numerically: measured phase == arg(H1 * conj(H2))).
    cs = H1 * np.conj(H2)
    return cs, np.angle(cs)


# =========================================================================
# Spectral products (stingray, exactly as Paper A)
# =========================================================================

def spectral_products(lc1: Lightcurve, lc2: Lightcurve, segment: float,
                      rebin: float | None = None):
    """PSD of each band + cross-spectrum, coherence and phase lag.

    Cross-spectrum is band2 x conj(band1): band 1 is the reference.
    """
    ps1 = AveragedPowerspectrum.from_lightcurve(lc1, segment_size=segment, norm="frac")
    ps2 = AveragedPowerspectrum.from_lightcurve(lc2, segment_size=segment, norm="frac")
    cs = AveragedCrossspectrum.from_lightcurve(lc2, lc1, segment_size=segment, norm="frac")
    if rebin is not None:
        ps1 = ps1.rebin_log(f=rebin)
        ps2 = ps2.rebin_log(f=rebin)
        cs = cs.rebin_log(f=rebin)
    lag, lag_e = cs.phase_lag()
    try:
        coh = cs.intrinsic_coherence()[0]
    except Exception:
        coh = cs.coherence()[0]
    return dict(ps1=ps1, ps2=ps2, cs=cs, lag=lag, lag_e=lag_e, coh=coh)


# =========================================================================
# Experiment driver
# =========================================================================

# Paper B reference band (nu_j = 1, Delta_j = 0.2 -> Q = 5, m_j = 0.5, phi_j = 0)
REF = dict(nu=1.0, Delta=0.2, m=0.5, phi=0.0)

DT = 1.0 / 512.0
T_TOTAL = 1024.0
SEGMENT = 16.0

# offset -> (label, human title, list of offset values, colour meaning)
EXPERIMENTS = {
    "dnu":    dict(param="nu",    base=REF["nu"],    values=[0.0, 0.01, 0.02, 0.05, 0.10],
                   sym=r"$\delta\nu$", title="Frequency offset"),
    "dDelta": dict(param="Delta", base=REF["Delta"], values=[0.0, 0.02, 0.05, 0.10, 0.20],
                   sym=r"$\delta\Delta$", title="Width offset"),
    "phi":    dict(param="phi",   base=REF["phi"],   values=[0.0, 0.1, 0.2, 0.35, 0.5],
                   sym=r"$\phi$", title="QPO phase offset (a genuine lag)"),
    "dm":     dict(param="m",     base=REF["m"],     values=[-0.4, -0.2, 0.0, 0.2, 0.4],
                   sym=r"$\delta m$", title="Modulation-fraction offset"),
}


def run_experiment(key: str, c: float = 1.0, poisson: bool = True,
                  frac_rms: float = 0.3, mean: float = 1.0e4, seed: int = 11):
    """Run one offset sweep; return a list of per-offset result dicts."""
    cfg = EXPERIMENTS[key]
    N = int(round(T_TOTAL / DT))
    t = np.arange(N) * DT
    freq_grid = np.fft.rfftfreq(N, d=DT)
    rng_pois = np.random.default_rng(seed)

    d1, d2 = mixed_drivers(N, c, seed_shared=1, seed1=101, seed2=202)

    out = []
    for val in cfg["values"]:
        s1 = BandSpec(mean=mean, frac_rms=frac_rms, **REF)
        s2_kwargs = dict(REF)
        s2_kwargs[cfg["param"]] = cfg["base"] + val
        s2 = BandSpec(mean=mean, frac_rms=frac_rms, **s2_kwargs)

        lc1 = make_band(d1, t, DT, s1, rng_pois, poisson=poisson)
        lc2 = make_band(d2, t, DT, s2, rng_pois, poisson=poisson)

        prod = spectral_products(lc1, lc2, SEGMENT, rebin=0.03)
        cs_an, ph_an = analytic_products(prod["cs"].freq, s1, s2)

        out.append(dict(val=val, s1=s1, s2=s2, prod=prod,
                        an_cs=cs_an, an_phase=ph_an))
    return out, cfg


# =========================================================================
# Plotting (Paper B visual style: nu|CS| / phase / time-lag stack, plus a
# coherence panel for this work's stochastic extension)
# =========================================================================

# Per-experiment colormaps chosen to match Paper B's Figs. 5-8: a single-hue
# sequential ramp (dark = zero offset -> light = largest) for the one-signed
# sweeps, and a diverging red-blue map centred on zero for the signed dm sweep.
_SEQ_CMAP = {"dnu": "Greens", "dDelta": "Blues", "phi": "Oranges"}

X_LIM = (0.1, 10.0)     # frequency axis range, as in Paper B
_VLINE = "0.55"         # grey dotted centroid marker


def _fmt_val(key, val):
    return f"{val:+.2f}" if key == "dm" else f"{val:.2f}"


def _colors_for(key, values):
    """Colours for a sweep, matching Paper B's per-figure palettes."""
    if key == "dm":  # diverging, centred on zero offset (red -ve, blue +ve)
        vmax = max((abs(v) for v in values), default=1.0) or 1.0
        cmap = plt.cm.RdBu
        cols = []
        for v in values:
            if v == 0:
                cols.append((0.25, 0.25, 0.25, 1.0))   # dark grey so 0 stays visible
            else:
                cols.append(cmap(0.5 + 0.42 * (v / vmax)))
        return cols
    cmap = getattr(plt.cm, _SEQ_CMAP.get(key, "Greens"))
    return [cmap(x) for x in np.linspace(0.9, 0.42, len(values))]  # dark -> light


def plot_experiment(results, cfg, key, savepath, c=1.0):
    fig, axes = plt.subplots(4, 1, figsize=(6.6, 11.0), sharex=True,
                             gridspec_kw={"hspace": 0})
    ax_mag, ax_ph, ax_tl, ax_co = axes
    cols = _colors_for(key, [r["val"] for r in results])
    dashed = (0, (4, 2))

    tl_an_all = []
    for r, col in zip(results, cols):
        p = r["prod"]
        f = p["cs"].freq
        two_pi_f = 2 * np.pi * f
        # Panel 1: nu * |CS(nu)| (measured), log-log
        ax_mag.plot(f, f * np.abs(p["cs"].power), color=col, lw=1.8)
        # Panel 2: phase lag -- measured (solid) with analytic prediction (dashed)
        ax_ph.plot(f, p["lag"], color=col, lw=1.8)
        ax_ph.plot(f, r["an_phase"], color=col, lw=0.9, ls=dashed, alpha=0.7)
        # Panel 3: time lag = phase / (2 pi nu), same measured/analytic pairing
        ax_tl.plot(f, p["lag"] / two_pi_f, color=col, lw=1.8)
        tl_an = r["an_phase"] / two_pi_f
        ax_tl.plot(f, tl_an, color=col, lw=0.9, ls=dashed, alpha=0.7)
        tl_an_all.append(tl_an)
        # Panel 4: intrinsic coherence (this work's addition to Paper B)
        ax_co.plot(f, p["coh"], color=col, lw=1.8, drawstyle="steps-mid")

    for ax in axes:
        ax.axvline(REF["nu"], color=_VLINE, ls=":", lw=1.3)
        ax.set_xscale("log")

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=col, lw=2.6, label=_fmt_val(key, r["val"]))
               for r, col in zip(results, cols)]
    ncol = 2 if len(results) > 4 else 1
    ax_mag.legend(handles=handles, title=cfg["sym"], fontsize=8, ncol=ncol,
                  loc="upper right", framealpha=0.9)

    ax_mag.set_yscale("log")
    ax_mag.set_ylabel(r"$\nu\,|CS(\nu)|$")
    ax_mag.set_title(f'{cfg["title"]}   (stochastic bands, c={c:g})', fontsize=10)

    ax_ph.axhline(0, color="gray", lw=0.5)
    ax_ph.set_ylabel("Phase (rad)")

    ax_tl.axhline(0, color="gray", lw=0.5)
    ax_tl.set_ylabel("Time-lag (s)")

    ax_co.set_ylabel("Intrinsic coherence")
    ax_co.set_ylim(0.0, 1.08)
    ax_co.set_xlabel(r"$\nu$ (Hz)")
    ax_co.set_xlim(*X_LIM)

    # Bound the phase and time-lag panels to the analytic feature amplitude so a
    # sensible window shows and measured noise spikes are clipped out of view.
    fmask = (results[0]["prod"]["cs"].freq >= X_LIM[0]) & \
            (results[0]["prod"]["cs"].freq <= X_LIM[1])
    p_amp = max((np.nanmax(np.abs(r["an_phase"][fmask])) for r in results), default=0.5)
    p_amp = float(np.clip(p_amp * 1.5, 0.3, np.pi))
    ax_ph.set_ylim(-p_amp, p_amp)
    tl_stack = np.concatenate([t[fmask] for t in tl_an_all]) if tl_an_all else np.array([0.0])
    tl_amp = float(np.clip(np.nanmax(np.abs(tl_stack)) * 1.6, 1e-3, np.inf))
    ax_tl.set_ylim(-tl_amp, tl_amp)

    fig.align_ylabels(axes)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {savepath}")


# =========================================================================
# CLI
# =========================================================================

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"],
                   help="Which inter-band offset to sweep ('all' = every one).")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--ext", default="png", choices=["png", "pdf"])
    p.add_argument("-c", "--coherence", type=float, default=1.0,
                   help="Shared driver fraction c in [0,1] (1 = Paper B fully "
                        "coherent regime; <1 = partially incoherent bands).")
    p.add_argument("--frac_rms", type=float, default=0.3,
                   help="Total fractional rms of each band's variability.")
    p.add_argument("--mean", type=float, default=1.0e4,
                   help="Mean count rate (higher -> less Poisson noise).")
    p.add_argument("--no-poisson", action="store_true",
                   help="Disable Poisson sampling (ideal, noise-free limit).")
    p.add_argument("--seed", type=int, default=11)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    keys = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    tag = "" if args.coherence >= 1.0 else f"_c{args.coherence:g}"
    for key in keys:
        print(f"[{key}]  c={args.coherence}")
        results, cfg = run_experiment(
            key, c=args.coherence, poisson=not args.no_poisson,
            frac_rms=args.frac_rms, mean=args.mean, seed=args.seed)
        savepath = os.path.join(args.outdir, f"{key}{tag}.{args.ext}")
        plot_experiment(results, cfg, key, savepath, c=args.coherence)


if __name__ == "__main__":
    main()