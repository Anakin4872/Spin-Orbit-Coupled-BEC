# Spin-Orbit Coupled BEC
> Numerical study of the phase diagram of a spin-orbit coupled Bose–Einstein condensate of
> potassium-39, from the single-particle band structure to the interacting mean-field
> regime with Feshbach-tunable contact interactions.

---

## Overview

A pair of counter-propagating Raman lasers dresses a two-component BEC into a pseudospin-1/2
system with a synthetic spin-orbit coupling of the Lin/Spielman form
$$ 
H_\mathrm{SOC} = (\hbar k_L/m)\,p_x\,\sigma_z + (\hbar\Omega/2)\,\sigma_x + (\hbar\delta/2)\,\sigma_z.
$$
The lower dressed band reorganizes into a competition between three many-body phases —
a single plane-wave (PW), a stripe with broken translational symmetry, and a zero-momentum
(ZM) phase — controlled by the Raman coupling $\Omega$, the two-photon detuning $\delta$,
and the contact interactions.

For $^{87}Rb$ in the canonical NIST setup the three intra/inter-spin scattering lengths are
nearly identical and the symmetric Li/Stringari phase diagram applies. **In $^{39}K$ the picture
changes qualitatively**: the broad $|F=1, m_F=+1\rangle \otimes |F=1, m_F=+1\rangle$ Feshbach
resonance at $B_0 \simeq 402.9\,$G is a powerful experimental knob on $a_{\downarrow\downarrow}$
without comparable handles on the other two channels, so the relevant theory is the
*asymmetric* extension (Martone, Pepe, Capuzzi, Pitaevskii, Stringari, PRA **86**, 063621 (2012)).
The asymmetry $g_{\downarrow\downarrow}/g_{\uparrow\uparrow}$ in the working window is large —
often two orders of magnitude — and the consequences for the phase diagram, the mean-field
shift of the PW→ZM boundary, the role of the trap, and the (in-)accessibility of stripes
are the subject of this repository.

This is a forked and extended version of the **spinor-GPE** quasi-2D pseudospinor GPE solver
of Daley *et al.* (see citation below); the SOC propagator, Strang split-step machinery, and
GPU-accelerated PyTorch backend are theirs. The work here adds ⁳⁹K species support with
field-dependent multi-pole Feshbach scattering lengths wired through the GPE, a two-stage
imaginary-time protocol tuned for the asymmetric K39 g-tensor, and a companion analytical
module for the asymmetric Li/Stringari mean-field theory that runs alongside the simulation.

---
## Built on `spinor-GPE`

All GPE machinery — the `PSpinor` class, the `TensorPropagator`, the Strang split-step
imaginary/real-time loops, the FFT-based kinetic propagation — is from: <https://github.com/ultracoldYEG/spinor-gpe>

Installation and dependencies (PyTorch, CUDA toolkit if GPU-accelerated, NumPy, SciPy,
matplotlib) follow the upstream instructions verbatim. Clone the upstream, install in
editable mode, then drop in the files of this repo over the corresponding paths.

## What this repo adds

### $^{39}K$ species support in the core simulator

- `spinor_gpe/constants.py` — atomic data for $^{39}K$ and a multi-pole Moerdijk
  parametrization of the three pseudospin-1/2 scattering channels
  ($a_{\uparrow\uparrow}$, $a_{\downarrow\downarrow}$, $a_{\uparrow\downarrow}$),
  fit to the per-channel data of Lysebo & Veseth, *Phys. Rev. A* **81**, 032702 (2010).
  Convention: $|\uparrow\rangle = |F=1,m_F=0\rangle$, $|\downarrow\rangle = |F=1,m_F=+1\rangle$,
  so the broad 402.9 G resonance moves $a_{\downarrow\downarrow}$.
- `spinor_gpe/pspinor/pspinor.py` — `PSpinor(species='K39', B=...)` derives the dimensional
  quasi-2D couplings $g_{ij}(B,\omega_x,\omega_z)$ from the multi-pole fit at run time,
  bypassing the Rb87-style `g_scale` factorization that breaks when the coupling is
  $B$-dependent. Backward-compatible with the upstream Rb87 path.
- A **two-stage imaginary-time cool** that lands the dressed-band ground state cleanly
  in spite of the asymmetric, partly attractive K39 g-tensor — stage 1 relaxes the
  non-interacting Raman Hamiltonian, stage 2 turns on the K39 interactions inside the
  dressed band. (Without this, single-stage cooling can fall into bare-component
  kinetic minima for K39.)
- A species-aware time step ($dt = 1/200$ for K39, where $E_L \approx 86\,\hbar\omega_x$
  pushes the symmetric-split error band lower than for Rb87).

### Analytical companion module

`v4_analytical/li_stringari_overlay.py` implements the asymmetric lower-band PW ansatz
with the 2D parabolic Thomas-Fermi virial closure, including the proper
$\langle V\rangle + e_\mathrm{int} = 2\mu/3$ per-particle contribution in the trap.
Closed-form `g_eff(q)` lets you minimize over $q$ directly — no self-consistent loop —
and the result matches the full 2D GPE to within the simulation's $k$-space grid
resolution across the entire $(\Omega, \delta, B, N)$ grid we tested.

## Running

After installing the upstream `spinor-gpe` package, every script runs from the
repo root with:

```bash
PYTHONPATH=. MPLBACKEND=Agg python <script>.py
```

A typical $(\Omega, \delta, B, N)$ point takes $\sim 1$ minute on a single GPU at
mesh $128 \times 128$ with two cool stages of 6000 steps each. Full 2D phase
diagrams (≈ 75–225 points) take 1–4 hours.

---

## Physics references

- Y.-J. Lin, K. Jiménez-García, I. B. Spielman, *Spin–orbit-coupled Bose–Einstein
  condensates*, **Nature 471** (2011) 83.
- Y. Li, L. P. Pitaevskii, S. Stringari, *Quantum tricriticality and phase transitions
  in spin–orbit coupled Bose–Einstein condensates*, **PRL 108** (2012) 225301.
- G. I. Martone, F. Pepe, P. Capuzzi, L. P. Pitaevskii, S. Stringari,
  *Tricriticalities and quantum phases in spin–orbit-coupled spinor BECs*,
  **PRA 86** (2012) 063621.
- M. Lysebo, L. Veseth, *Feshbach resonances and transition rates for cold homonuclear
  collisions between ⁳⁹K atoms*, **PRA 81** (2010) 032702.
- L. Tanzi, S. Maffei, M. Inguscio, G. Modugno, *et al.*, on K39 BEC near the broad
  402.9 G resonance (Pisa/Florence programme).
