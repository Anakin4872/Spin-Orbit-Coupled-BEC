"""
Single-particle SOC phase diagram: Omega vs delta for Raman-coupled pseudospin-1/2.

Computes the ground-state quasimomentum q_min by minimising the lower
dressed-band dispersion across the (Omega, delta) plane.  Provides helpers for
validating the GPE code against these analytical results.

Units:  energies in E_L = hbar^2 k_L^2 / (2m),  momenta in k_L.

The single-particle Hamiltonian in recoil units (q = k_x/k_L, E in E_L) is

    H(q)/E_L = | q^2 + 2q + d/2      O/2      |
               |      O/2       q^2 - 2q - d/2 |

where O = Omega/E_L, d = delta/E_L.  The factor of 2 in the SOC term arises
from k_L^2 = 2 E_L.

Lower band:
    E_-(q)/E_L = q^2 - sqrt[(2q + d/2)^2 + (O/2)^2]

KEY PHYSICS:
    At delta = 0:  sharp 2nd-order PW -> ZM transition at  Omega_c = 4 E_L.
    At delta != 0: q_min varies smoothly -- NO sharp transition exists in the
                   single-particle spectrum.  A true 1st-order phase boundary
                   at finite delta emerges ONLY when mean-field interactions
                   are included.

References
----------
[1] Lin, Jimenez-Garcia, Spielman, Nature 471, 83 (2011)
[2] Li et al., PRL 108, 225301 (2012)
"""

import numpy as np
from scipy.optimize import minimize_scalar
import scipy.constants as const
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ============================================================================
#  0.  Unit conversion
# ============================================================================

def recoil_energy_hz(mass_amu, wavel_m):
    """Compute the recoil energy E_L = hbar^2 k_L^2 / (2m) in Hz.

    Parameters
    ----------
    mass_amu : float
        Atomic mass in unified atomic mass units.
    wavel_m : float
        Raman laser wavelength in metres.  k_L = 2pi / wavel_m.

    Returns
    -------
    EL_hz : float
        Recoil energy in Hz  (i.e. E_L / h).
    """
    u   = const.physical_constants['atomic mass constant'][0]
    kL  = 2 * np.pi / wavel_m
    EL  = const.hbar**2 * kL**2 / (2 * mass_amu * u)
    return EL / const.h


# Convenience presets
SPECIES = {
    'K39':  {'mass_amu': 38.96370668, 'wavel_m': 770.108e-9},
    'Rb87': {'mass_amu': 86.909180520, 'wavel_m': 780.241e-9},
    'Na23': {'mass_amu': 22.98976928,  'wavel_m': 589.000e-9},
}


# ============================================================================
#  1.  Lower-band dispersion
# ============================================================================

def lower_band(q, omega, delta):
    """Lower dressed-band energy E_-(q) in recoil units.

    Parameters
    ----------
    q : float or ndarray
        Quasimomentum in units of k_L.
    omega : float
        Raman coupling Omega / E_L.
    delta : float
        Detuning delta / E_L.
    """
    return q**2 - np.sqrt((2*q + delta / 2)**2 + (omega / 2)**2)


def upper_band(q, omega, delta):
    """Upper dressed-band energy E_+(q) in recoil units."""
    return q**2 + np.sqrt((2*q + delta / 2)**2 + (omega / 2)**2)


def find_kmin(omega, delta, q_range=(-3.0, 3.0)):
    """Find the quasimomentum that minimises E_-(q).

    Returns
    -------
    q_min : float    -- in units of k_L
    E_min : float    -- in units of E_L
    """
    res = minimize_scalar(lower_band, bounds=q_range, method='bounded',
                          args=(omega, delta))
    return res.x, res.fun


def qmin_analytic_delta0(omega):
    """Exact q_min at delta = 0:  q^2 = 1 - (Omega/4)^2  for Omega < 4 E_L."""
    omega = np.asarray(omega, dtype=float)
    sq = 1.0 - (omega / 4.0)**2
    return np.where(sq > 0, np.sqrt(sq), 0.0)


# ============================================================================
#  1b.  Dressed spin composition
# ============================================================================
#
# At each q the lower-band eigenstate of
#
#   H(q) = | q^2 + 2q + d/2     O/2     |
#          |     O/2       q^2 - 2q - d/2 |
#
# is  |-,q> = cos(theta_q/2)|up> - sin(theta_q/2)|dn>  with
#
#   tan(theta_q) = (O/2) / (2q + d/2)
#
# The Pauli-z expectation value in this state is
#
#   <sigma_z>(q) = -(2q + d/2) / sqrt[(2q + d/2)^2 + (O/2)^2]
#
# The sign convention here is such that  <sigma_z> -> +1  for delta -> -inf
# (system polarised into |up>) and  <sigma_z> -> -1  for delta -> +inf, which
# matches the Lin et al. (2011) figure where the |down> band lies above the
# |up> band at delta > 0.

def sigma_z(q, omega, delta):
    """Pauli-z expectation in the lower dressed band at quasimomentum q.

    Parameters
    ----------
    q : float or ndarray
        Quasimomentum in units of k_L.
    omega : float
        Raman coupling Omega / E_L.
    delta : float
        Detuning delta / E_L.

    Returns
    -------
    <sigma_z> : same shape as q,  in [-1, +1].
    """
    num = -(2*q + delta / 2)
    den = np.sqrt((2*q + delta / 2)**2 + (omega / 2)**2)
    return num / den


def sigma_z_at_min(omega, delta):
    """<sigma_z> evaluated at the ground-state quasimomentum q_min."""
    q_min, _ = find_kmin(omega, delta)
    return sigma_z(q_min, omega, delta)


def population_fraction_at_min(omega, delta):
    """Spin populations (n_up, n_dn) of the dressed ground state.

    Convenience wrapper:  n_up = (1 + <sigma_z>)/2,  n_dn = 1 - n_up.
    Useful for direct comparison with the background colouring in
    Lin et al. (2011) Fig. 2a.
    """
    sz = sigma_z_at_min(omega, delta)
    n_up = 0.5 * (1.0 + sz)
    return n_up, 1.0 - n_up


# ============================================================================
#  2.  Phase diagram scan
# ============================================================================

def scan_phase_diagram(omega_pts=300, delta_pts=300,
                       omega_range=(0.01, 8.0), delta_range=(-8.0, 8.0)):
    """Compute q_min, E_min, and <sigma_z>_min over a grid of (Omega, delta).

    Returns
    -------
    omegas, deltas : 1D arrays
    Q_MIN  : 2D array  (delta_pts x omega_pts)  -- ground-state quasimomentum
    E_MIN  : 2D array                            -- ground-state energy
    SZ_MIN : 2D array                            -- <sigma_z> at q_min
    """
    omegas = np.linspace(*omega_range, omega_pts)
    deltas = np.linspace(*delta_range, delta_pts)
    Q_MIN  = np.empty((delta_pts, omega_pts))
    E_MIN  = np.empty((delta_pts, omega_pts))
    SZ_MIN = np.empty((delta_pts, omega_pts))

    for i, d in enumerate(deltas):
        for j, o in enumerate(omegas):
            q, e = find_kmin(o, d)
            Q_MIN[i, j]  = q
            E_MIN[i, j]  = e
            SZ_MIN[i, j] = sigma_z(q, o, d)

    return omegas, deltas, Q_MIN, E_MIN, SZ_MIN


def count_band_minima(omega, delta, nq=3000):
    """Count the number of local minima in E_-(q)."""
    q = np.linspace(-2.5, 2.5, nq)
    E = lower_band(q, omega, delta)
    is_min = np.zeros(nq, dtype=bool)
    is_min[1:-1] = (E[1:-1] < E[:-2]) & (E[1:-1] < E[2:])
    return int(np.sum(is_min))


def two_minimum_boundary(omega_arr):
    """Critical detuning delta_c(Omega) above which E_-(q) has only one minimum.

    Derived by requiring E_-'(q)=0 and E_-''(q)=0 simultaneously, then
    eliminating q.  From E_-''(q) = 2 - Omega^2/f^3 = 0:

        f = (Omega^2 / 2)^{1/3}

    where f = sqrt[(2q + delta/2)^2 + (Omega/2)^2].  Substituting into
    E_-'(q) = 0 and solving for delta:

        delta^2 = 4 (f^2 - Omega^2/4)(f - 2)^2 / f^2

    The boundary is monotonically decreasing from delta -> inf at Omega -> 0
    to delta = 0 at Omega = 4 E_L.  For Omega >= 4, the lower band always
    has a single minimum.

    Parameters
    ----------
    omega_arr : array_like
        Raman coupling values in E_L units.  Must satisfy 0 < Omega <= 4.

    Returns
    -------
    delta_c : ndarray
        Critical |delta|/E_L (positive branch; boundary is symmetric in delta).
    """
    omega = np.asarray(omega_arr, dtype=float)
    f = (omega**2 / 2.0)**(1.0 / 3.0)
    delta_sq = 4.0 * (f**2 - omega**2 / 4.0) * (f - 2.0)**2 / f**2
    return np.sqrt(np.maximum(delta_sq, 0.0))


# ============================================================================
#  3.  GPE validation helpers
# ============================================================================

def classify_ground_state(psik, kx_array, kL_recoil, threshold=0.05):
    """Classify a converged GPE state as PW or ZM from its k-space density.

    Parameters
    ----------
    psik : list of 2D ndarray
        Momentum-space wavefunctions [psi_k_up, psi_k_dn].
    kx_array : 1D ndarray
        k_x grid in code units (1/a_x).
    kL_recoil : float
        Recoil momentum in code units.
    threshold : float
        |k_peak| / k_L below which the state is classified as ZM.

    Returns
    -------
    phase : str   -- 'PW' or 'ZM'
    k_peak : float -- peak position in units of k_L
    """
    densk_total = np.abs(psik[0])**2 + np.abs(psik[1])**2
    nkx = np.sum(densk_total, axis=0)
    idx_peak = np.argmax(nkx)
    k_peak = kx_array[idx_peak]
    k_peak_kL = k_peak / kL_recoil
    phase = 'ZM' if np.abs(k_peak_kL) < threshold else 'PW'
    return phase, k_peak_kL


def run_gpe_validation(omega_EL_list, delta_EL_list, ps_kwargs,
                       t_step=1e-3, n_steps=5000):
    """Run imaginary-time GPE at selected (Omega, delta) points and classify.

    Template -- requires spinor_gpe to be importable.

    Parameters
    ----------
    omega_EL_list : list of float
        Omega values in E_L units.
    delta_EL_list : list of float
        delta values in E_L units (same length as omega_EL_list).
    ps_kwargs : dict
        Base kwargs for PSpinor.  Use near-zero g_sc for single-particle
        validation.  Example::

            ps_kwargs = dict(
                path='soc_test',
                omeg={'x': 2*np.pi*50, 'y': 2*np.pi*50, 'z': 2*np.pi*2000},
                g_sc={'uu': 1e-10, 'dd': 1e-10, 'ud': 1e-10},
                mesh_points=(256, 256),
                r_sizes=(16, 16),
                atom_num=1e4,
                pop_frac=(0.5, 0.5),
                overwrite=True,
            )

    t_step, n_steps : float, int
        Imaginary-time parameters.

    Notes
    -----
    The workflow per test point is::

        ps = PSpinor(**kw)
        ps.coupling_setup(wavel=770.108e-9, kin_shift=True)
        ps.coupling_uniform(Omega_code)   # Omega in hbar*omega_x
        ps.detuning_uniform(delta_code)   # delta in hbar*omega_x
        ps.shift_momentum(scale=0.5)      # seed PW initial condition
        result, _ = ps.imaginary(t_step, n_steps)

    shift_momentum() helps imaginary-time convergence when the ground
    state is PW: without it the symmetric TF initial condition projects
    mostly onto ZM.  For a ZM test point the seeding is harmless.
    """
    from spinor_gpe.pspinor import pspinor as ps_module

    results = []
    for i, (om_EL, de_EL) in enumerate(zip(omega_EL_list, delta_EL_list)):
        kw = {**ps_kwargs, 'path': f"soc_validation/pt_{i:03d}",
              'overwrite': True}
        ps = ps_module.PSpinor(**kw)
        ps.coupling_setup(wavel=770.108e-9, kin_shift=True)

        # Convert from E_L to code units (hbar * omega_x)
        EL = ps.EL_recoil
        omega_code = om_EL * EL
        delta_code = de_EL * EL

        ps.coupling_uniform(omega_code)
        ps.detuning_uniform(delta_code)
        ps.shift_momentum(scale=0.5, frac=(0.5, 0.5))

        result, _ = ps.imaginary(t_step, n_steps)

        phase, kpeak_kL = classify_ground_state(
            result.psik, ps.space['kx'], ps.kL_recoil, threshold=0.05)

        qm_expected, _ = find_kmin(om_EL, de_EL)

        results.append({
            'omega_EL': om_EL, 'delta_EL': de_EL,
            'q_min_analytic': qm_expected,
            'k_peak_kL': kpeak_kL,
            'phase': phase,
            'energy': result.eng_final,
        })

        print(f"[{i+1}] O/EL={om_EL:.1f}  d/EL={de_EL:.1f}  "
              f"q_ana={qm_expected:+.4f}  "
              f"k_peak/kL={kpeak_kL:+.4f}  [{phase}]")

    return results


# ============================================================================
#  4.  Plotting
# ============================================================================

def plot_phase_diagram(omegas, deltas, Q_MIN, EL_kHz=None,
                       save_path=None, show=True):
    """Single-particle phase diagram as |q_min| colour map.

    Parameters
    ----------
    EL_kHz : float, optional
        Recoil energy in kHz.  If provided, axis labels and ticks are
        shown in kHz instead of E_L units.  Obtain from
        ``recoil_energy_hz(**SPECIES['K39']) / 1e3``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if EL_kHz is not None:
        om_axis = omegas * EL_kHz
        d_axis  = deltas  * EL_kHz
        xlabel  = r'$\Omega\ [\mathrm{kHz}]$'
        ylabel  = r'$\delta\ [\mathrm{kHz}]$'
        ylabel2 = r'$|q_{\min}|\ [k_L]$'
        vline   = 4.0 * EL_kHz
        vline_label = rf'$\Omega_c = {vline:.1f}\ \mathrm{{kHz}}$'
        d_slices = [d * EL_kHz for d in [0.0, 0.5, 1.0, 2.0, 4.0]]
        d_labels = [rf'$\delta = {d*EL_kHz:.1f}\ \mathrm{{kHz}}$'
                    for d in [0.0, 0.5, 1.0, 2.0, 4.0]]
    else:
        om_axis = omegas
        d_axis  = deltas
        xlabel  = r'$\Omega\ [E_L]$'
        ylabel  = r'$\delta\ [E_L]$'
        ylabel2 = r'$|q_{\min}|\ [k_L]$'
        vline   = 4.0
        vline_label = r'$\Omega_c = 4\,E_L$'
        d_slices = [0.0, 0.5, 1.0, 2.0, 4.0]
        d_labels = [rf'$\delta/E_L = {d:.1f}$' for d in d_slices]

    extent = [om_axis[0], om_axis[-1], d_axis[0], d_axis[-1]]

    ax = axes[0]
    im = ax.imshow(np.abs(Q_MIN), origin='lower', extent=extent,
                   aspect='auto', cmap='inferno', vmin=0, vmax=1.0)
    fig.colorbar(im, ax=ax, shrink=0.82, label=r'$|q_{\min}|\ [k_L]$')
    ax.axvline(vline, color='cyan', ls='--', lw=1.2, alpha=0.8,
               label=vline_label)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(r'$|q_{\min}|$: ground-state momentum', fontsize=12)

    ax = axes[1]
    om_arr_r = np.linspace(0.1, omegas[-1], 300)
    om_arr   = om_arr_r * EL_kHz if EL_kHz else om_arr_r
    for d_r, label in zip([0.0, 0.5, 1.0, 2.0, 4.0], d_labels):
        qm_arr = np.array([find_kmin(o, d_r)[0] for o in om_arr_r])
        ax.plot(om_arr, np.abs(qm_arr), lw=1.8, label=label)
    ax.axvline(vline, color='grey', ls=':', lw=1)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel2, fontsize=12)
    #ax.set_title(r'Sharp transition only at $\delta = 0$', fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_dispersions(cases=None, save_path=None, show=True):
    """Plot E_-(q) at selected (Omega, delta) points."""
    if cases is None:
        cases = [
            (1.0, 0.0, 'PW'),
            (3.0, 0.0, 'PW'),
            (5.0, 0.0, 'ZM'),
            (2.0, 1.0, r'PW, $\delta > 0$'),
            (2.0, 3.0, r'Large $\delta$'),
            (6.0, 2.0, r'Large $\Omega$'),
        ]
    nrows = (len(cases) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 3.5*nrows))
    q = np.linspace(-2.5, 2.5, 800)
    for ax, (om, de, title) in zip(axes.flat, cases):
        E = lower_band(q, om, de)
        qm, em = find_kmin(om, de)
        ax.plot(q, E, 'b-', lw=1.5)
        ax.plot(qm, em, 'ro', ms=7, zorder=5)
        ax.axvline(0, color='grey', ls=':', lw=0.5)
        ax.set_title(rf'$\Omega/E_L={om:.0f},\;\delta/E_L={de:.0f}$: {title}',
                     fontsize=9)
        ax.set_xlabel(r'$q\ [k_L]$')
        ax.set_ylabel(r'$E_-\ [E_L]$')
        ax.set_ylim(em - 0.5, max(E.max(), em + 2.0))
        ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_lin_diagram(omegas, deltas, Q_MIN, SZ_MIN, EL_kHz=None,
                     save_path=None, show=True):
    """Reproduce the single-particle phase diagram of Lin et al. (2011) Fig. 2a.

    Background colour: <sigma_z> at the dressed-band minimum  -- maps to the
    spin population fraction shown in red/blue in the original figure.
    Dashed lines: the boundary of the two-minimum region, from the
    inflection condition on E_-(q).

    Parameters
    ----------
    EL_kHz : float, optional
        Recoil energy in kHz.  If provided, axes are shown in kHz.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if EL_kHz is not None:
        om_axis  = omegas * EL_kHz
        d_axis   = deltas  * EL_kHz
        xlabel   = r'Raman coupling, $\Omega\ [\mathrm{kHz}]$'
        ylabel   = r'Detuning, $\delta\ [\mathrm{kHz}]$'
        crit_val = 4.0 * EL_kHz
        crit_lbl = rf'$({crit_val:.1f}\ \mathrm{{kHz}},\ 0)$'
        om_b_r   = np.linspace(0.05, 3.99, 400)
        om_b     = om_b_r * EL_kHz
        d_b      = two_minimum_boundary(om_b_r) * EL_kHz
        label_x_double   = 1.5  * EL_kHz
        label_x_single   = 6.5  * EL_kHz
        annot_xy  = (4.0 * EL_kHz, 0.0)
        annot_txt = (4.3 * EL_kHz, 0.7 * EL_kHz)
    else:
        om_axis  = omegas
        d_axis   = deltas
        xlabel   = r'Raman coupling, $\Omega/E_L$'
        ylabel   = r'Detuning, $\delta/E_L$'
        crit_val = 4.0
        crit_lbl = r'$(4\,E_L,\ 0)$'
        om_b_r   = np.linspace(0.05, 3.99, 400)
        om_b     = om_b_r
        d_b      = two_minimum_boundary(om_b_r)
        label_x_double   = 1.5
        label_x_single   = 6.5
        annot_xy  = (4.0, 0.0)
        annot_txt = (4.3, 0.7)

    extent = [om_axis[0], om_axis[-1], d_axis[0], d_axis[-1]]

    im = ax.imshow(SZ_MIN, origin='lower', extent=extent, aspect='auto',
                   cmap='RdBu', vmin=-1, vmax=1)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(r'$\langle\sigma_z\rangle$  (population imbalance)',
                 fontsize=11)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.ax.text(2.7, 1.0, r'$|\!\uparrow\rangle$', va='center',
               transform=cb.ax.transAxes, fontsize=12)
    cb.ax.text(2.7, 0.0, r'$|\!\downarrow\rangle$', va='center',
               transform=cb.ax.transAxes, fontsize=12)

    ax.plot(om_b,  d_b, 'k--', lw=1.3)
    ax.plot(om_b, -d_b, 'k--', lw=1.3)
    ax.plot(crit_val, 0.0, 'ko', mfc='white', ms=8, zorder=5)
    ax.annotate(crit_lbl, xy=annot_xy, xytext=annot_txt,
                fontsize=9, arrowprops=dict(arrowstyle='-', lw=0.7))

    d_max = d_axis[-1]
    ax.text(label_x_double,  0.35 * d_max, r"$|\!\downarrow'\rangle$",
            fontsize=14, ha='center', color='darkred')
    ax.text(label_x_double, -0.35 * d_max, r"$|\!\uparrow'\rangle$",
            fontsize=14, ha='center', color='darkblue')
    #ax.text(label_x_single, 0.0, 'Single\nminimum', fontsize=11,
            #ha='center', va='center', color='black')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    #ax.set_title('Single-particle phase diagram '
                 #'(after Lin, Jimenez-Garcia & Spielman 2011)',
                 #fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_sigma_z_slices(save_path=None, show=True):
    """<sigma_z>(q_min) vs delta at fixed Omega, and vs Omega at fixed delta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: <sigma_z> vs delta at fixed Omega
    ax = axes[0]
    d_arr = np.linspace(-6, 6, 400)
    for om in [0.5, 1.0, 2.0, 4.0, 6.0]:
        sz = np.array([sigma_z_at_min(om, d) for d in d_arr])
        ax.plot(d_arr, sz, lw=1.7, label=rf'$\Omega/E_L = {om:.1f}$')
    ax.axhline(0, color='grey', lw=0.6, ls=':')
    ax.axvline(0, color='grey', lw=0.6, ls=':')
    ax.set_xlabel(r'$\delta/E_L$', fontsize=11)
    ax.set_ylabel(r'$\langle\sigma_z\rangle$ at $q_{\min}$', fontsize=11)
    ax.set_title(r'$\langle\sigma_z\rangle$ vs $\delta$', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)

    # Right: <sigma_z> vs Omega at fixed delta
    ax = axes[1]
    om_arr = np.linspace(0.01, 8, 400)
    for de in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        sz = np.array([sigma_z_at_min(o, de) for o in om_arr])
        ax.plot(om_arr, sz, lw=1.7, label=rf'$\delta/E_L = {de:+.1f}$')
    ax.axvline(4.0, color='grey', ls=':', lw=1)
    ax.axhline(0, color='grey', lw=0.6, ls=':')
    ax.set_xlabel(r'$\Omega/E_L$', fontsize=11)
    ax.set_ylabel(r'$\langle\sigma_z\rangle$ at $q_{\min}$', fontsize=11)
    ax.set_title(r'$\langle\sigma_z\rangle$ vs $\Omega$', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ============================================================================
#  Main
# ============================================================================

if __name__ == '__main__':
    print("Computing single-particle phase diagram...")
    omegas, deltas, Q_MIN, E_MIN, SZ_MIN = scan_phase_diagram(
        omega_pts=300, delta_pts=300,
        omega_range=(0.01, 8.0), delta_range=(-8.0, 8.0))

    # --- Recoil units (universal) ---
    plot_phase_diagram(omegas, deltas, Q_MIN,
                       save_path='sp_phase_diagram.png', show=False)
    plot_dispersions(save_path='sp_dispersions.png', show=False)
    plot_lin_diagram(omegas, deltas, Q_MIN, SZ_MIN,
                     save_path='sp_lin_diagram.png', show=False)
    plot_sigma_z_slices(save_path='sp_sigma_z_slices.png', show=False)

    # --- Lab units: K39 @ 770.108 nm ---
    EL_kHz_K39 = recoil_energy_hz(**SPECIES['K39']) / 1e3
    print(f"\nK39 recoil energy: E_L/h = {EL_kHz_K39*1e3:.2f} Hz "
          f"= {EL_kHz_K39:.4f} kHz")
    print(f"Critical coupling: Omega_c = 4 E_L/h = {4*EL_kHz_K39:.3f} kHz")

    plot_phase_diagram(omegas, deltas, Q_MIN, EL_kHz=EL_kHz_K39,
                       save_path='sp_phase_diagram_K39_kHz.png', show=False)
    plot_lin_diagram(omegas, deltas, Q_MIN, SZ_MIN, EL_kHz=EL_kHz_K39,
                     save_path='sp_lin_diagram_K39_kHz.png', show=False)

    # --- Lab units: Rb87 @ 780.241 nm (comparison) ---
    EL_kHz_Rb = recoil_energy_hz(**SPECIES['Rb87']) / 1e3
    print(f"\nRb87 recoil energy: E_L/h = {EL_kHz_Rb*1e3:.2f} Hz "
          f"= {EL_kHz_Rb:.4f} kHz")
    print(f"Critical coupling: Omega_c = 4 E_L/h = {4*EL_kHz_Rb:.3f} kHz")

    # --- Sanity checks ---
    print(f"\nOmega_c at delta=0:  4 E_L  (analytical)")
    for om in [3.9, 3.95, 4.0, 4.05, 4.1]:
        qm, _ = find_kmin(om, 0.0)
        print(f"  O/EL = {om:.2f}  ->  |q_min| = {abs(qm):.6f}")

    print(f"\nComparison with exact formula q^2 = 1 - (O/4)^2:")
    for om in [1.0, 2.0, 3.0, 3.5]:
        qm_num = abs(find_kmin(om, 0.0)[0])
        qm_ex  = qmin_analytic_delta0(om)
        print(f"  O/EL = {om:.1f}  numerical={qm_num:.6f}  exact={qm_ex:.6f}")

    print(f"\n<sigma_z> sanity tests:")
    print(f"  delta=0, Omega=5 (ZM):  {sigma_z_at_min(5.0,  0.0):+.4f}  (expected 0)")
    print(f"  delta=+10, Omega=2:     {sigma_z_at_min(2.0, 10.0):+.4f}  (expected -> -1)")
    print(f"  delta=-10, Omega=2:     {sigma_z_at_min(2.0,-10.0):+.4f}  (expected -> +1)")
