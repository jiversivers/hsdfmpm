from typing import Tuple

import numpy as np
from scipy.special import spherical_jn, spherical_yn

def mie_scattering_coefficients(m: complex, x: float) -> np.ndarray:
    """
      Compute the Mie scattering coefficients aₙ, bₙ, cₙ, and dₙ for a homogeneous sphere.

      This implements C. Mätzler’s June 2002 routine based on Bohren & Huffman (1983, pp. 100, 477).

      :param m: Complex refractive index of the sphere (m = m′ + i m″).
      :type m: complex
      :param x: Size parameter x = k₀·a, where k₀ is the wave number in the ambient medium
                and a is the sphere radius.
      :type x: float
      :returns: 2D array of shape (4, nmax) containing the Mie coefficients for orders n = 1…nmax.
                - Row 0: aₙ
                - Row 1: bₙ
                - Row 2: cₙ
                - Row 3: dₙ
      :rtype: numpy.ndarray
      """
    # determine highest order
    nmax = int(np.round(2 + x + 4 * x ** (1 / 3)))
    n = np.arange(1, nmax + 1)

    # scaled argument inside sphere
    z = m * x
    m2 = m * m

    # spherical Bessel functions j_n(x), j_n(z) and y_n(x)
    bx = spherical_jn(n, x)
    bz = spherical_jn(n, z)
    yx = spherical_yn(n, x)

    # spherical Hankel function h_n^(1)(x)
    hx = bx + 1j * yx

    # j_{n-1}(x) and j_{n-1}(z) via prepending order‑0
    b1x = np.concatenate(([spherical_jn(0, x)], bx[:-1]))
    b1z = np.concatenate(([spherical_jn(0, z)], bz[:-1]))
    # y_{n-1}(x)
    y1x = np.concatenate(([spherical_yn(0, x)], yx[:-1]))
    h1x = b1x + 1j * y1x

    # auxiliary combinations
    ax = x * b1x - n * bx
    az = z * b1z - n * bz
    ahx = x * h1x - n * hx

    # Mie coefficients
    an = (m2 * bz * ax - bx * az) / (m2 * bz * ahx - hx * az)
    bn = (bz * ax - bx * az) / (bz * ahx - hx * az)
    cn = (bx * ahx - hx * ax) / (bz * ahx - hx * az)
    dn = m * (bx * ahx - hx * ax) / (m2 * bz * ahx - hx * az)

    # stack into a single (4, nmax) array
    return np.vstack((an, bn, cn, dn))

def mie_efficiency(m: complex, x:float) -> np.ndarray:
    """
    Compute Mie efficiencies for extinction, scattering, absorption, backscattering,
    the asymmetry parameter, and the backscatter ratio for a homogeneous sphere,
    using Mie coefficients from ``mie_scattering_coefficients``.

    :param m: Complex refractive-index ratio (m = m' + i·m").
    :type m: complex
    :param x: Size parameter x = k₀·a, where k₀ is the wave number in the ambient medium
              and a is the sphere radius.
    :type x: float
    :returns: 1D array with the following entries:

        0. Real part of *m*  
        1. Imaginary part of *m*  
        2. Size parameter *x*  
        3. Q<sub>ext</sub> (extinction efficiency)  
        4. Q<sub>sca</sub> (scattering efficiency)  
        5. Q<sub>abs</sub> (absorption efficiency)  
        6. Q<sub>back</sub> (backscattering efficiency)  
        7. Asymmetry parameter ⟨cos θ⟩  
        8. Backscatter ratio Q<sub>back</sub> / Q<sub>sca</sub>  
    :rtype: numpy.ndarray of shape (9,)
    """
    # Handle singular case at x == 0
    if x == 0:
        return np.array([m.real, m.imag, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5])

    # Determine number of terms
    nmax = int(np.round(2 + x + 4 * x ** (1 / 3)))
    n = np.arange(1, nmax + 1)
    cn = 2 * n + 1
    c1n = n * (n + 2) / (n + 1)
    c2n = cn / (n * (n + 1))
    x2 = x * x

    # Fetch Mie coefficients a_n, b_n, c_n, d_n from mie_scattering_coefficients
    f = mie_scattering_coefficients(m, x)  # shape (4, nmax)
    an = f[0, :]
    bn = f[1, :]
    anp, anpp = np.real(an), np.imag(an)
    bnp, bnpp = np.real(bn), np.imag(bn)

    # Prepare g1 arrays for asymmetry calculation
    n1 = nmax - 1
    g1 = np.zeros((4, nmax))
    g1[0, :n1] = anp[1:]
    g1[1, :n1] = anpp[1:]
    g1[2, :n1] = bnp[1:]
    g1[3, :n1] = bnpp[1:]

    # Extinction efficiency Q_ext
    dn = cn * (anp + bnp)
    q = np.sum(dn)
    qext = 2 * q / x2

    # Scattering efficiency Q_sca
    en = cn * (anp ** 2 + anpp ** 2 + bnp ** 2 + bnpp ** 2)
    q = np.sum(en)
    qsca = 2 * q / x2

    # Absorption efficiency Q_abs
    qabs = qext - qsca

    # Backscattering efficiency Q_back
    fn = (an - bn) * cn
    gn = (-1) ** n
    fb = fn * gn
    qsum = np.sum(fb)
    qb = (qsum * np.conj(qsum)) / x2

    # Asymmetry parameter ⟨cos θ⟩
    asy1 = c1n * (anp * g1[0, :] + anpp * g1[1, :] + bnp * g1[2, :] + bnpp * g1[3, :])
    asy2 = c2n * (anp * bnp + anpp * bnpp)
    asy = 4 / x2 * np.sum(asy1 + asy2) / qsca

    # Backscatter ratio
    qratio = qb / qsca

    return np.array([m.real, m.imag, x, qext, qsca, qabs, qb, asy, qratio])

def mie_scatter_and_anisotropy(
        l_min: float,
        l_max: float,
        dl: float,
        r: float,
        sphere_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Mie scattering efficiencies and anisotropy versus wavelength.

    :param l_min: Minimum wavelength (μm).
    :type l_min: float
    :param l_max: Maximum wavelength (μm).
    :type l_max: float
    :param dl: Wavelength step size (μm).
    :type dl: float
    :param r: Particle radius (μm).
    :type r: float
    :param sphere_type: Type of sphere, either 'tissue' or 'beads'.
    :type sphere_type: str
    :returns: Tuple of four 1D arrays, each of length N = ⌊(l_max−l_min)/dl⌋+1:

        - **wavelengths**: Wavelengths (μm).  
        - **Qsca**: Total scattering efficiency (1/μm³).  
        - **Qback**: Backscattering efficiency (1/μm³).  
        - **g**: Anisotropy parameter ⟨cos θ⟩.  
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    # create wavelength vector
    wavelengths = np.arange(l_min, l_max + dl / 2, dl)
    N = wavelengths.size

    Qsca = np.zeros(N, dtype=float)
    Qback = np.zeros(N, dtype=float)
    g = np.zeros(N, dtype=float)

    for i, lam in enumerate(wavelengths):
        lt = sphere_type.lower()
        if lt == 'tissue':
            n_sph = 1.424
            n_med = 1.36
            m = n_sph / n_med
        elif lt in ('beads', 'bead'):
            n_sph = 1.5663
            n_med = 1.33
            m = (n_sph + 0.00785 / lam ** 2 + 0.000334 / lam ** 4) / n_med
        else:
            raise ValueError(f"Unknown sphere_type '{sphere_type}'")

        k = 2 * np.pi * n_med / lam
        x = r * k

        #  mie_efficiency returns [Re(m), Im(m), x, Qext, Qsca, Qabs, Qback, g, Qratio]
        F =  mie_efficiency(m, x)

        Qsca[i] = F[4].real
        Qback[i] = F[6].real
        g[i] = F[7].real

    return wavelengths, Qsca, Qback, g


import numpy as np


def compute_volume_distribution_by_scattering(
        redscat_coef_mm_inv: np.ndarray,
        red_scat_curve: np.ndarray,
        tot_vol_ml: float,
        per_sol_percent: float,
        act_bead_diameter_um: float,
        ref_wavelength_idx: int = 22
) -> np.ndarray:
    """
    Compute the bead‐and‐water volumes (in μL) required to achieve specified
    reduced scattering coefficients.

    :param redscat_coef_mm_inv: Array of desired reduced scattering coefficients (mm⁻¹).
    :type redscat_coef_mm_inv: numpy.ndarray
    :param red_scat_curve: Reduced scattering cross‐section vs. wavelength (units μm⁻¹).
    :type red_scat_curve: numpy.ndarray
    :param tot_vol_ml: Total phantom volume (mL).
    :type tot_vol_ml: float
    :param per_sol_percent: Percent solids in the bead stock solution (%).
    :type per_sol_percent: float
    :param act_bead_diameter_um: Actual bead diameter (μm).
    :type act_bead_diameter_um: float
    :param ref_wavelength_idx: Index into `red_scat_curve` to pick the reference value.
    :type ref_wavelength_idx: int
    :returns:
        A (N×3) array `volume_dist_ul` (μL) whose columns are:

        1. beads_volume_ul
        2. water_volume_ul
        3. total_volume_ul
    :rtype: numpy.ndarray
    """
    # 1) Compute density [1/μm³]
    density = redscat_coef_mm_inv / red_scat_curve[ref_wavelength_idx] / 1000.0

    # 2) Number of beads in phantom
    part_beads = tot_vol_ml * density * 1e12

    # 3) Total number of beads per mL of bead solution
    tot_beads = (6 * (per_sol_percent / 100.0) * 1e12) / (1.05 * np.pi * act_bead_diameter_um ** 3)

    # 4) Bead solution volume needed (μL)
    beads_volume_ul = (part_beads / tot_beads) * 1000.0

    # 5) Water volume to reach total (μL)
    water_volume_ul = tot_vol_ml * 1000.0 - beads_volume_ul

    # 6) Stack into (beads, water, total) columns
    volume_dist_ul = np.column_stack((
        beads_volume_ul,
        water_volume_ul,
        beads_volume_ul + water_volume_ul
    ))

    return volume_dist_ul


def compute_volume_distribution_by_bead_number(
        bead_volume_ul: np.ndarray,
        tot_vol_ml: float,
        per_sol_percent: float,
        act_bead_diameter_um: float,
        red_scat_curve: np.ndarray,
        ref_wavelength_idx: int = 22
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the updated reduced scattering coefficient and the bead‐and‐water
    volumes (in μL) based on a specified bead‐solution volume.

    :param bead_volume_ul: Array of bead volumes to add (μL).
    :type bead_volume_ul: numpy.ndarray
    :param tot_vol_ml: Total phantom volume (mL).
    :type tot_vol_ml: float
    :param per_sol_percent: Percent solids in the bead stock solution (%).
    :type per_sol_percent: float
    :param act_bead_diameter_um: Actual bead diameter (μm).
    :type act_bead_diameter_um: float
    :param red_scat_curve: Reduced scattering cross‐section vs. wavelength (μm⁻¹).
    :type red_scat_curve: numpy.ndarray
    :param ref_wavelength_idx: Index into `red_scat_curve` for reference (0‐based).
    :type ref_wavelength_idx: int
    :returns:
        - `volume_dist_ul`: (N×3) array whose columns are
            1. input bead_volume_ul
            2. water_volume_ul
            3. total_volume_ul
        - `updated_redscat_coef_mm_inv`: Array of updated reduced scattering coefficients (mm⁻¹).
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    # 1) Total beads per mL of bead solution
    tot_beads = (6 * (per_sol_percent / 100.0) * 1e12) / (1.05 * np.pi * act_bead_diameter_um ** 3)

    # 2) Water volume to reach total (μL)
    water_volume_ul = tot_vol_ml * 1000.0 - bead_volume_ul

    # 3) Number of beads in the phantom
    part_beads = bead_volume_ul * tot_beads

    # 4) Density [1/μm³]
    density = part_beads / (tot_vol_ml * 1e12)

    # 5) Updated reduced scattering coefficient (mm⁻¹)
    updated_redscat_coef_mm_inv = density * red_scat_curve[ref_wavelength_idx] * 1000.0

    # 6) Stack volumes
    volume_dist_ul = np.column_stack((
        bead_volume_ul,
        water_volume_ul,
        bead_volume_ul + water_volume_ul
    ))

    return volume_dist_ul, updated_redscat_coef_mm_inv


# def generate_phantom_profiles(
#         l_min: float,
#         l_max: float,
#         dl: float,
#         bead_radius_um: float,
#         sphere_type: str,
#         tot_vol_ml: float,
#         per_sol_percent: float,
#         mode: str,
#         redscat_coef_mm_inv: np.ndarray = None,
#         bead_volumes_ul: np.ndarray = None,
# ) -> dict:
#     """
#     Wrapper to compute reduced‐scattering profiles, reduced scattering coefficients,
#     phantom volume distributions, and reduced scattering coefficient profiles (μₛ′)
#     for one or more phantoms in a single call.
#
#     :param l_min: Minimum wavelength (μm).
#     :type l_min: float
#     :param l_max: Maximum wavelength (μm).
#     :type l_max: float
#     :param dl: Wavelength step size (μm).
#     :type dl: float
#     :param bead_radius_um: Bead radius (μm).
#     :type bead_radius_um: float
#     :param sphere_type: 'tissue' or 'beads' (passed through to ``mie_scatter_and_anisotropy``).
#     :type sphere_type: str
#     :param tot_vol_ml: Total phantom volume (mL).
#     :type tot_vol_ml: float
#     :param per_sol_percent: Percent solids in the bead stock solution (%).
#     :type per_sol_percent: float
#     :param mode: Selection mode, either:
#                  - ``'scattering'``: supply ``redscat_coef_mm_inv`` (mm⁻¹) to target μₛ′.
#                  - ``'beads'``: supply ``bead_volumes_ul`` (μL) to fix bead volume.
#     :type mode: str
#     :param redscat_coef_mm_inv: Target reduced scattering coefficients (mm⁻¹),
#                                 one per phantom. Required if ``mode=='scattering'``.
#     :type redscat_coef_mm_inv: numpy.ndarray or None
#     :param bead_volumes_ul: Bead volumes to add (μL), one per phantom.
#                              Required if ``mode=='beads'``.
#     :type bead_volumes_ul: numpy.ndarray or None
#
#     :returns: A dictionary with keys:
#
#         - **wavelengths_um**: 1D array of wavelengths (μm).
#         - **wavelengths_nm**: 1D array of wavelengths (nm).
#         - **red_scat_um_inv**: Reduced scattering curve μₛ′(λ) in μm⁻¹.
#         - **volume_distribution_ul**: (N×3) array of (beads, water, total) in μL.
#         - **musp_cm_inv**: List of 1D arrays of μₛ′(λ) in cm⁻¹, one per phantom.
#         - **updated_redscat_coef_mm_inv**: Final reduced scattering coefficients (mm⁻¹),
#           either the input (``mode=='scattering'``) or recomputed (``mode=='beads'``).
#     :rtype: dict
#     """
#     # 1) Compute Mie efficiencies via mie_scatter_and_anisotropy
#     wavelengths_um, Qsca, Qback, g = mie_scatter_and_anisotropy(
#         l_min=l_min,
#         l_max=l_max,
#         dl=dl,
#         r=bead_radius_um,
#         sphere_type=sphere_type,
#     )
#
#     # 2) Build reduced scattering curve: red_scat_um_inv = Qsca·π·r²·(1 − g)
#     Tsca = Qsca * np.pi * bead_radius_um ** 2
#     red_scat_um_inv = Tsca * (1.0 - g)
#
#     # 3) Convert wavelengths to nm
#     wavelengths_nm = wavelengths_um * 1e3
#
#     # 4) Compute volume distributions & final redscat coefficients
#     if mode == 'scattering':
#         if redscat_coef_mm_inv is None:
#             raise ValueError("Must supply redscat_coef_mm_inv for mode='scattering'")
#         volume_dist_ul = compute_volume_distribution_by_scattering(
#             redscat_coef_mm_inv=redscat_coef_mm_inv,
#             red_scat_curve=red_scat_um_inv,
#             tot_vol_ml=tot_vol_ml,
#             per_sol_percent=per_sol_percent,
#             act_bead_diameter_um=bead_radius_um * 2,
#         )
#         updated_redscat = redscat_coef_mm_inv
#
#     elif mode == 'beads':
#         if bead_volumes_ul is None:
#             raise ValueError("Must supply bead_volumes_ul for mode='beads'")
#         volume_dist_ul, updated_redscat = compute_volume_distribution_by_bead_number(
#             bead_volume_ul=bead_volumes_ul,
#             tot_vol_ml=tot_vol_ml,
#             per_sol_percent=per_sol_percent,
#             act_bead_diameter_um=bead_radius_um * 2,
#             red_scat_curve=red_scat_um_inv,
#         )
#
#     else:
#         raise ValueError("mode must be 'scattering' or 'beads'")
#
#     # 5) Compute μs' profiles for each phantom in cm⁻¹
#     #    density is derived inside each helper; we re-extract it here:
#     if mode == 'scattering':
#         # density_i = redscat_coef_mm_inv[i] / red_scat_um_inv[22] / 1000
#         density = redscat_coef_mm_inv / (red_scat_um_inv[22] * 1e3)
#         factor = 1e4  # convert μm⁻¹ to cm⁻¹
#     else:
#         # bead mode: density = updated_redscat / red_scat_um_inv[22] / 1000
#         density = updated_redscat / (red_scat_um_inv[22] * 1e3)
#         factor = 10  # original MATLAB used *10 for bead mode
#
#     musp_cm_inv = [
#         (red_scat_um_inv * dens) * factor
#         for dens in np.atleast_1d(density)
#     ]
#
#     return {
#         'wavelengths_nm': wavelengths_nm,
#         'volume_distribution_ul': volume_dist_ul,
#         'musp_cm_inv': musp_cm_inv,
#         'updated_redscat_coef_mm_inv': updated_redscat,
#     }

def generate_phantom_profiles(
        *,
        l_min: float,
        l_max: float,
        dl: float,
        bead_radius_um: float,
        sphere_type: str,
        tot_vol_ml: float,
        per_sol_percent: float,
        mode: str = "both",
        red_scat_coef: np.ndarray = None,
        bead_volumes_ul: np.ndarray = None,
        ref_wavelength: float = 630,
) -> dict:
    """
    Compute bead volumes and reduced‑scattering profiles for phantom preparation.

    :param l_min: Minimum wavelength (nm).
    :param l_max: Maximum wavelength (nm).
    :param dl: Wavelength step size (nm).
    :param bead_radius_um: Bead radius (μm).
    :param sphere_type: 'tissue' or 'beads'.
    :param tot_vol_ml: Total phantom volume (mL).
    :param per_sol_percent: Percent solids in the bead stock solution (%).
    :param mode: One of
        - 'scattering': compute bead volumes to achieve target μₛ′
        - 'beads'     : compute μₛ′ resulting from given bead volumes
        - 'both'      : do both in sequence (and sanity‑check)
    :param red_scat_coef: Array of target reduced scattering coefficients μₛ′ (cm⁻¹).
                          Required if mode includes 'scattering'.
    :param bead_volumes_ul: Array of bead volumes to add (μL).
                             Required if mode includes 'beads'.
    :param ref_wavelength: Reference wavelength (nm) at which to match μₛ′.
    :returns: A dict containing (keys depend on `mode`):

      - **wavelengths_nm** (np.ndarray): wavelengths in nm.

      - **bead_volumes_ul_scatter** (np.ndarray):
        μL of beads needed to hit each `redscat_coef` (mode 'scattering' or 'both').

      - **volume_distribution_scatter_ul** (np.ndarray):
        shape (P×3), columns = [beads, water, total] μL (mode 'scattering' or 'both').

      - **volume_distribution_bead_ul** (np.ndarray):
        shape (P×3), columns = [beads, water, total] μL (mode 'beads' or 'both').

      - **musp_profiles_cm_inv_scatter** (list[np.ndarray]):
        full μₛ′(λ) curves in cm⁻¹ for each phantom (mode 'scattering' or 'both').

      - **musp_profiles_cm_inv_beads** (list[np.ndarray]):
        full μₛ′(λ) curves in cm⁻¹ for each phantom (mode 'beads' or 'both').
    """
    # --- 0) convert units ---
    l_min *= 1e-3  # nm -> um
    l_max *= 1e-3  # nm -> um
    dl *= 1e-3  # nm -> um
    ref_wavelength *= 1e-3  # nm -> um
    red_scat_coef /= 10  # cm-1 -> mm_1

    # --- 1) get the red‑scattering cross‑section curve per sphere ---
    wavelengths_um, Qsca, Qback, g = mie_scatter_and_anisotropy(
        l_min=l_min, l_max=l_max, dl=dl,
        r=bead_radius_um, sphere_type=sphere_type
    )
    wavelengths_nm = wavelengths_um * 1e3
    sigma_scat_um2 = Qsca * np.pi * bead_radius_um ** 2 * (1.0 - g)

    # find the index of the reference wavelength
    idx = np.argmin(np.abs(wavelengths_um - ref_wavelength))

    out = {
        "wavelengths_nm": wavelengths_nm,
    }

    # --- 2a) scattering mode: target μₛ′ → bead volumes ---
    if mode in ("scattering", "both"):
        if red_scat_coef is None:
            raise ValueError("need redscat_coef_mm_inv for scattering mode")
        vol_dist = compute_volume_distribution_by_scattering(
            redscat_coef_mm_inv=red_scat_coef,
            red_scat_curve=sigma_scat_um2,
            tot_vol_ml=tot_vol_ml,
            per_sol_percent=per_sol_percent,
            act_bead_diameter_um=bead_radius_um * 2,
            ref_wavelength_idx=idx,
        )
        bead_ul = vol_dist[:, 0]  # first column = bead volumes
        out["bead_volumes_ul_scatter"] = bead_ul
        out["volume_distribution_scatter_ul"] = vol_dist

    # --- 2b) bead mode: bead volumes → resulting μₛ′ ---
    if mode in ("beads", "both"):
        if mode == "beads" and bead_volumes_ul is None:
            raise ValueError("need bead_volumes_ul for bead mode")
        # if 'both', override the passed‑in bead_volumes with the scatter‑computed ones
        bv = (
            bead_volumes_ul
            if mode == "beads"
            else out["bead_volumes_ul_scatter"]
        )
        vol_dist_b, mu_mm = compute_volume_distribution_by_bead_number(
            bead_volume_ul=bv,
            tot_vol_ml=tot_vol_ml,
            per_sol_percent=per_sol_percent,
            act_bead_diameter_um=bead_radius_um * 2,
            red_scat_curve=sigma_scat_um2,
            ref_wavelength_idx=idx,
        )
        out["volume_distribution_bead_ul"] = vol_dist_b.squeeze()

    # --- 3) build full μₛ′(λ) in cm⁻¹ curves for each phantom ---
    # density = μₛ′(mm⁻¹)/(σ_scat_um2[idx]*1e3) ;  factor = 1e4 (scattering mode)
    # or factor = 10 (bead mode) – match your original MATLAB logic
    if mode in ("scattering", "both"):
        density_sc = red_scat_coef / (sigma_scat_um2[idx] * 1e3)
        factor_sc = 1e4
        out["musp_profiles_cm_inv_scatter"] = np.array([
            sigma_scat_um2 * d * factor_sc
            for d in np.atleast_1d(density_sc)
        ]).squeeze()
    if mode in ("beads", "both"):
        # extract densities from mu_mm:  mu_mm = density * σ_scat_um2[idx] *1000
        dens_b = np.array(mu_mm) / (sigma_scat_um2[idx] * 1e3)
        factor_b = 10
        out["musp_profiles_cm_inv_beads"] = np.array([
            sigma_scat_um2 * d * factor_b
            for d in dens_b
        ]).squeeze()

    return out

