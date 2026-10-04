"""CLASS reference values for the three-independent-mass neutrino path of
BackgroundCosmologyExt (AbstractCosmologicalEmulators.jl).

Usage (classy 3.3.4 was used, miniconda base env):
    python generate_class_reference.py

Writes two plain-text fixtures next to this script:
  class_background_three_mass.txt  H(z)/H0, distances, rho_ncdm, rho_ur, rho_g
  class_growth_three_mass.txt      synchronous-gauge delta_cb / delta_m growth
                                   ratios D(k,z)/D(k,0) and f(k,z) = dln delta/dln a

Neutrino convention (CLASS standard Fermi-Dirac anchor):
  three ncdm species, deg_ncdm = 1, T_ncdm = 0.71611 T_cmb, T_cmb = 2.7255 K,
  N_ur = 3.044 - 3 (0.71611 / (4/11)^(1/3))^4  (massless remainder, Neff = 3.044).
A species with exactly zero mass is passed to CLASS as massless radiation with the
same temperature, i.e. its (0.71611/(4/11)^(1/3))^4 is added to N_ur (CLASS ncdm
species need m > 0). This is the identical density for an m = 0 FD species.

Dark energy: Lambda when (w0, wa) = (-1, 0), otherwise CPL fluid (PPF) filling
the closure, Omega_Lambda = 0, so both codes close with the same sum rule.
"""
import platform
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from classy import Class
import classy

T_CMB = 2.7255
T_NCDM = 0.71611
NEFF = 3.044
NEFF_PER_FD = (T_NCDM / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4
N_UR_3NU = NEFF - 3.0 * NEFF_PER_FD

# id, (m1, m2, m3) [eV], h, omega_b, omega_cdm, Omega_k, w0, wa
BASE = dict(h=0.67, omega_b=0.0224, omega_cdm=0.12, Omega_k=0.0, w0=-1.0, wa=0.0)
CASES = [
    ("zeros", (0.0, 0.0, 0.0), BASE),
    ("single_0.06", (0.06, 0.0, 0.0), BASE),
    ("degenerate_0.1", (0.1, 0.1, 0.1), BASE),
    ("unequal_0.01_0.02_0.03", (0.01, 0.02, 0.03), BASE),
    ("NH_min", (0.0, 0.0086, 0.0502), BASE),
    ("IH_min", (0.0492, 0.05, 0.0), BASE),
    ("degenerate_0.25", (0.25, 0.25, 0.25), BASE),
    ("w0wa_unequal", (0.01, 0.02, 0.03),
     dict(h=0.7, omega_b=0.022, omega_cdm=0.11, Omega_k=0.0, w0=-0.9, wa=-0.3)),
    ("curved_degenerate", (0.1, 0.1, 0.1),
     dict(h=0.65, omega_b=0.023, omega_cdm=0.13, Omega_k=0.02, w0=-1.0, wa=0.0)),
]
GROWTH_CASES = ["zeros", "single_0.06", "NH_min", "unequal_0.01_0.02_0.03",
                "degenerate_0.1", "degenerate_0.25"]

Z_BG = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0, 1100.0, 1.0e4]
Z_GROWTH = [0.0, 1.0, 3.0, 5.0]
K_GROWTH = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0]  # h/Mpc
FD_STEP = 0.02     # step in ln a for f(k,z)
FD_STEP_ALT = 0.04  # second step: |f(h)-f(h_alt)| reported as FD uncertainty

PRECISION = {
    "tol_ncdm_bg": 1e-10,
    "tol_ncdm_synchronous": 1e-6,
    "tol_perturbations_integration": 1e-7,
    "perturbations_sampling_stepsize": 0.02,
    "k_per_decade_for_pk": 40,
}


def class_params(masses, cosmo, growth):
    massive = [m for m in masses if m > 0.0]
    n_zero = len(masses) - len(massive)
    p = {
        "h": cosmo["h"], "omega_b": cosmo["omega_b"], "omega_cdm": cosmo["omega_cdm"],
        "Omega_k": cosmo["Omega_k"], "T_cmb": T_CMB,
        "N_ur": N_UR_3NU + n_zero * NEFF_PER_FD,
        "N_ncdm": len(massive),
        "tol_ncdm_bg": PRECISION["tol_ncdm_bg"],
    }
    if massive:
        p["m_ncdm"] = ",".join(repr(m) for m in massive)
        p["T_ncdm"] = ",".join([repr(T_NCDM)] * len(massive))
        p["deg_ncdm"] = ",".join(["1"] * len(massive))
    if (cosmo["w0"], cosmo["wa"]) != (-1.0, 0.0):
        p.update({"Omega_Lambda": 0.0, "w0_fld": cosmo["w0"], "wa_fld": cosmo["wa"],
                  "use_ppf": "yes"})
    if growth:
        p.update({"output": "mTk", "z_max_pk": 6.0, "P_k_max_h/Mpc": 2.0})
        p.update({k: v for k, v in PRECISION.items() if k != "tol_ncdm_bg"})
    return p


def header(io, what):
    io.write(f"# {what}\n")
    io.write("# generator: test/fixtures/neutrino_three_mass/generate_class_reference.py\n")
    io.write(f"# classy: {getattr(classy, '__version__', '?')} ({classy.__file__})\n")
    io.write("# CLASS source of the sibling checkout: class_public v3.3.4 e8580832\n")
    io.write(f"# python: {sys.version.split()[0]} numpy: {np.__version__} host: {platform.node()}\n")
    io.write(f"# T_cmb={T_CMB} T_ncdm/T_cmb={T_NCDM} deg_ncdm=1 Neff={NEFF} "
             f"N_ur(3 massive)={N_UR_3NU:.10f} Neff_per_FD_species={NEFF_PER_FD:.10f}\n")
    io.write("# zero masses -> massless species added to N_ur with the same temperature\n")
    io.write(f"# precision: {PRECISION}\n")


def growth_ratio_and_f(cosmo, z, h_step):
    """delta(k,z)/delta(k,0) and f = dln delta/dln a for d_cb and d_m."""
    hh = cosmo.h()
    lna0 = -np.log1p(z)
    if z > 0:
        offs, coef = np.array([-2, -1, 1, 2]), np.array([1, -8, 8, -1]) / 12.0
    else:  # one-sided 4th order, a <= 1 only
        offs, coef = np.array([0, -1, -2, -3, -4]), np.array([25, -48, 36, -16, 3]) / 12.0

    def delta_at(zz):
        tk = cosmo.get_transfer(zz)
        k = tk["k (h/Mpc)"]
        dm = tk["d_m"]
        # delta_cb from the rho-weighted b and cdm transfers (weights are a-independent)
        wb, wc = cosmo.omega_b(), cosmo.Omega0_cdm() * hh**2
        dcb = (wb * tk["d_b"] + wc * tk["d_cdm"]) / (wb + wc)
        lk = np.log(k)
        return (np.interp(np.log(K_GROWTH), lk, dcb), np.interp(np.log(K_GROWTH), lk, dm))

    dcb0, dm0 = delta_at(0.0)
    dcbz, dmz = delta_at(z)
    logd_cb = []
    logd_m = []
    for o in offs:
        zz = np.expm1(-(lna0 + o * h_step))
        zz = max(zz, 0.0)
        a, b = delta_at(zz)
        logd_cb.append(np.log(np.abs(a)))
        logd_m.append(np.log(np.abs(b)))
    logd_cb, logd_m = np.array(logd_cb), np.array(logd_m)
    f_cb = coef @ logd_cb / h_step
    f_m = coef @ logd_m / h_step
    return dcbz / dcb0, dmz / dm0, f_cb, f_m


def main():
    bg = open("class_background_three_mass.txt", "w")
    header(bg, "CLASS background reference for three independent neutrino masses")
    bg.write("# rho_X are rho_X(z)/rho_crit,0 (CLASS (.)rho_X / H0^2); distances in Mpc\n")
    bg.write("# columns: case m1 m2 m3 h omega_b omega_cdm Omega_k w0 wa z "
             "H_over_H0 dM dA dL rho_ncdm rho_ur rho_g\n")
    gr = open("class_growth_three_mass.txt", "w")
    header(gr, "CLASS synchronous-gauge growth diagnostics for three independent neutrino masses")
    gr.write("# Dcb, Dm: d_cb(k,z)/d_cb(k,0), d_m(k,z)/d_m(k,0); f = dln d/dln a by 4th-order FD in ln a\n")
    gr.write(f"# FD step {FD_STEP}; df_* = |f(step {FD_STEP}) - f(step {FD_STEP_ALT})| (FD/interp. uncertainty)\n")
    gr.write("# d_cb = (omega_b d_b + omega_cdm d_cdm)/(omega_b + omega_cdm); equals d_m when no species is massive\n")
    gr.write("# columns: case m1 m2 m3 h omega_b omega_cdm Omega_k w0 wa z k_hMpc "
             "Dcb Dm f_cb f_m df_cb df_m\n")
    for cid, masses, c in CASES:
        growth = cid in GROWTH_CASES
        cosmo = Class()
        cosmo.set(class_params(masses, c, growth))
        cosmo.compute()
        H0 = cosmo.Hubble(0.0)
        b = cosmo.get_background()
        lna_all = -np.log1p(b["z"])
        lna_tab, order = np.unique(lna_all, return_index=True)  # strictly increasing ln a

        def tab(col):
            return b[col][order]
        ncdm_cols = [k for k in b if k.startswith("(.)rho_ncdm[")]
        rho_ncdm_tab = sum(tab(k) for k in ncdm_cols) if ncdm_cols else np.zeros_like(lna_tab)
        ln_rho_ncdm = CubicSpline(lna_tab, np.log(rho_ncdm_tab)) if ncdm_cols else None
        ln_rho_ur = CubicSpline(lna_tab, np.log(tab("(.)rho_ur")))
        ln_rho_g = CubicSpline(lna_tab, np.log(tab("(.)rho_g")))
        pre = f"{cid} {masses[0]!r} {masses[1]!r} {masses[2]!r} {c['h']!r} {c['omega_b']!r} " \
              f"{c['omega_cdm']!r} {c['Omega_k']!r} {c['w0']!r} {c['wa']!r}"
        for z in Z_BG:
            la = -np.log1p(z)
            Hz = cosmo.Hubble(z)
            dA = cosmo.angular_distance(z)
            dL = cosmo.luminosity_distance(z)
            dM = dA * (1.0 + z)
            rn = np.exp(ln_rho_ncdm(la)) / H0**2 if ln_rho_ncdm is not None else 0.0
            ru = np.exp(ln_rho_ur(la)) / H0**2
            rg = np.exp(ln_rho_g(la)) / H0**2
            bg.write(f"{pre} {z!r} {Hz / H0:.15e} {dM:.15e} {dA:.15e} {dL:.15e} "
                     f"{rn:.15e} {ru:.15e} {rg:.15e}\n")
        if growth:
            for z in Z_GROWTH:
                Dcb, Dm, fcb, fm = growth_ratio_and_f(cosmo, z, FD_STEP)
                _, _, fcb2, fm2 = growth_ratio_and_f(cosmo, z, FD_STEP_ALT)
                for i, k in enumerate(K_GROWTH):
                    gr.write(f"{pre} {z!r} {k!r} {Dcb[i]:.12e} {Dm[i]:.12e} {fcb[i]:.12e} "
                             f"{fm[i]:.12e} {abs(fcb[i] - fcb2[i]):.3e} {abs(fm[i] - fm2[i]):.3e}\n")
        cosmo.struct_cleanup()
        cosmo.empty()
        print("done", cid, flush=True)
    bg.close()
    gr.close()


if __name__ == "__main__":
    main()
