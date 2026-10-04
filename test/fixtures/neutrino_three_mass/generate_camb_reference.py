"""CAMB background reference values for the three-independent-mass neutrino path of
BackgroundCosmologyExt (AbstractCosmologicalEmulators.jl).

Usage (camb 2.0.0 was used, miniconda base env; NOT the current 2.0.4 tag):
    python generate_camb_reference.py

Writes camb_background_three_mass.txt next to this script (same cases / redshifts
as generate_class_reference.py).

This CAMB 2.0.0 fixture does NOT use the CLASS thermal convention:
  * every CAMB species has temperature T_nu = (4/11)^(1/3) T_cmb; a "degeneracy" g
    multiplies its density, rho_i = g * rho_FD(T_nu, m_i), with y = m_i / (k_B T_nu);
  * masses are not inputs: CAMB solves for y_i from omnuh2 * nu_mass_fraction_i.
Here each nonzero mass is one eigenstate with g = (0.71611 / (4/11)^(1/3))^4
= 1.0132016 (the CLASS per-species Neff), share_delta_neff = False, and
num_nu_massless = 3.044 - (number of massive eigenstates) * g (zero masses are
massless with the same g), so Neff = 3.044 and the relativistic densities equal
CLASS. omnuh2 is chosen (by quadrature below) so that CAMB's y_i reproduces the
physical mass m_i at T_nu. The non-relativistic density then differs from the CLASS
anchor (deg 1, T = 0.71611 T_cmb) by g ((4/11)^(1/3) / 0.71611)^3 - 1
= 0.71611 / (4/11)^(1/3) - 1 = +0.328 %; a genuine convention difference.
This describes this fixture's setup, not a limitation of CAMB in general.
CAMB 2.0.4 has a newer physical-mass mapping and is not exercised here.

CAMB thermal density is a fit, not quadrature: rho/P use series for a*y <= 0.42,
an asymptotic expansion for a*y >= 70 and a smooth fit in between (CAMB source
fortran/massive_neutrinos.f90 in the fetched 2.0.4 source states "max relative
errors ... about 2.4e-5 in rho"; that comment alone does not establish the
installed 2.0.0 binary's accuracy). Measured for the installed camb 2.0.0 against direct quadrature
(photon-normalised, so constants cancel): |rel| < 1.1e-6 for a*y < 0.1 and
< 1e-8 for a*y > 75, oscillating up to 2.28e-5 (at a*y = 0.56) in between.
The rho_nu_massive column carries that error; it is not an ACE error.

Dark energy: Lambda when (w0, wa) = (-1, 0), otherwise CPL with PPF (crosses -1).
"""
import platform
import sys

import numpy as np
from scipy.integrate import quad
import camb
from camb import constants as cc

from generate_class_reference import CASES, Z_BG, T_CMB, T_NCDM, NEFF

T_NU_CAMB = (4.0 / 11.0) ** (1.0 / 3.0)            # T_nu / T_cmb in CAMB
G_PER_SPECIES = (T_NCDM / T_NU_CAMB) ** 4          # CLASS per-species Neff
KB_EV = cc.k_B / cc.eV                             # CAMB constants, eV/K


def rho_hat(y):
    """FD density at mass parameter y over its massless value (CAMB's normalisation)."""
    num = quad(lambda x: x * x * np.sqrt(x * x + y * y) * np.exp(-x) / (1.0 + np.exp(-x)), 0, np.inf,
               epsabs=0, epsrel=1e-13, limit=400)[0]
    return num / (7.0 * np.pi ** 4 / 120.0)


def camb_params(masses, c):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * c["h"], ombh2=c["omega_b"], omch2=c["omega_cdm"],
                       omk=c["Omega_k"], mnu=0.0, nnu=NEFF, num_massive_neutrinos=0,
                       TCMB=T_CMB)
    massive = [m for m in masses if m > 0]
    n = len(massive)
    pars.num_nu_massless = NEFF - n * G_PER_SPECIES
    if n:
        kT = KB_EV * T_NU_CAMB * T_CMB
        # omnuh2_i = g * omega_r,1 * rho_hat(y_i), omega_r,1 = 7/8 (4/11)^(4/3) omega_gamma
        om_g = (pars.TCMB / cc.COBE_CMBTemp) ** 4 * OMEGA_G_COBE
        om_r1 = 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * om_g
        omi = [G_PER_SPECIES * om_r1 * rho_hat(m / kT) for m in massive]
        pars.omnuh2 = sum(omi)
        pars.num_nu_massive = n
        pars.nu_mass_eigenstates = n
        pars.share_delta_neff = False
        pars.nu_mass_degeneracies = [G_PER_SPECIES] * n
        pars.nu_mass_fractions = [o / sum(omi) for o in omi]
        pars.nu_mass_numbers = [1] * n
    if (c["w0"], c["wa"]) != (-1.0, 0.0):
        pars.set_dark_energy(w=c["w0"], wa=c["wa"], dark_energy_model="ppf")
    pars.WantTransfer = False
    pars.WantCls = False
    return pars


# omega_gamma at the COBE temperature from CAMB's own background (massless run)
def _omega_g_cobe():
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.0, ombh2=0.0224, omch2=0.12, mnu=0.0, nnu=NEFF,
                    num_massive_neutrinos=0, TCMB=cc.COBE_CMBTemp)
    r = camb.get_background(p)
    d = r.get_background_densities(1.0, vars=["photon"])
    return float(d["photon"][0] / r.grhocrit) * (0.67) ** 2


OMEGA_G_COBE = _omega_g_cobe()


def main():
    out = open("camb_background_three_mass.txt", "w")
    w = out.write
    w("# CAMB background reference for three independent neutrino masses\n")
    w("# generator: test/fixtures/neutrino_three_mass/generate_camb_reference.py\n")
    w(f"# camb: v{camb.__version__} ({camb.__file__}); current upstream tag 2.0.4 differs, not used\n")
    w(f"# python: {platform.python_version()} numpy: {np.__version__} host: {platform.node()}\n")
    w(f"# T_cmb={T_CMB} CAMB T_nu/T_cmb=(4/11)^(1/3)={T_NU_CAMB:.10f} g per massive eigenstate="
      f"{G_PER_SPECIES:.10f} share_delta_neff=False Neff={NEFF}\n")
    w("# zero masses -> massless with the same g (num_nu_massless = 3.044 - n_massive g)\n")
    w("# CAMB NR neutrino density = (1 + 0.0033) x CLASS anchor at equal physical mass (convention)\n")
    w(f"# omega_gamma(T_cmb) from CAMB = {OMEGA_G_COBE:.12e}\n")
    w("# rho_X are rho_X(z)/rho_crit,0; distances in Mpc (dA, dL from CAMB, dM = (1+z) dA)\n")
    w("# y_i = CAMB nu_masses read back; m_i = y_i k_B T_nu (eV) agrees with input when massive\n")
    w("# columns: case m1 m2 m3 h omega_b omega_cdm Omega_k w0 wa z "
      "H_over_H0 dM dA dL rho_nu_massive rho_nu_massless rho_g\n")
    for cid, masses, c in CASES:
        pars = camb_params(masses, c)
        res = camb.get_background(pars)
        massive = [m for m in masses if m > 0]
        if massive:
            kT = KB_EV * T_NU_CAMB * T_CMB
            ys = np.array(res.nu_masses[: len(massive)])
            back = ys * kT
            assert np.allclose(sorted(back), sorted(massive), rtol=1e-6), (cid, back, massive)
            w(f"# {cid}: CAMB m_i read back {', '.join(f'{b:.9g}' for b in back)} eV\n")
        H0 = 100 * c["h"]
        for z in Z_BG:
            a = 1.0 / (1.0 + z)
            d = res.get_background_densities(a, vars=["photon", "neutrino", "nu"])
            conv = 1.0 / (res.grhocrit * a ** 4)
            dA = float(res.angular_diameter_distance(z))
            dL = float(res.luminosity_distance(z))
            Hz = float(res.hubble_parameter(z)) / H0
            vals = [Hz, dA * (1 + z), dA, dL,
                    float(d["nu"][0]) * conv, float(d["neutrino"][0]) * conv,
                    float(d["photon"][0]) * conv]
            w(f"{cid} {masses[0]!r} {masses[1]!r} {masses[2]!r} {c['h']!r} {c['omega_b']!r} "
              f"{c['omega_cdm']!r} {c['Omega_k']!r} {c['w0']!r} {c['wa']!r} {z!r} "
              + " ".join(f"{v:.15e}" for v in vals) + "\n")
    out.close()


if __name__ == "__main__":
    sys.exit(main())
