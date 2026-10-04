"""Save high-precision CLASS background/ODE references for ACE Neff presets.

Run with a Python environment containing CLASS 3.3.4. Outputs text beside this
script. Matching CAMB 2.0.4 comparisons were performed before implementation;
this fixture deliberately targets CLASS's scale-independent background ODE.
"""
from pathlib import Path
import platform
import classy

TCMB, TREF, NREF = 2.7255, 0.71611, 3.044
NPER = (TREF / (4 / 11) ** (1 / 3)) ** 4
URREF = NREF - 3 * NPER
CASES = [
    ("zeros", (0.0, 0.0, 0.0), -1.0, 0.0),
    ("single", (0.06, 0.0, 0.0), -1.0, 0.0),
    ("unequal", (0.01, 0.02, 0.03), -1.0, 0.0),
    ("heavy", (0.25, 0.25, 0.25), -1.0, 0.0),
    ("w0wa", (0.01, 0.02, 0.03), -0.9, -0.3),
]
ZS = (0.0, 0.5, 1.0, 3.0, 5.0, 100.0, 1100.0, 1e4)
with (Path(__file__).parent / "class_neff_reference.txt").open("w") as out:
    out.write(f"# CLASS {classy.__version__} {classy.__file__}; Python {platform.python_version()}\n")
    out.write("# h=.67 omega_b=.0224 omega_cdm=.12 Tcmb=2.7255 YHe=.245 deg_ncdm=1\n")
    out.write("# tol_ncdm_bg=1e-12 tol_background_integration=1e-12 background_integration_stepsize=.1/3\n")
    out.write("# zero mass species transferred to N_ur; w0wa uses PPF, Omega_Lambda=0\n")
    out.write("# D/f are scale_independent_growth_factor/_f, not transfer-derived growth\n")
    out.write("# case policy Neff m1 m2 m3 w0 wa z H_km_s_Mpc chi_Mpc D_normalized f\n")
    for policy, neffs in (("temperature", (2.0, 3.044, 5.0)), ("radiation", (3.044, 5.0))):
        for name, masses, w0, wa in CASES:
            for neff in neffs:
                scale = neff / NREF
                temp = TREF * scale ** 0.25 if policy == "temperature" else TREF
                ur = URREF * scale if policy == "temperature" else neff - 3 * NPER
                massive = [m for m in masses if m > 0]
                ur += (3 - len(massive)) * (temp / (4 / 11) ** (1 / 3)) ** 4
                params = dict(h=.67, omega_b=.0224, omega_cdm=.12, T_cmb=TCMB,
                              YHe=.245, N_ncdm=len(massive), N_ur=ur,
                              tol_ncdm_bg=1e-12, tol_background_integration=1e-12,
                              background_integration_stepsize=.1/3)
                if massive:
                    params.update(m_ncdm=",".join(map(str, massive)),
                                  T_ncdm=",".join([str(temp)] * len(massive)),
                                  deg_ncdm=",".join(["1"] * len(massive)))
                if (w0, wa) != (-1.0, 0.0):
                    params.update(Omega_Lambda=0.0, w0_fld=w0, wa_fld=wa, use_ppf="yes")
                c = classy.Class()
                c.set(params)
                c.compute(["background"])
                for z in ZS:
                    values = (c.Hubble(z)*299792.458, c.angular_distance(z)*(1+z),
                              c.scale_independent_growth_factor(z), c.scale_independent_growth_factor_f(z))
                    out.write(f"{name} {policy} {neff} {' '.join(map(str,masses))} {w0} {wa} {z} "
                              + " ".join(f"{v:.16e}" for v in values) + "\n")
                c.struct_cleanup()
                c.empty()
