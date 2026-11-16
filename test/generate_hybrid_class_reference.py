"""Regenerate hybrid_class_reference.tsv with CLASS (reference used v3.3.4).

Run explicitly, not in CI. Copy the resulting text fixture to
jaxace/tests/data/hybrid_class_reference.tsv for the cross-language tests.
"""
from pathlib import Path

import numpy as np
from classy import Class, __version__


def main():
    rows = []
    for masses in ((.02, .02, .02), (.01, .02, .03)):
        for neff in (2.8, 3.044 - 1e-4, 3.044, 3.044 + 1e-4, 3.3):
            scale = neff / 3.044
            nper = (.71611 / (4 / 11)**(1 / 3))**4
            temp = .71611 * scale**.25 if neff < 3.044 else .71611
            nur = (3.044 - 3 * nper) * scale if neff < 3.044 else neff - 3 * nper
            c = Class()
            c.set(dict(
                h=.67, omega_b=.0224, omega_cdm=.12, T_cmb=2.7255, YHe=.245,
                N_ncdm=3, m_ncdm=','.join(map(str, masses)), deg_ncdm='1,1,1',
                T_ncdm=','.join([repr(temp)] * 3), N_ur=nur,
                tol_ncdm_bg=1e-12, tol_background_integration=1e-12,
                background_integration_stepsize=.005,
            ))
            try:
                c.compute(['background'])
                for z in (.5, 1., 3.):
                    rows.append([neff, *masses, z, c.Hubble(z) * 299792.458 / 67.,
                                 c.comoving_distance(z)])
            finally:
                c.struct_cleanup()
                c.empty()
    header = f'''CLASS {__version__}; generated with generate_hybrid_class_reference.py.
h=.67 omega_b=.0224 omega_cdm=.12 T_cmb=2.7255 YHe=.245 flat LCDM.
N_ncdm=3 deg_ncdm=1,1,1; Tref=.71611 Nref=3.044 Nper=(Tref/(4/11)^(1/3))^4.
Below Nref: T=Tref*(Neff/Nref)^.25 Nur=(Nref-3*Nper)*Neff/Nref.
At/above Nref: T=Tref Nur=Neff-3*Nper.
tol_ncdm_bg=1e-12 tol_background_integration=1e-12 background_integration_stepsize=.005.
Neff m1_eV m2_eV m3_eV z E chi_Mpc'''
    np.savetxt(Path(__file__).with_name('hybrid_class_reference.tsv'), rows, header=header)


if __name__ == '__main__':
    main()
