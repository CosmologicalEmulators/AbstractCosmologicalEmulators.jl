"""Regenerate matched CLASS background and jaxace values, saved as text.

CLASS settings follow test/fixtures/neutrino_neff/generate_class_reference.py.
Density residuals use CLASS's native background rows, not interpolated densities.
"""
import json
import argparse
import importlib.metadata
import subprocess
from pathlib import Path

import classy
import numpy as np
from scipy.interpolate import CubicSpline
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxace import background as bg
from jaxace import _neutrinos

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path,
                    default=Path(__file__).resolve().parent / 'output')
OUT = parser.parse_args().output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
H, WB, WC = .67, .0224, .12
MASSES = (.01, .02, .03)
NREF, NEFF, TREF = 3.044, 4.0, .71611
NPER = (TREF / (4 / 11) ** (1 / 3)) ** 4
URREF = NREF - 3 * NPER


def compute(policy, neff, step=.01):
    scale = neff / NREF
    temp = TREF * scale**.25 if policy == "temperature" else TREF
    nur = URREF * scale if policy == "temperature" else neff - 3 * NPER
    params = dict(
        h=H, omega_b=WB, omega_cdm=WC, T_cmb=2.7255, YHe=.245,
        N_ncdm=3, m_ncdm=','.join(map(str, MASSES)),
        T_ncdm=','.join([repr(temp)] * 3), deg_ncdm='1,1,1', N_ur=nur,
        tol_ncdm_bg=1e-12, tol_background_integration=1e-12,
        background_integration_stepsize=step,
    )
    c = classy.Class()
    c.set(params)
    c.compute(['background'])
    b = c.get_background()
    a = 1 / (1 + b['z'])
    rho = sum(b[k] for k in b if k.startswith('(.)rho_ncdm[')) + b['(.)rho_ur']
    rho /= (100 * H / 299792.458)**2
    hubble = b['H [1/Mpc]'] * 299792.458
    c.struct_cleanup()
    c.empty()
    return a, rho, hubble, params


base_a, base_rho, base_h, base_params = compute('temperature', NREF)
base_rho_interp = CubicSpline(np.log(base_a), np.log(base_rho))
base_h_interp = CubicSpline(np.log(base_a), np.log(base_h))
metadata = dict(
    CLASS_version=classy.__version__,
    jax_version=jax.__version__,
    jaxace_version=importlib.metadata.version('jaxace'),
    mass_eV=MASSES, Neff=NEFF, baseline_Neff=NREF,
    density_definition='rho_nu = sum of all ncdm densities plus ur; units rho_crit,0',
    grid='Selected native CLASS background rows; baseline interpolated in log(a), log(quantity)',
    CLASS_parameters={},
)
for policy in ('temperature', 'radiation'):
    a, rho, hubble, params = compute(policy, NEFF)
    original_grid = a.copy()
    valid = np.flatnonzero((a >= 1e-5) & (a <= 1))
    idx = valid[np.unique(np.round(np.linspace(0, len(valid)-1, 321)).astype(int))]
    a, rho, hubble = a[idx], rho[idx], hubble[idx]
    ref_rho = np.exp(base_rho_interp(np.log(a)))
    ref_h = np.exp(base_h_interp(np.log(a)))
    data = np.column_stack((a, rho, hubble, ref_rho, ref_h))
    np.savetxt(OUT / f'class_{policy}.txt', data,
               header='a rho_nu_over_rhocrit0 H_km_s_Mpc baseline_rho_nu_over_rhocrit0 baseline_H_km_s_Mpc')
    ar = jnp.asarray(a)
    jrho = np.asarray(bg.ΩνE2(ar, _neutrinos.OMEGA_GAMMA_H2/H**2,
                            jnp.asarray(MASSES), NEFF, policy))
    jh = np.asarray(100*H*bg.E_a(ar, (WB+WC)/H**2, H, mν=MASSES,
                                Neff=NEFF, neutrino_prescription=policy))
    np.savetxt(OUT / f'jaxace_{policy}.txt', np.column_stack((a, jrho, jh)),
               header='a rho_nu_over_rhocrit0 H_km_s_Mpc')
    # A smaller requested step may leave CLASS's output grid unchanged.
    # Record that explicitly; in that case this is not a resolution test.
    af, rf, hf, _ = compute(policy, NEFF, step=.005)
    metadata[f'CLASS_grid_check_{policy}'] = dict(
        original_rows=len(original_grid), requested_smaller_step_rows=len(af),
        identical_nodes=bool(np.array_equal(original_grid, af)))
    refined_rho = np.exp(CubicSpline(np.log(af), np.log(rf))(np.log(a)))
    refined_h = np.exp(CubicSpline(np.log(af), np.log(hf))(np.log(a)))
    drift = dict(rho=float(np.max(np.abs(refined_rho/rho-1))),
                 H=float(np.max(np.abs(refined_h/hubble-1))))
    assert max(drift.values()) < 1e-8, drift
    metadata['CLASS_parameters'][policy] = params
    metadata[f'CLASS_grid_refinement_{policy}'] = drift
    print(policy, 'CLASS refinement', drift, 'JAX residual',
          np.max(abs(jrho/rho-1)), np.max(abs(jh/hubble-1)), flush=True)

# Baseline value consistency under the requested step change is checked separately.
af, rf, hf, _ = compute('temperature', NREF, step=.005)
common = (af >= 1e-5) & (af <= 1)
baseline_drift = dict(
    rho=float(np.max(abs(np.exp(base_rho_interp(np.log(af[common])))/rf[common]-1))),
    H=float(np.max(abs(np.exp(base_h_interp(np.log(af[common])))/hf[common]-1))),
)
assert max(baseline_drift.values()) < 1e-8
metadata['baseline_interpolation_refinement'] = baseline_drift
def revision(path):
    # An installed package inside a repository-local venv is not that repo.
    if not (path / '.git').exists():
        return None
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

metadata['git'] = {
    'AbstractCosmologicalEmulators.jl': revision(Path(__file__).resolve().parents[3]),
    'jaxace': revision(Path(bg.__file__).resolve().parents[1]),
}
(OUT / 'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
