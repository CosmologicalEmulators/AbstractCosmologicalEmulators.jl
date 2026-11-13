"""Render the saved comparison; requires real LaTeX, never silently falls back."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, StrMethodFormatter

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('implementation', choices=('ace', 'jaxace'))
parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
args = parser.parse_args()
implementation = args.implementation
DATA = args.data_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
display_name = 'AbstractCosmologicalEmulators.jl' if implementation == 'ace' else 'jaxace'
assert shutil.which('latex') and shutil.which('dvipng'), 'LaTeX rendering is required'
plt.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 14,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.linewidth': .8, 'lines.linewidth': 1.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
colors = {'temperature': '#0072B2', 'radiation': '#D55E00'}
labels = {'temperature': r'Temperature rescaling', 'radiation': r'Extra massless radiation'}
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True,
                         gridspec_kw={'height_ratios': [2.2, 1], 'hspace': .12, 'wspace': .27})
metrics = {}
for policy in colors:
    reference = np.loadtxt(DATA / f'class_{policy}.txt')
    data = np.loadtxt(DATA / f'{implementation}_{policy}.txt')
    assert np.array_equal(reference[:, 0], data[:, 0])
    x = 1/reference[:, 0]
    for col, quantity in enumerate(('rho', 'H')):
        i = col + 1
        baseline = reference[:, col + 3]
        axes[0, col].plot(x, 100*(data[:, i]/baseline-1), color=colors[policy])
        markers = np.arange(0, len(x), 20)
        axes[0, col].plot(x[markers], 100*(reference[markers, i]/baseline[markers]-1),
                         linestyle='none', marker='o', markersize=4.2,
                         markerfacecolor='white', markeredgewidth=.9, color=colors[policy])
        scale = 1e8 if col == 0 else 1e9
        residual = data[:, i]/reference[:, i]-1
        axes[1, col].plot(x, scale*residual, color=colors[policy], linewidth=1.6)
        metrics[f'{policy}_{quantity}_{display_name}'] = float(np.max(np.abs(residual)))

for ax in axes.flat:
    ax.set_xscale('log')
    ax.set_xlim(1, 1e5)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(StrMethodFormatter(r'${x:g}$'))
    ax.grid(axis='y', color='.88', linewidth=.6)
    ax.tick_params(which='major', length=4)
    ax.tick_params(which='minor', length=2)
for ax in axes[1]:
    ax.axhline(0, color='.35', linewidth=.75, zorder=0)
    ax.set_xlabel(r'$1+z$')
axes[0, 0].set_title(r'Neutrino-sector energy density')
axes[0, 1].set_title(r'Expansion rate')
axes[0, 0].set_ylabel(r'$100\,[\rho_\nu/\rho_{\nu,\mathrm{ref}}-1]\;(\%)$')
axes[0, 1].set_ylabel(r'$100\,[H/H_{\mathrm{ref}}-1]\;(\%)$')
axes[1, 0].set_ylabel(r'$10^{8}\,[\rho_\nu/\rho_\nu^{\mathrm{CLASS}}-1]$')
axes[1, 1].set_ylabel(r'$10^{9}\,[H/H^{\mathrm{CLASS}}-1]$')
axes[0, 0].set_ylim(-1, 34)
axes[0, 1].set_ylim(-.2, 6.8)
axes[1, 0].set_ylim(-10, 10)
axes[1, 1].set_ylim(-2.3, 2.3)
axes[1, 1].legend(handles=[
    Line2D([], [], color='.15', label=display_name, linestyle='-'),
], loc='lower left', frameon=False, handlelength=2.0)
fig.suptitle(display_name + r': two thermal prescriptions at $N_{\mathrm{eff}}=4$',
             y=.985, fontsize=14 if implementation == 'ace' else 17)
fig.text(.5, .931,
         r'$m_\nu=(0.01,\,0.02,\,0.03)\,\mathrm{eV},\quad h=0.67,\quad '
         r'\omega_{\mathrm{b}}=0.0224,\quad\omega_{\mathrm{c}}=0.12,\quad\Omega_{\mathrm{k}}=0,\quad w=-1$',
         ha='center', fontsize=11)
fig.legend(handles=[
    Line2D([], [], color=colors[p], label=labels[p]) for p in colors
] + [Line2D([], [], color='.2', linestyle='none', marker='o', markerfacecolor='white',
            markersize=4.5, label=r'Matched CLASS')],
    loc='upper center', bbox_to_anchor=(.5, .909), ncol=3, frameon=False)
fig.subplots_adjust(left=.085, right=.98, bottom=.12, top=.805)
fig.text(.5, .025,
         r'Reference: $N_{\mathrm{eff}}=3.044$ with identical masses and $H_0$. '
         r'$\rho_\nu$ includes the massless remainder; CLASS v3.3.4.',
         ha='center', fontsize=10)
for ext in ('png', 'pdf', 'svg'):
    fig.savefig(OUT / f'thermal_prescriptions_{implementation}.{ext}', dpi=200, bbox_inches='tight')
(OUT / f'metrics_{implementation}.json').write_text(json.dumps(metrics, indent=2)+'\n')
print(json.dumps(metrics, indent=2))
print(OUT / f'thermal_prescriptions_{implementation}.png')
