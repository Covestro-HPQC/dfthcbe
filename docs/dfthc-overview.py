"""Draws the README's opening figure, dfthc-overview-light.png and -dark.png; needs pyscf."""
import json
import pathlib
import subprocess
import tempfile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci
from qualtran.resource_counting import get_cost_value, QECGatesCost

from dfthcbe import DFTHCWalk, walk_e_sos, walk_integrals

DOCS = pathlib.Path(__file__).resolve().parent
DATA = DOCS.parent / 'data' / 'dfthc'
KEYS = ('U', 'W', 'w_B', 'w_plus', 'u_plus', 'w_minus', 'u_minus')
SYSTEMS = {  # label, active space (electrons, orbitals)
    'h2_sto3g_R2_B2_C1': (r'H$_2$ STO-3G', (2, 2)),
    'h2_631g_R2_B3_C3': (r'H$_2$ 6-31G', (2, 4)),
    'naphthalene': (r'C$_{10}$H$_8$', (10, 10)),
    'fe2s2': (r'Fe$_2$S$_2$', (30, 20)),
    'fe4s4': (r'Fe$_4$S$_4$', (54, 36)),
}
SWEEP = ('h2_sto3g_R2_B2_C1', 'naphthalene')  # small enough for FCI
BITS = range(4, 17)
BOXES = (('PREP', r'$\mathrm{PREP}$'), ('PREPB', r'$\mathrm{PREP}_b$'),
         ('SEL', r'$\mathrm{SEL}_b$'), ('REF', r'$\mathrm{REF}$'))
THEMES = {  # GitHub's background and text; box hues checked for colour blindness and contrast
    'light': dict(bg='#ffffff', ink='#1f2328', ink2='#59636e', line2='#818b98', grid='#d1d9e0',
                  hues=('#1baf7a', '#eda100', '#e87ba4', '#008300'), tint=28),
    'dark': dict(bg='#0d1117', ink='#e6edf3', ink2='#9198a1', line2='#9198a1', grid='#3d444d',
                 hues=('#199e70', '#c98500', '#d55181', '#008300'), tint=45),
}


def factors(name):
    f = np.load(DATA / name / 'factors.npz')
    return {k: f[k] for k in KEYS}


def toffolis(bloq):
    return get_cost_value(bloq, QECGatesCost()).total_toffoli_only()


# --- data ---

def energy_error(name, bits):
    """``E_0 - E_FCI`` of the Hamiltonian the walk encodes, with every precision at ``bits``."""
    f, ints = factors(name), np.load(DATA / name / 'integrals.npz')
    e_gap = json.loads((DATA / name / 'summary.json').read_text())['e_gap']
    walk = DFTHCWalk.from_factors(**f, b_rot=bits, b_k1=bits, b_k2=bits, s=bits)
    enc = walk.block_encoding.encoded_factors()
    nelec = (int(ints['nalpha']), int(ints['nbeta']))
    e_0 = fci.direct_spin1.FCI().kernel(*walk_integrals(**enc), f['U'].shape[2], nelec)[0]
    return e_0 - walk_e_sos(enc['W'], enc['w_B'], enc['w_minus']) - e_gap


def box_toffolis(name):
    """Toffolis per walk step at 15 bits, per box of the circuit, each box with its adjoint."""
    walk = DFTHCWalk.from_factors(**factors(name), b_rot=15, b_k1=15, b_k2=15)
    be = walk.block_encoding
    boxes = {'PREP': toffolis(be.outer) + toffolis(be.outer.adjoint()),
             'PREPB': 2 * (toffolis(be.inner) + toffolis(be.inner.adjoint())),
             'SEL': toffolis(be.select) + toffolis(be.select.adjoint()),
             'REF': toffolis(be.ref_t2) + toffolis(walk.ref_ab)}
    assert sum(boxes.values()) == toffolis(walk)
    return boxes


# --- the circuit, docs/dfthc-walk.tex with each box tinted ---

def circuit_png(theme, build):
    t = THEMES[theme]
    macros = (DOCS / 'circuit-macros.tex').read_text()
    macros = macros.replace('rounded corners=6pt}', 'rounded corners=6pt,fill=bg}')
    macros = macros.replace(r'\scriptsize #1', r'\scriptsize\color{ink2}#1')
    (build / 'circuit-macros.tex').write_text(macros)
    tex = (DOCS / 'dfthc-walk.tex').read_text()
    for old, new in [('fill=black!8', 'fill=PREP'), ('fill=black!9', 'fill=SEL'),
                     ('fill=black!3}]{\\mathrm{PREP}', 'fill=PREPB}]{\\mathrm{PREP}'),
                     ('fill=black!3}]{\\mathrm{SGN}', 'fill=bg}]{\\mathrm{SGN}'),
                     ('fill=red!10', 'fill=REF')]:
        tex = tex.replace(old, new)
    assert 'black!' not in tex and 'red!' not in tex
    colors = [rf'\definecolor{{{k}}}{{HTML}}{{{t[k][1:]}}}' for k in ('bg', 'ink', 'ink2')]
    for (k, _), hue in zip(BOXES, t['hues']):
        colors += [rf'\definecolor{{{k}hue}}{{HTML}}{{{hue[1:]}}}',
                   rf'\colorlet{{{k}}}{{{k}hue!{t["tint"]}!bg}}']
    tex = tex.replace(r'\pagecolor{white}', '\n'.join(colors + [r'\color{ink}']))
    (build / f'walk-{theme}.tex').write_text(tex)
    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', f'walk-{theme}.tex'],
                   cwd=build, check=True, capture_output=True)
    subprocess.run(['pdftocairo', '-png', '-r', '400', '-transp', '-singlefile',
                    f'walk-{theme}.pdf', f'walk-{theme}'], cwd=build, check=True)
    return build / f'walk-{theme}.png'


# --- the figure ---

def draw(theme, errors, boxes, circuit):
    t = THEMES[theme]
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True, 'font.size': 11, 'text.color': t['ink'],
        'axes.facecolor': 'none', 'figure.facecolor': 'none', 'axes.edgecolor': t['ink2'],
        'axes.labelcolor': t['ink2'], 'xtick.color': t['ink2'], 'ytick.color': t['ink2'],
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig = plt.figure(figsize=(11, 6.1), layout='constrained')
    top, bottom = fig.subfigures(2, 1, height_ratios=(0.82, 1.35))
    ax = top.subplots()
    ax.imshow(mpimg.imread(circuit))
    ax.axis('off')
    left, right = bottom.subplots(1, 2, width_ratios=(1, 1.15))

    # the energy error against the bits; the molecules in ink, so that colour means a box
    for name, color, marker in zip(SWEEP, (t['ink'], t['line2']), 'os'):
        err = np.abs(errors[name])
        left.plot(BITS, err, '-', marker=marker, color=color, lw=2, ms=7, mec=t['bg'], mew=1.5)
        label, (n_el, n_orb) = SYSTEMS[name]
        left.text(BITS[0] - 0.4, err[0], f'{label} ({n_el}, {n_orb})', ha='right', va='center',
                  fontsize=10)
    left.axhline(1.6e-3, color=t['ink2'], lw=1, ls='--')
    left.text(BITS[-1], 1.6e-3 * 1.4, 'chemical accuracy', color=t['ink2'], fontsize=10, ha='right')
    left.set(yscale='log', xlim=(-2.2, 16.5), xticks=range(4, 17, 2),
             xlabel=r'bits of precision, $b_{\mathrm{rot}} = b_{k_1} = b_{k_2} = s_u$',
             ylabel=r'$|E_0 - E_{\mathrm{FCI}}|$ (Ha)')
    left.grid(color=t['grid'], lw=0.8)

    # the Toffolis per walk step, split into the boxes of the circuit
    y, start = np.arange(len(SYSTEMS)), np.zeros(len(SYSTEMS))
    for (key, label), hue in zip(BOXES, t['hues']):
        width = np.array([boxes[name][key] for name in SYSTEMS])
        right.barh(y, width, left=start, height=0.62, color=hue, edgecolor=t['bg'], lw=1.5,
                   label=label)
        start += width
    for yi, total in zip(y, start):
        right.text(total + 80, yi, f'{total:.0f}', va='center', fontsize=10)
    right.set_yticks(y, [f'{label} ({n_el}, {n_orb})' for label, (n_el, n_orb) in SYSTEMS.values()])
    right.tick_params(axis='y', length=0, labelcolor=t['ink'])
    right.spines['left'].set_visible(False)
    right.invert_yaxis()
    right.set(xlim=(0, 7300), xlabel=r'Toffolis per walk step, '
              r'$b_{\mathrm{rot}} = b_{k_1} = b_{k_2} = 15$, $s_u = 7$')
    right.grid(axis='x', color=t['grid'], lw=0.8)
    right.set_axisbelow(True)
    right.legend(frameon=False, loc='upper right', fontsize=10, labelcolor=t['ink'])

    fig.savefig(DOCS / f'dfthc-overview-{theme}.png', dpi=170, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    errors = {name: [energy_error(name, bits) for bits in BITS] for name in SWEEP}
    boxes = {name: box_toffolis(name) for name in SYSTEMS}
    with tempfile.TemporaryDirectory() as build:
        for theme in THEMES:
            draw(theme, errors, boxes, circuit_png(theme, pathlib.Path(build)))
