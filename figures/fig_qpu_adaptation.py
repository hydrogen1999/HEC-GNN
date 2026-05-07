#!/usr/bin/env python3
"""Figure: QPU Adaptation Results (600 instances, D-Wave Advantage).
Self-contained with inline data. Produces 4-panel figure."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'axes.labelsize': 10, 'legend.fontsize': 8})

# ═══════════════════════════════════════════════════════════════════
# INLINE DATA (from qpu_adaptation_results_full.json)
# ═══════════════════════════════════════════════════════════════════

GRID = [0.02, 0.026745, 0.035764, 0.047825, 0.063953, 0.08552, 0.11436,
        0.152926, 0.204498, 0.273462, 0.365682, 0.489002, 0.653911,
        0.874431, 1.169319, 1.563654, 2.09097, 2.796117, 3.739062, 5.0]

# Per-size δE≤5% (8 sizes × 5 methods)
SIZES = [8, 10, 12, 15, 20, 25, 30, 40]
DE5 = {
    'HEC-GNN':     [91.7, 92.2, 96.1, 91.8, 89.1, 87.1, 88.2, 83.5],
    'FlatGNN':     [97.2, 89.6, 97.4, 93.2, 85.9, 87.1, 88.2, 83.5],
    'UTC':         [97.2, 92.2, 90.8, 94.5, 92.4, 83.5, 89.5, 83.5],
    'Scaled(2.0)': [100., 97.4, 97.4, 93.2, 84.8, 87.1, 88.2, 83.5],
}
GAP = {
    'HEC-GNN':     [1.50, 1.71, 0.80, 1.62, 2.84, 2.33, 2.10, 2.14],
    'FlatGNN':     [1.35, 2.27, 0.81, 1.63, 3.13, 2.34, 2.10, 2.14],
    'UTC':         [1.30, 2.07, 2.33, 2.76, 3.48, 3.01, 2.20, 2.14],
    'Scaled(2.0)': [0.54, 0.75, 0.65, 1.50, 3.22, 2.34, 2.10, 2.14],
}

# Example curves: QPU (real) vs HEC-GNN (predicted)
EX_QPU = {
    10: [-20.44,-20.78,-20.79,-20.58,-20.63,-20.79,-20.66,-21.18,-21.31,
         -21.76,-21.99,-22.26,-22.32,-22.39,-22.49,-22.42,-22.36,-22.39,-22.17,-21.82],
    15: [-19.42,-18.99,-18.77,-19.2,-19.43,-19.9,-19.79,-20.23,-20.63,
         -20.97,-21.29,-22.09,-22.61,-24.27,-29.05,-31.5,-31.76,-31.8,-31.54,-30.76],
    25: [-25.15,-25.07,-25.36,-25.38,-24.91,-25.52,-25.54,-24.65,-26.23,
         -26.97,-29.4,-28.9,-32.29,-35.87,-41.92,-48.33,-50.65,-50.94,-50.26,-49.31],
}
EX_HEC = {
    10: [-7.87,-7.85,-7.92,-7.96,-7.97,-8.03,-8.11,-8.21,-8.34,
         -8.59,-8.79,-9.07,-9.32,-9.6,-9.76,-9.78,-9.72,-9.49,-9.15,-8.75],
    15: [-9.07,-9.08,-9.18,-9.27,-9.34,-9.48,-9.65,-9.9,-10.21,
         -10.77,-11.25,-11.95,-12.84,-13.72,-14.49,-14.94,-15.23,-14.96,-14.27,-13.28],
    25: [-14.26,-14.29,-14.47,-14.61,-14.72,-14.97,-15.27,-15.68,-16.21,
         -17.15,-17.97,-19.17,-20.73,-22.27,-23.65,-24.49,-25.05,-24.63,-23.35,-21.4],
}
EX_RSTAR = {10: 1.1693, 15: 2.7961, 25: 2.7961}

# CDF quantiles (50 points each, δE values at quantile 0%, 2%, 4%, ..., 98%)
CDF_Q = np.linspace(0, 1, 50, endpoint=False)
CDF = {
    'HEC-GNN': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                0.00046,0.00097,0.00151,0.00219,0.00272,0.00315,0.00398,0.00452,0.00509,
                0.00564,0.00645,0.00802,0.00855,0.00927,0.01048,0.0115,0.01212,0.014,
                0.01506,0.01593,0.01765,0.02038,0.02439,0.02769,0.03097,0.03656,0.04083,
                0.04519,0.05185,0.05856,0.0753,0.10065,0.13281],
    'FlatGNN': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3e-05,
                0.0004,0.00103,0.00142,0.00203,0.00285,0.00335,0.00389,0.00428,0.00467,
                0.00547,0.00614,0.00671,0.00802,0.00878,0.00957,0.01142,0.01199,0.01308,
                0.01466,0.01581,0.01757,0.01954,0.02276,0.02577,0.02853,0.03141,0.03656,
                0.04143,0.04712,0.05287,0.06228,0.08146,0.10745,0.14969],
    'UTC':     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1e-05,0.00162,0.00329,
                0.00494,0.00655,0.00734,0.00814,0.00945,0.01081,0.01164,0.01308,0.0147,
                0.01593,0.01705,0.01829,0.01917,0.02107,0.02201,0.02349,0.02494,0.02601,
                0.02756,0.02844,0.02969,0.03091,0.03289,0.03472,0.03656,0.03852,0.04028,
                0.04226,0.04468,0.04694,0.05274,0.0567,0.06312,0.08021,0.1243],
    'Scaled(2.0)': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                    0.0003,0.00074,0.00106,0.00164,0.00239,0.00276,0.00321,0.00373,0.00435,
                    0.00494,0.00547,0.00592,0.0065,0.00756,0.00838,0.009,0.01083,0.01175,
                    0.01267,0.01405,0.01565,0.01671,0.01941,0.02222,0.02654,0.02936,0.03587,
                    0.04197,0.04819,0.05775,0.0753,0.09456,0.12483],
}

# Summary table
SUMMARY = {
    'Oracle':          {'de5': 100.0, 'de2': 100.0, 'gap': 0.00},
    'HEC-GNN (FT)':   {'de5': 90.3,  'de2': 73.8,  'gap': 1.91, 'de5_std': 2.0, 'gap_std': 0.57},
    'HEC-GNN (SA)':   {'de5': 87.8,  'de2': 73.5,  'gap': 2.16},
    'FlatGNN (SA)':   {'de5': 88.1,  'de2': 73.1,  'gap': 2.19},
    'UTC':            {'de5': 89.8,  'de2': 55.5,  'gap': 2.52},
    'Scaled(2.0)':    {'de5': 90.5,  'de2': 78.5,  'gap': 1.79},
}

# r* histogram
RSTAR_COUNTS = [259,10,5,10,30,0,44,0,66,0,0,133,0,0,41,0,0,0,0,2]
RSTAR_EDGES = [0.02,0.269,0.518,0.767,1.016,1.265,1.514,1.763,2.012,2.261,
               2.51,2.759,3.008,3.257,3.506,3.755,4.004,4.253,4.502,4.751,5.0]

# ═══════════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════════
C = {
    'HEC-GNN': '#2166ac',
    'FlatGNN': '#67a9cf',
    'UTC': '#ef8a62',
    'Scaled(2.0)': '#999999',
    'HEC-GNN (FT)': '#b2182b',
    'QPU': '#1a1a1a',
}

# ═══════════════════════════════════════════════════════════════════
# FIGURE: 2×2 panels
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.0, 5.5))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35,
                       left=0.09, right=0.97, top=0.95, bottom=0.08)

# ─── (a) Example QPU vs predicted curves ───
ax1 = fig.add_subplot(gs[0, 0])
for n, ls, alpha in [(10, '-', 0.7), (15, '--', 0.8), (25, '-.', 0.9)]:
    qpu = np.array(EX_QPU[n])
    hec = np.array(EX_HEC[n])
    # Normalize both to [0,1] for shape comparison
    qpu_n = (qpu - qpu.min()) / max(qpu.max() - qpu.min(), 1e-8)
    hec_n = (hec - hec.min()) / max(hec.max() - hec.min(), 1e-8)
    ax1.plot(GRID, qpu_n, ls, color=C['QPU'], alpha=alpha, lw=1.5, label=f'QPU n={n}')
    ax1.plot(GRID, hec_n, ls, color=C['HEC-GNN'], alpha=alpha, lw=1.2)
ax1.set_xscale('log')
ax1.set_xlabel('Chain strength ratio $r$')
ax1.set_ylabel('Normalized energy')
ax1.set_title('(a) QPU curves vs HEC-GNN predicted', fontsize=9, fontweight='bold')
# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=C['QPU'], lw=1.5, label='QPU (real)'),
    Line2D([0], [0], color=C['HEC-GNN'], lw=1.2, label='HEC-GNN (pred)'),
]
ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.8)

# ─── (b) CDF of δE ───
ax2 = fig.add_subplot(gs[0, 1])
for method, label in [('HEC-GNN', 'HEC-GNN'), ('FlatGNN', 'FlatGNN'),
                       ('UTC', 'UTC'), ('Scaled(2.0)', 'Scaled(2.0)')]:
    vals = np.array(CDF[method]) * 100  # to percent
    ax2.plot(vals, CDF_Q * 100, color=C.get(method, '#333'), lw=1.5, label=label)
ax2.axvline(5, color='gray', ls=':', lw=0.8, alpha=0.5)
ax2.axvline(2, color='gray', ls=':', lw=0.8, alpha=0.5)
ax2.set_xlabel('Energy regret $\\delta_E$ (%)')
ax2.set_ylabel('Cumulative fraction (%)')
ax2.set_xlim(0, 15)
ax2.set_title('(b) CDF of energy regret on QPU', fontsize=9, fontweight='bold')
ax2.legend(loc='lower right', framealpha=0.8)
ax2.text(5.3, 5, '$\\tau$=5%', fontsize=7, color='gray')
ax2.text(2.3, 5, '$\\tau$=2%', fontsize=7, color='gray')

# ─── (c) Per-size δE≤5% ───
ax3 = fig.add_subplot(gs[1, 0])
x = np.arange(len(SIZES))
w = 0.2
for i, (method, offset) in enumerate([('HEC-GNN', -1.5), ('FlatGNN', -0.5),
                                        ('UTC', 0.5), ('Scaled(2.0)', 1.5)]):
    ax3.bar(x + offset * w, DE5[method], w, color=C.get(method, '#333'),
            label=method, alpha=0.85, edgecolor='white', linewidth=0.3)
ax3.set_xticks(x)
ax3.set_xticklabels(SIZES)
ax3.set_xlabel('Problem size $n$')
ax3.set_ylabel('$\\delta_E \\leq 5\\%$ (%)')
ax3.set_ylim(70, 102)
ax3.set_title('(c) QPU compliance by problem size', fontsize=9, fontweight='bold')
ax3.legend(loc='lower left', ncol=2, framealpha=0.8)

# ─── (d) Summary: SA→QPU transfer + fine-tuning ───
ax4 = fig.add_subplot(gs[1, 1])
methods = ['Scaled(2.0)', 'HEC-GNN (FT)', 'UTC', 'FlatGNN (SA)', 'HEC-GNN (SA)']
de5_vals = [SUMMARY[m]['de5'] for m in methods]
gap_vals = [SUMMARY[m]['gap'] for m in methods]
colors_bar = [C.get(m.split(' ')[0], '#999') for m in methods]
colors_bar[1] = C['HEC-GNN (FT)']  # fine-tuned in red

y = np.arange(len(methods))
bars = ax4.barh(y, de5_vals, color=colors_bar, alpha=0.85, edgecolor='white', height=0.6)
ax4.set_yticks(y)
ax4.set_yticklabels(methods, fontsize=8)
ax4.set_xlabel('$\\delta_E \\leq 5\\%$ (%)')
ax4.set_xlim(82, 95)
ax4.set_title('(d) SA→QPU transfer & fine-tuning', fontsize=9, fontweight='bold')
# Add gap% annotation
for i, (d5, gap) in enumerate(zip(de5_vals, gap_vals)):
    ax4.text(d5 + 0.3, i, f'Gap {gap:.1f}%', va='center', fontsize=7, color='#333')

plt.savefig('figures/fig_qpu_adaptation.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_qpu_adaptation.png', dpi=200, bbox_inches='tight')
print("Saved figures/fig_qpu_adaptation.pdf & .png")
