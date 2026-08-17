# -*- coding: utf-8 -*-
"""可视化:路径图 + 甘特图(p1/p2/p3 通用)。"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import load_points, A_DIR

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

PROBLEM = sys.argv[1] if len(sys.argv) > 1 else 'p1'   # p1|p2|p3
CASE = sys.argv[2] if len(sys.argv) > 2 else 'Case1'
FIG_DIR = os.path.join(A_DIR, 'output', 'figs')
os.makedirs(FIG_DIR, exist_ok=True)

LEVEL_COLOR = {'I': '#d62728', 'II': '#ff7f0e', 'III': '#1f77b4'}
PALETTE = ['#2ca02c', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22',
           '#d62728', '#ff7f0e', '#1f77b4', '#7f7f7f']

p = load_points(CASE)
xy, level = p['xy'], p['level']

with open(os.path.join(A_DIR, 'output', f'{PROBLEM}_{CASE}.json'), encoding='utf-8') as f:
    data = json.load(f)
if PROBLEM == 'p1':
    routes, span_h = data['best']['routes'], data['best']['span_h']
elif PROBLEM == 'p2':
    routes, span_h = data['grid'][0]['routes'], data['grid'][0]['span_h']
else:
    routes, span_h = data['routes'], data['summary']['tmax_h']

# ---------- 路径图 ----------
fig, ax = plt.subplots(figsize=(11, 9))
for lv, c in LEVEL_COLOR.items():
    m = level == lv
    ax.scatter(xy[m, 0], xy[m, 1], s=30, color=c, alpha=0.85, edgecolors='white',
               linewidths=0.4, zorder=3, label=f'{lv} 级')
ax.scatter([0], [0], marker='*', s=420, color='black', zorder=6, label='基地 (0,0)')
for r in routes:
    if r['n_tasks'] == 0:
        continue
    if r.get('legs_wp'):
        pts = [r['legs_wp'][0][0]] + [q for L in r['legs_wp'] for q in L[1:]]
        pts = np.array(pts)
    else:
        pts = np.vstack([[0, 0]] + [xy[s['point_id'] - 1] for s in r['seq']] + [[0, 0]])
    c = PALETTE[(r['veh'] - 1) % len(PALETTE)]
    ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=1.6, alpha=0.9, zorder=4,
            label=f"无人机{r['veh']} ({r['busy_h']:.2f}h)")
    mid = pts[len(pts) // 2]
    ax.annotate(f"U{r['veh']}", mid, color=c, fontsize=11, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=c, alpha=0.85))
ax.set_aspect('equal'); ax.grid(True, ls='--', alpha=0.35)
ax.set_title(f'{CASE} · 问题{PROBLEM[-1]} 多机巡检路径(makespan={span_h:.2f} h)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('X 坐标(单位,1 单位=100 m)'); ax.set_ylabel('Y 坐标(单位)')
ax.legend(loc='best', fontsize=8.5, ncol=2, framealpha=0.92)
fig.tight_layout()
p1 = os.path.join(FIG_DIR, f'{PROBLEM}_{CASE}_路径图.png')
fig.savefig(p1, dpi=180); plt.close(fig)

# ---------- 甘特图 ----------
fig, ax = plt.subplots(figsize=(11, 3.2 + 0.55 * len(routes)))
active = [r for r in routes if r['n_tasks'] > 0]
for i, r in enumerate(active):
    ax.barh(i, r['end_s'] / 3600, left=0, height=0.6, color=PALETTE[(r['veh'] - 1) % len(PALETTE)],
            alpha=0.45, edgecolor='none')
    for k, s in enumerate(r['seq']):
        t0 = r['arr_min'][k] / 60
        ax.barh(i, 5 / 60, left=t0, height=0.6,
                color='#d62728' if p['level'][s['point_id'] - 1] == 'I' else
                      ('#ff7f0e' if p['level'][s['point_id'] - 1] == 'II' else '#1f77b4'))
ax.set_yticks(range(len(active)))
ax.set_yticklabels([f"无人机{r['veh']}  ({r['busy_h']:.2f}h)" for r in active])
ax.set_xlabel('时间(小时,8:00 起)')
ax.set_title(f'{CASE} · 甘特图(蓝=III 橙=II 红=I 级巡检,条长=5min;浅色=总时长)',
             fontsize=12, fontweight='bold')
ax.axvline(9, color='gray', ls=':', lw=1.2)
ax.text(9, len(active) - 0.25, ' 9h 上限', fontsize=9, color='gray')
ax.grid(True, axis='x', ls='--', alpha=0.35)
fig.tight_layout()
p2 = os.path.join(FIG_DIR, f'{PROBLEM}_{CASE}_甘特图.png')
fig.savefig(p2, dpi=180); plt.close(fig)
print('saved:', p1, '|', p2)
