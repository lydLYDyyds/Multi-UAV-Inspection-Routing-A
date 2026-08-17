# -*- coding: utf-8 -*-
"""问题3 背景图:禁飞区空间分布(圆) + 生效时间段(时间条)。

每算例输出一张图 output/figs/p3_zones_{case}.png:
- 左面板:巡检点(按等级着色)+ 基地 + 禁飞圆(半透明填充,Z_i 标签与半径);
- 右面板:时间条 8:00–17:00,每个禁飞区一行,条 = 生效区间;
  Case4 Z8(17:00–17:00 零长度)画瞬时标记。
用法:python src/plot_zones.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from common import load_points, load_zones, A_DIR

plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

CASES = ['Case1', 'Case2', 'Case3', 'Case4']
FIG_DIR = os.path.join(A_DIR, 'output', 'figs')
os.makedirs(FIG_DIR, exist_ok=True)

LEVEL_COLOR = {'I': '#d62728', 'II': '#ff7f0e', 'III': '#1f77b4'}
ZONE_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
               '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990']


def hhmm(minutes):
    minutes = int(round(minutes))
    return f'{8 + minutes // 60:02d}:{minutes % 60:02d}'


def draw_case(case):
    p = load_points(case)
    zones = load_zones(case)
    zones_sorted = sorted(zones, key=lambda z: (z['t0'], z['t1']))
    zcol = {z['zone_id']: ZONE_COLORS[i % len(ZONE_COLORS)]
            for i, z in enumerate(zones_sorted)}

    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.06)

    # ---------- 左:空间分布 ----------
    ax = fig.add_subplot(gs[0])
    xy, level = p['xy'], p['level']
    for lv, c in LEVEL_COLOR.items():
        m = level == lv
        ax.scatter(xy[m, 0], xy[m, 1], s=16, color=c, alpha=0.75,
                   edgecolors='white', linewidths=0.3, zorder=2,
                   label=f'{lv} 级巡检点')
    ax.scatter([0], [0], marker='*', s=480, color='black', zorder=6, label='基地 (0,0)')
    for z in zones_sorted:
        c = zcol[z['zone_id']]
        ax.add_patch(Circle(z['c'], z['r'], facecolor=c, alpha=0.16,
                            edgecolor=c, lw=1.8, zorder=3))
        ax.annotate(z['zone_id'], z['c'], ha='center', va='center', zorder=5,
                    fontsize=11, fontweight='bold', color='#333333',
                    bbox=dict(boxstyle='circle,pad=0.28', fc='white',
                              ec=c, alpha=0.9))
        ax.annotate(f"r={z['r']:.0f}", (z['c'][0], z['c'][1] - z['r'] - 8),
                    ha='center', fontsize=8, color=c)
    xs = np.concatenate([xy[:, 0], [0.0],
                         np.array([z['c'][0] + z['r'] for z in zones]),
                         np.array([z['c'][0] - z['r'] for z in zones])])
    ys = np.concatenate([xy[:, 1], [0.0],
                         np.array([z['c'][1] + z['r'] for z in zones]),
                         np.array([z['c'][1] - z['r'] for z in zones])])
    pad = max(60, (xs.max() - xs.min()) * 0.04)
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect('equal')
    ax.grid(True, ls='--', alpha=0.35)
    ax.set_xlabel('X 坐标(单位,1 单位=100 m)')
    ax.set_ylabel('Y 坐标(单位)')
    ax.set_title(f'{case} · 禁飞区空间分布', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.92)

    # ---------- 右:生效时间段 ----------
    ax2 = fig.add_subplot(gs[1])
    n = len(zones_sorted)
    for i, z in enumerate(zones_sorted):
        y = n - 1 - i            # 最早生效的在最上方(甘特图惯例)
        c = zcol[z['zone_id']]
        if z['t1'] > z['t0']:
            ax2.barh(y, z['t1'] - z['t0'], left=z['t0'], height=0.5,
                     color=c, alpha=0.55, edgecolor=c, lw=1.4, zorder=3)
        else:  # 零长度窗口(如 Case4 Z8):瞬时标记
            ax2.plot([z['t0']], [y], marker='D', ms=9, color=c, zorder=4)
        ax2.text(544, y, f"{z['zone_id']}  {hhmm(z['t0'])}–{hhmm(z['t1'])}"
                 + ('(瞬时)' if z['t1'] <= z['t0'] else ''),
                 va='center', ha='left', fontsize=9.5, color='#222222')
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([zones_sorted[n - 1 - k]['zone_id'] for k in range(n)])
    ax2.set_ylim(-0.9, n - 0.1)
    ax2.set_xlim(-8, 730)
    ax2.axvline(0, color='gray', lw=1)
    ax2.axvline(540, color='gray', ls=':', lw=1.4)
    ax2.text(541, -0.6, '17:00 作业截止', fontsize=8.5, color='gray', va='center')
    ticks = range(0, 541, 60)
    ax2.set_xticks(list(ticks))
    ax2.set_xticklabels([hhmm(t) for t in ticks])
    ax2.set_xlabel('时刻(自 8:00 起)')
    ax2.set_title('禁飞区生效时间段', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='x', ls='--', alpha=0.35)

    fig.suptitle(f'{case} · 附件2 禁飞区(共 {n} 个)· 区域与生效时段总览',
                 fontsize=14, fontweight='bold')
    out = os.path.join(FIG_DIR, f'p3_zones_{case}.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('saved:', out)
    for z in zones_sorted:
        print(f"  {z['zone_id']}: c=({z['c'][0]:.1f},{z['c'][1]:.1f}) r={z['r']:.0f} "
              f"{hhmm(z['t0'])}–{hhmm(z['t1'])}")


if __name__ == '__main__':
    for c in CASES:
        draw_case(c)
