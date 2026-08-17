# -*- coding: utf-8 -*-
"""问题1.1:Nmin 理论下界计算(修正版)。

符号:点 i 距基地 d_i、坐标 p_i、巡检次数 k_i、总次数 M=Σk_i。
T_srv = 5M (min) 精确;飞行总时间 T_fly 的有效下界(无载客限制、N 台机全投入):

  LB1  T_fly ≥ 2τ·d_(1)                                  (最远点所在机的往返)
  LB2  T_fly ≥ 2τ·(Σd_i / n)                             (每机 L_r≥2·max≥2·均值 → 求和)
  LB3  T_fly ≥ τ·(d_a+d_b+|p_a-p_b|)                      (最远两点同机:min TSP{0,a,b};
                                                          不同机:2d_a+2d_b,而 |p_a-p_b|≤d_a+d_b
                                                          故 w_ab=d_a+d_b+|p_a-p_b| ≤ 2(d_a+d_b),取 w_ab 恒成立)
  LB4  T_fly ≥ τ·min_{a<b∈Top(N+1)} w_ab                 (N 台机、最远 N+1 点必有两点同机,
                                                          同机则其机长≥w_ab)
  LB5  T_fly ≥ 2τ·(d_(1) + 最小的 N-1 个 d_i)             (N 台机均非空时各机长≥2×其最远点)

LB(N) = T_srv + max(LB1..LB5)。N 可行 ⇒ LB(N) ≤ 540N,
故 N_LB = min{N : LB(N) ≤ 540N} 是 Nmin 的下界;再由构造(OR-Tools)验证可达。
另:makespan 下界 T_max ≥ 2τ·d_(1) + 5 (任何调度)。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from common import load_points, travel_min, HORIZON_MIN, SERVICE_MIN

for case in ['Case1', 'Case2', 'Case3', 'Case4']:
    p = load_points(case)
    xy, visits = p['xy'], p['visits']
    d = np.hypot(xy[:, 0], xy[:, 1])
    n = len(d)
    M = int(visits.sum())
    t_srv = SERVICE_MIN * M

    order = np.argsort(-d)                       # 距离降序索引
    ds = d[order]
    ps = xy[order]
    w = ds[0] + ds[1] + float(np.linalg.norm(ps[0] - ps[1]))     # 最远两点
    lb1 = 2 * ds[0]
    lb2 = 2 * d.sum() / n
    makespan_lb = 2 * travel_min(ds[0]) + SERVICE_MIN

    def lb(N):
        v = [lb1, lb2, w]
        if N >= 2:
            top = N + 1
            pairs = []
            for a in range(min(top, n)):
                for b in range(a + 1, min(top, n)):
                    pairs.append(ds[a] + ds[b] + float(np.linalg.norm(ps[a] - ps[b])))
            if pairs:
                v.append(min(pairs))
        if N >= 2 and n >= N:
            v.append(2 * (ds[0] + d[np.argsort(d)[:N-1]].sum()))
        return t_srv + travel_min(max(v))

    n_lb = next((N for N in range(1, 80) if lb(N) <= HORIZON_MIN * N), None)
    print(f'{case}: n={n} M={M} T_srv={t_srv:.0f}min  d_max={ds[0]:.1f}u  Σd/n={d.mean():.0f}u')
    print(f'   LB1={travel_min(lb1):.1f}  LB2={travel_min(lb2):.1f}  LB3(w_12)={travel_min(w):.1f} min')
    for N in range(1, 9):
        print(f'   N={N}: LB={lb(N):.1f} min  vs 540N={540*N:4d}  -> {"可行上界通过" if lb(N) <= 540*N else "不可能"}')
    print(f'   makespan下界 = {makespan_lb:.1f} min = {makespan_lb/60:.2f} h')
    print(f'   => N_LB(理论下界) = {n_lb}')
    print()
