# -*- coding: utf-8 -*-
"""问题1 第一层:MTZ 混合整数规划(min-max mTSP)建模与求解(scipy/HiGHS)。

用途:
1. 严格证明 N 不可行:min Tmax,若最优值(或对偶下界 mip_dual_bound)> 540 min,
   则严格证明不存在 Tmax≤540 的 N 机调度 —— 把第一层的经验证据升级为证明;
2. 交叉验证:若 MILP 在某 N 找到 ≤540 的可行解,则 Nmin 结论应相应修正。

模型(节点 0=基地,任务 1..M;每机 r 一条从基地出发返回基地的路线):
  决策: x^r_ij ∈{0,1}(机 r 是否走弧 i->j,i≠j);u^r_j(MTZ 序号,任务 j 在机 r 路线上);
        a_j(到达任务 j 的时刻,8:00=0);Tmax(makespan)
  约束:
    任务恰一次:      Σ_r Σ_{i≠j} x^r_ij = 1, Σ_r Σ_{j≠i} x^r_ij = 1
    每机流平衡:      Σ_i x^r_ij = Σ_i x^r_ji (j≥1)
    起止基地:        Σ_j x^r_0j = 1, Σ_i x^r_i0 = 1
    消子环(MTZ):    u^r_j ≤ u^r_i − 1 + BigU(1−x^r_ij)  (i,j≥1,i≠j;u 沿路线递减)
    时间递推:        a_j ≥ a_i + τ·d_ij + s·[i≠0] − BigT(1−x^r_ij)
    返航完工:        Tmax ≥ a_i + τ·d_i0 − BigT(1−x^r_i0)
    边界:            Tmax ≤ BigT, a_j ≤ Tmax
  目标: min Tmax。BigT=900 min(>9h 的可行域足够,且不影响 ≤540 的结论)。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import coo_matrix

from common import SERVICE_MIN, TAU_MIN, load_points
from ga_minmax import build_D_ext, route_time_full

BIG_T = 600.0          # 时间大 M(分钟)。GA 上界 561.6 min 之内,收紧可强化松弛
HORIZON = 540.0


def build_model(tasks, xy, N):
    """构造 min-max mTSP 的 MTZ MILP。返回 (c, integrality, bounds, A_ub, lb_ub)。"""
    M = len(tasks)
    D_ext, _ = build_D_ext(tasks, xy)          # 任务节点 0..M-1,基地节点 = M
    DEP = M
    BigU = float(M)
    T = list(range(M))                          # 任务

    # 变量编号
    nvars = 0
    x_id = {}                                   # (r, i, j) -> idx
    for r in range(N):
        for i in range(M + 1):
            for j in range(M + 1):
                if i != j:
                    x_id[(r, i, j)] = nvars
                    nvars += 1
    u_id = {}
    for r in range(N):
        for j in T:
            u_id[(r, j)] = nvars
            nvars += 1
    a_id = {}
    for j in T:
        a_id[j] = nvars
        nvars += 1
    tmax_id = nvars
    nvars += 1

    c = np.zeros(nvars)
    c[tmax_id] = 1.0
    integrality = np.zeros(nvars, dtype=int)
    for v in x_id.values():
        integrality[v] = 1
    lb = np.zeros(nvars)
    ub = np.full(nvars, BIG_T)
    for v in u_id.values():
        ub[v] = BigU
    for v in x_id.values():
        ub[v] = 1.0
    ub[tmax_id] = BIG_T

    rows, cols, vals = [], [], []
    lb_rows, ub_rows = [], []

    def add_row(cs, lo, hi):
        n = len(lb_rows)
        for idx, coef in cs:
            rows.append(n)
            cols.append(idx)
            vals.append(coef)
        lb_rows.append(lo)
        ub_rows.append(hi)

    def arc_coef(r, i, j, coef=1.0):
        return x_id[(r, i, j)], coef

    # 任务恰一次(入/出)
    for j in T:
        add_row([arc_coef(r, i, j) for r in range(N) for i in range(M + 1) if i != j],
                1.0, 1.0)
        add_row([arc_coef(r, j, i) for r in range(N) for i in range(M + 1) if i != j],
                1.0, 1.0)
    # 每机流平衡 + 起止基地
    for r in range(N):
        for j in T:
            add_row([(x_id[(r, i, j)], 1.0) for i in range(M + 1) if i != j] +
                    [(x_id[(r, j, i)], -1.0) for i in range(M + 1) if i != j], 0.0, 0.0)
        add_row([arc_coef(r, DEP, j) for j in T], 1.0, 1.0)
        add_row([arc_coef(r, i, DEP) for i in T], 1.0, 1.0)
    # MTZ 消子环: u_j - u_i + BigU x_ij <= BigU - 1
    for r in range(N):
        for i in T:
            for j in T:
                if i == j:
                    continue
                add_row([(u_id[(r, j)], 1.0), (u_id[(r, i)], -1.0),
                         (x_id[(r, i, j)], BigU)], -np.inf, BigU - 1.0)
    # 时间递推: a_i - a_j + BigT x_ij <= BigT - c_ij, c_ij = τ d_ij + s·[i≠基地]
    # 注:只对到达任务 j 的弧建立;回程弧(i→基地)由返航完工约束处理
    for r in range(N):
        for i in range(M + 1):
            for j in T:
                if i == j:
                    continue
                cij = TAU_MIN * D_ext[i, j] + (SERVICE_MIN if i != DEP else 0.0)
                cs = [(x_id[(r, i, j)], BIG_T)]
                if i != DEP:
                    cs.append((a_id[i], 1.0))
                cs.append((a_id[j], -1.0))
                add_row(cs, -np.inf, BIG_T - cij)
    # 返航完工: a_i - Tmax + BigT x_i0 <= BigT - τ d_i0 - s
    # (到达任务 i 后需完成 5 min 巡检再返航,故含 SERVICE_MIN)
    for r in range(N):
        for i in T:
            add_row([(x_id[(r, i, DEP)], BIG_T), (a_id[i], 1.0), (tmax_id, -1.0)],
                    -np.inf, BIG_T - TAU_MIN * D_ext[i, DEP] - SERVICE_MIN)
    # a_j ≤ Tmax
    for j in T:
        add_row([(a_id[j], 1.0), (tmax_id, -1.0)], -np.inf, 0.0)
    # makespan 有效下界割(严格成立):
    #   LB1: Tmax ≥ 2τ·d_max + 5(服务最远点的机至少往返一次并巡检)
    #   LB2: Tmax ≥ (5M + 2τ·Σd_i/n) / N(总工作量均摊)
    d_dep = D_ext[DEP, T]
    add_row([(tmax_id, -1.0)], -np.inf, -(2 * TAU_MIN * d_dep.max() + SERVICE_MIN))
    add_row([(tmax_id, -1.0)], -np.inf,
            -((SERVICE_MIN * M + 2 * TAU_MIN * d_dep.sum() / M) / N))

    A = coo_matrix((vals, (rows, cols)), shape=(len(lb_rows), nvars)).tocsr()
    maps = dict(x=x_id, u=u_id, a=a_id, tmax=tmax_id)
    return c, integrality, lb, ub, A, np.array(lb_rows), np.array(ub_rows), maps, D_ext


def extract_routes(x, N, M, D_ext):
    """由 x 向量还原每机路线(用于可行性自检)。"""
    routes = []
    for r in range(N):
        route = []
        cur = M
        for _ in range(M + 1):
            nxt = None
            for j in range(M + 1):
                if j != cur and x.get((r, cur, j), 0) > 0.5:
                    nxt = j
                    break
            if nxt is None or nxt == M:
                break
            route.append(nxt)
            cur = nxt
        routes.append(route)
    return routes


def solve(case, N, time_limit=180.0, tol=1e-9, disp=False):
    p = load_points(case)
    tasks = [(i, k + 1) for i in range(len(p['ids'])) for k in range(int(p['visits'][i]))]
    M = len(tasks)
    t0 = time.time()
    c, integrality, lb, ub, A, lb_r, ub_r, maps, D_ext = build_model(tasks, p['xy'], N)
    x_id = maps['x']
    print('[%s N=%d] 建模: %d 变量(%d 二元), %d 约束, %.1f s'
          % (case, N, len(c), len(x_id), A.shape[0], time.time() - t0), flush=True)
    cons = LinearConstraint(A, lb_r, ub_r)
    res = milp(c=c, integrality=integrality, bounds=(lb, ub), constraints=cons,
               options={'time_limit': time_limit, 'disp': disp, 'mip_rel_gap': 0.0})
    wall = time.time() - t0
    fun = res.fun if res.fun is not None else float('nan')
    bound = getattr(res, 'mip_dual_bound', None)
    gap = getattr(res, 'mip_gap', None)
    out = dict(case=case, N=N, M=M, wall=wall, status=res.status, message=res.message,
               obj_min=fun, obj_h=fun / 60 if np.isfinite(fun) else None,
               dual_bound=bound, bound_h=bound / 60 if bound is not None else None,
               mip_gap=gap, nodes=getattr(res, 'mip_node_count', None))
    print('  状态: %s | 目标 min Tmax = %s min (%s h) | 对偶界 = %s min (%s h) | '
          'gap=%s | 节点=%s | wall=%.1f s'
          % (res.message, ('%.1f' % fun) if np.isfinite(fun) else 'n/a',
             ('%.2f' % (fun / 60)) if np.isfinite(fun) else 'n/a',
             ('%.1f' % bound) if bound is not None else 'n/a',
             ('%.2f' % (bound / 60)) if bound is not None else 'n/a',
             gap, getattr(res, 'mip_node_count', 'n/a'), wall), flush=True)
    if np.isfinite(fun) and res.x is not None:
        # 可行解还原与自检
        xv = {key: res.x[i] for key, i in x_id.items()}
        routes = extract_routes(xv, N, M, D_ext)
        tms = [route_time_full(r, D_ext, M) for r in routes]
        used = sorted(t for r in routes for t in r)
        ok = (len(used) == M and used == list(range(M)) and
              abs(max(tms) - fun) < 1.0)
        out.update(routes=routes, route_times=tms, self_check=ok)
        print('  可行解自检: 覆盖=%s, max 路线时间=%.1f min(与目标一致性容差 1 min)'
              % (ok, max(tms)), flush=True)
    if bound is not None and bound > HORIZON + 1e-6:
        print('  >> 对偶下界 > 540 min:严格证明 N=%d 不可行' % N, flush=True)
    elif fun is not None and np.isfinite(fun) and res.status == 0 and fun > HORIZON + 1e-6:
        print('  >> 已证最优且最优值 %.1f > 540 min:严格证明 N=%d 不可行' % (fun, N), flush=True)
    return out


def verify_solution(case, N, routes, tol=1e-6):
    """把已知可行调度代入 MILP 模型,检查全部约束(在真实规模上验证模型)。"""
    p = load_points(case)
    tasks = [(i, k + 1) for i in range(len(p['ids'])) for k in range(int(p['visits'][i]))]
    M = len(tasks)
    c, integrality, lb, ub, A, lb_r, ub_r, maps, D_ext = build_model(tasks, p['xy'], N)
    x_id, u_id, a_id, tmax_id = maps['x'], maps['u'], maps['a'], maps['tmax']
    nvars = len(c)
    v = np.zeros(nvars)
    tmax_val = 0.0
    for r, route in enumerate(routes):
        if not route:
            continue
        seq = [M] + route + [M]
        for a, b in zip(seq[:-1], seq[1:]):
            v[x_id[(r, a, b)]] = 1.0
        # 到达时刻(与调度一致)
        t = 0.0
        prev = M
        for pos, x in enumerate(route):
            v[u_id[(r, x)]] = len(route) - pos        # MTZ 采用沿路线递减的约定
            t += TAU_MIN * D_ext[prev, x]
            v[a_id[x]] = t
            t += SERVICE_MIN
            prev = x
        t += TAU_MIN * D_ext[prev, M]
        tmax_val = max(tmax_val, t)
    v[tmax_id] = tmax_val
    viol = 0
    Av = A @ v
    for i in range(A.shape[0]):
        if Av[i] > ub_r[i] + tol or Av[i] < lb_r[i] - tol:
            viol += 1
            if viol <= 5:
                print('  约束 %d 违反: %.6f 不在 [%.6f, %.6f]'
                      % (i, Av[i], lb_r[i], ub_r[i]))
    bound_ok = np.all(v >= lb - tol) and np.all(v <= ub + tol)
    print('[%s N=%d] 已知调度代入模型: %d 约束中违反 %d 条, 变量界%s, Tmax=%.2f min'
          % (case, N, A.shape[0], viol, 'OK' if bound_ok else '违反', tmax_val))
    return viol == 0 and bound_ok


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('N', type=int)
    ap.add_argument('--tl', type=float, default=180.0)
    ap.add_argument('--disp', action='store_true')
    a = ap.parse_args()
    solve(a.case, a.N, time_limit=a.tl, disp=a.disp)
