# -*- coding: utf-8 -*-
"""问题2:保证 makespan 尽量短的前提下最小化各机工作时长差 δ=|Tmax−Tmin|。

方法(分层/ε-约束):
- 第一阶段用问题1 的最优 makespan T*。
- 第二阶段网格搜索各机工作时长下界 L:
    对每个 L(分钟),加约束 CumulVar(End_v) ≥ 60L(所有机),min span 求解;
    得解后计算实际 δ = span − min_end(秒)。
  取 Pareto 最优:先按 span ≤ T*·(1+ε)(ε=0,1%,2%)分层,层内取 δ 最小。
  同时报告问题1 解的 δ 作为对照基线。

输出:output/p2_{case}.json;由 p2_export.py 写入 result2.xlsx。
"""
import sys, io, os, json
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common import load_points, A_DIR

CASE = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
TL = float(os.environ.get('P2_TL_SCALE', '1'))
SERVICE_S = 300
HORIZON_S = 9 * 3600
TAU_S = 0.1 / 55.0 * 3600.0

p = load_points(CASE)
xy, visits = p['xy'], p['visits']
tasks = [(i, k) for i in range(len(xy)) for k in range(visits[i])]
M = len(tasks)
coords = np.vstack([np.zeros((1, 2)), xy[[t[0] for t in tasks]]])


def build(N, L_sec):
    manager = pywrapcp.RoutingIndexManager(M + 1, N, 0)
    routing = pywrapcp.RoutingModel(manager)
    node2idx = {n: manager.NodeToIndex(n) for n in range(M + 1)}
    idx2node = {i: n for n, i in node2idx.items()}
    T = [[0] * (M + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(M + 1):
            T[i][j] = int(round(float(np.hypot(*(coords[i] - coords[j]))) * TAU_S)) \
                + (SERVICE_S if j > 0 else 0)

    def cb(fi, tj):
        return T[idx2node.get(fi, 0)][idx2node.get(tj, 0)]

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    routing.AddDimension(transit, 0, 20 * HORIZON_S, True, 'Time')
    tdim = routing.GetDimensionOrDie('Time')
    tdim.SetGlobalSpanCostCoefficient(100)
    if L_sec > 0:                                   # 每机工作时长下界
        for v in range(N):
            routing.solver().Add(tdim.CumulVar(manager.GetEndIndex(v)) >= L_sec)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.log_search = False
    return manager, routing, tdim, sp, idx2node


def solve(N, L_sec, tlimit, seed):
    manager, routing, tdim, sp, idx2node = build(N, L_sec)
    sp.time_limit.seconds = tlimit
    sp.sat_parameters.random_seed = seed
    sol = routing.SolveWithParameters(sp)
    if sol is None:
        return None
    routes = []
    for v in range(N):
        seq, idx = [], manager.GetStartIndex(v)
        while not routing.IsEnd(idx):
            node = idx2node.get(idx, 0)
            if node > 0:
                seq.append(dict(node=node, cumul=sol.Value(tdim.CumulVar(idx))))
            idx = sol.Value(routing.NextVar(idx))
        end_s = sol.Value(tdim.CumulVar(manager.GetEndIndex(v)))
        nodes = [0] + [s['node'] for s in seq] + [0]
        legs_u = [round(float(np.hypot(*(coords[a] - coords[b]))), 3)
                  for a, b in zip(nodes[:-1], nodes[1:])]
        arr = [round((s['cumul'] - SERVICE_S) / 60, 2) for s in seq]
        dep = [round(s['cumul'] / 60, 2) for s in seq]
        routes.append(dict(veh=v + 1,
                           seq=[dict(point_id=int(tasks[s['node'] - 1][0]) + 1,
                                     visit_no=int(tasks[s['node'] - 1][1]) + 1) for s in seq],
                           arr_min=arr, dep_min=dep, legs_u=legs_u,
                           travel_units=round(sum(legs_u), 2), end_s=end_s,
                           busy_h=round(end_s / 3600, 4), n_tasks=len(seq)))
    ends = [r['end_s'] for r in routes]
    span_s = max(ends)
    return dict(span_s=span_s, span_h=round(span_s / 3600, 4),
                delta_h=round((span_s - min(ends)) / 3600, 4), routes=routes)


# 网格搜索 L:粗 30min + 局部细化
def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    with open(os.path.join(A_DIR, 'output', f'p1_{CASE}.json'), encoding='utf-8') as f:
        p1 = json.load(f)
    N = p1['summary']['Nmin']
    TSTAR_S = int(round(p1['summary']['tmax_h'] * 3600))
    baseline_delta_h = round(p1['summary']['tmax_h'] - p1['summary']['tmin_h'], 4)
    print(f'{CASE}: N={N}, T*={TSTAR_S/3600:.3f}h, 基线δ(P1解)={baseline_delta_h:.3f}h', flush=True)

    tlimit = max(5, int(30 * TL))
    L_grid = list(range(0, TSTAR_S, 30 * 60))
    results = []
    # 候选 0:问题1 的原解本身(合法的问题2 调度)
    p1r = p1['best']['routes']
    ends1 = [r['end_s'] for r in p1r]
    results.append(dict(L_h=None, span_h=round(max(ends1) / 3600, 3),
                        span_s=max(ends1),
                        delta_h=round((max(ends1) - min(ends1)) / 3600, 4), sol=None,
                        p1_routes=p1r))
    for L_sec in L_grid:
        best = None
        for seed in (1, 2):
            r = solve(N, L_sec, tlimit, seed)
            if r is not None and (best is None or r['span_s'] < best['span_s']):
                best = r
        if best is not None:
            results.append(dict(L_h=round(L_sec / 3600, 3), span_h=best['span_h'],
                                span_s=best['span_s'], delta_h=best['delta_h'], sol=best))
            print(f'  L={L_sec/3600:5.2f}h -> span={best["span_h"]:.3f}h  δ={best["delta_h"]:.3f}h', flush=True)
        else:
            print(f'  L={L_sec/3600:5.2f}h -> 无可行解', flush=True)

    # Pareto 分层:ε=0 / 1% / 2%
    out = dict(case=CASE, N=N, Tstar_h=round(TSTAR_S / 3600, 4),
               baseline_delta_h=baseline_delta_h, grid=[])
    for eps in (0.0, 0.01, 0.02):
        cap = TSTAR_S * (1 + eps)
        cand = [r for r in results if r['span_s'] <= cap + 1]
        if cand:
            best = min(cand, key=lambda r: (r['span_s'], r['delta_h']))
            routes = best['sol']['routes'] if best['sol'] is not None else best['p1_routes']
            out['grid'].append(dict(eps=eps, L_h=best['L_h'], span_h=best['span_h'],
                                    delta_h=best['delta_h'], routes=routes))
            print(f'ε={eps:.0%}: 取 L={best["L_h"]} -> Tmax={best["span_h"]:.3f}h  δ={best["delta_h"]:.3f}h', flush=True)

    os.makedirs(os.path.join(A_DIR, 'output'), exist_ok=True)
    path = os.path.join(A_DIR, 'output', f'p2_{CASE}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'[P2 {CASE}] -> {path}')


if __name__ == '__main__':
    main()
