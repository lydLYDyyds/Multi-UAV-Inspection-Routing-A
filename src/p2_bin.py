# -*- coding: utf-8 -*-
"""问题2:问题一 GLS 机制 + 工作时长下界 L 二分搜索(ε-约束分层)。

可行域随 L 单调收缩,故对每个 ε 层在 L∈[0, T*(1+ε)] 上二分(容差 TOL 秒);
每个探针 L 用 OR-Tools 求解 min span(约束 End_v ≥ L,2 种子取优),探针结果
跨层缓存;层内取 span ≤ T*(1+ε) 的候选中 (span, δ) 词典序最优者。
输出 output/p2bin_{case}.json(与 p2_*.json 同结构,兼容导出与绘图脚本)。

用法:python src/p2_bin.py Case1 [TOL秒=120]
"""
import sys, io, os, json
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common import load_points, A_DIR

CASE = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
TOL_S = int(sys.argv[2]) if len(sys.argv) > 2 else 120
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
    if L_sec > 0:
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


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    out_path = os.path.join(A_DIR, 'output', f'p2bin_{CASE}.json')
    if os.path.exists(out_path):            # 断点续跑:已完成则跳过
        with open(out_path, encoding='utf-8') as f:
            prev = json.load(f)
        if len(prev.get('grid', [])) >= 3:
            print(f'[P2BIN {CASE}] 已存在完整结果({len(prev["grid"])} 层),跳过', flush=True)
            return
    with open(os.path.join(A_DIR, 'output', f'p1_{CASE}.json'), encoding='utf-8') as f:
        p1 = json.load(f)
    N = p1['summary']['Nmin']
    TSTAR_S = int(round(p1['summary']['tmax_h'] * 3600))
    baseline_delta_h = round(p1['summary']['tmax_h'] - p1['summary']['tmin_h'], 4)
    p1_routes = p1['best']['routes']
    tlimit = max(5, int(30 * TL))
    print(f'{CASE}: N={N}, T*={TSTAR_S/3600:.3f}h, 基线δ(P1解)={baseline_delta_h:.3f}h, '
          f'TOL={TOL_S}s', flush=True)

    cache = {}

    def probe(L):
        """oracle:P(L) = min span s.t. End_v ≥ L(秒)。结果缓存,2 种子取优。"""
        if L not in cache:
            best = None
            for seed in (1, 2):
                r = solve(N, L, tlimit, seed)
                if r is not None and (best is None or r['span_s'] < best['span_s']):
                    best = r
            cache[L] = best
            if best is not None:
                print(f'  probe L={L/60:6.1f}min -> span={best["span_h"]:.3f}h '
                      f'δ={best["delta_h"]:.3f}h', flush=True)
            else:
                print(f'  probe L={L/60:6.1f}min -> 无解', flush=True)
        return cache[L]

    def feasible(r, cap):
        return r is not None and r['span_s'] <= cap + 1

    cands = [dict(L_s=0, span_s=TSTAR_S, delta_h=baseline_delta_h,
                  routes=p1_routes, note='P1解(L=0 基线)')]

    probe(0)  # L=0 的 oracle 结果(=无下界约束的 min-span)也入缓存与候选
    for eps in (0.0, 0.01, 0.02):
        cap = TSTAR_S * (1 + eps)
        hi0 = int(cap) + TOL_S
        if feasible(probe(hi0), cap):          # 病理:全域可行
            Lstar = hi0
            print(f'  [ε={eps:.0%}] 全域可行,取 L={hi0/60:.1f}min', flush=True)
        else:
            lo = 0
            while hi0 - lo > TOL_S:            # 二分:lo 端恒可行,hi 端恒不可行
                L = (lo + hi0) // 2
                if feasible(probe(L), cap):
                    lo = L
                else:
                    hi0 = L
            Lstar = lo
            for k in (1, 2):                  # 噪声修复:上界方向再探 1~2 步
                Lc = Lstar + k * TOL_S
                if Lc > int(cap) + TOL_S:
                    break
                if feasible(probe(Lc), cap):
                    Lstar = Lc
                else:
                    break
        print(f'  [ε={eps:.0%}] 二分收敛 L*={Lstar/60:.1f}min', flush=True)

    # 层内选择:span ≤ cap 的全部探针 + P1 基线,按 (span, δ) 词典序取最优
    out = dict(case=CASE, N=N, Tstar_h=round(TSTAR_S / 3600, 4),
               baseline_delta_h=baseline_delta_h, method=f'binary-TOL{TOL_S}s',
               grid=[])
    for eps in (0.0, 0.01, 0.02):
        cap = TSTAR_S * (1 + eps)
        pool = cands + [dict(L_s=L, span_s=r['span_s'], delta_h=r['delta_h'],
                             routes=r['routes'], note='probe')
                        for L, r in cache.items() if r is not None]
        ok = [c for c in pool if c['span_s'] <= cap + 1]
        best = min(ok, key=lambda c: (c['span_s'], c['delta_h']))
        out['grid'].append(dict(eps=eps, L_h=None if best['note'] == 'P1解(L=0 基线)'
                                else round(best['L_s'] / 3600, 3),
                                span_h=round(best['span_s'] / 3600, 4),
                                delta_h=round(best['delta_h'], 4), routes=best['routes'],
                                note=best['note']))
        print(f'ε={eps:.0%}: L={out["grid"][-1]["L_h"]} -> Tmax='
              f'{out["grid"][-1]["span_h"]:.3f}h  δ={out["grid"][-1]["delta_h"]:.3f}h '
              f'({best["note"]})', flush=True)

    os.makedirs(os.path.join(A_DIR, 'output'), exist_ok=True)
    path = os.path.join(A_DIR, 'output', f'p2bin_{CASE}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'[P2BIN {CASE}] -> {path}')

    # 与旧网格结果对比
    old_path = os.path.join(A_DIR, 'output', f'p2_{CASE}.json')
    if os.path.exists(old_path):
        with open(old_path, encoding='utf-8') as f:
            old = json.load(f)
        for eps in (0.0, 0.01, 0.02):
            og = next((g for g in old['grid'] if g['eps'] == eps), None)
            ng = next((g for g in out['grid'] if g['eps'] == eps), None)
            if og and ng:
                print(f'对比 ε={eps:.0%}: 旧网格 span={og["span_h"]:.4f}h '
                      f'δ={og["delta_h"]:.4f}h | 新二分 span={ng["span_h"]:.4f}h '
                      f'δ={ng["delta_h"]:.4f}h', flush=True)


if __name__ == '__main__':
    main()
