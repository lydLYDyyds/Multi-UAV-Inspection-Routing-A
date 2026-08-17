# -*- coding: utf-8 -*-
"""问题1:min-makespan 多无人机 VRP 求解(OR-Tools 9.15)。

建模:
- 节点 0 = 基地(起终点);节点 1..M = 巡检任务(每个"巡检次数"是一个任务节点,
  同一点的多次巡检是坐标重合的多个节点,服务时间各 5 min)。
- transit(i,j) = 飞行时间(i→j,秒) + service(j)(j≠0 时 +300 s)。
  时间维 cumul 在节点 j 的值 = 到达 j 的时刻 + 服务 5 min;终点 cumul = 返航时刻。
- 目标:min 全局 span(= max 各机返航时刻),GlobalSpanCostCoefficient。
- 1 u = 0.1 km,55 km/h → 每单位 6.54545... s,取整到秒。

实现注记(重要):OR-Tools 9.15 + Py3.13 下,在 transit 回调内调用 SWIG 方法
(IndexToNode)会抛 OverflowError。规避:主线程预计算 transit 矩阵与 idx2node 字典,
回调内只做纯 Python 查表。

用法:python src/p1_case.py Case1  (写 output/p1_{case}.json)
"""
import sys, io, os, json
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common import load_points, A_DIR

CASE = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
TL = float(os.environ.get('P1_TL_SCALE', '1'))     # 时间缩放(调试用)
N_LB = {'Case1': 2, 'Case2': 2, 'Case3': 2, 'Case4': 3}   # 由 p1_lb.py 得出
SERVICE_S = 300
HORIZON_S = 9 * 3600
TAU_S = 0.1 / 55.0 * 3600.0          # 秒/单位

p = load_points(CASE)
xy, visits = p['xy'], p['visits']
tasks = [(i, k) for i in range(len(xy)) for k in range(visits[i])]
M = len(tasks)
# 按“任务节点”建坐标:节点 0 = 基地,节点 1..M = 各巡检任务(同点多任务是重合节点)
coords = np.vstack([np.zeros((1, 2)), xy[[t[0] for t in tasks]]])


def dist_u(i, j):
    return float(np.hypot(*(coords[i] - coords[j])))


def build(N):
    manager = pywrapcp.RoutingIndexManager(M + 1, N, 0)
    routing = pywrapcp.RoutingModel(manager)
    # --- 主线程预计算:节点↔索引映射 + transit 矩阵(节点维度,秒) ---
    node2idx = {n: manager.NodeToIndex(n) for n in range(M + 1)}
    idx2node = {i: n for n, i in node2idx.items()}
    T = [[0] * (M + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(M + 1):
            T[i][j] = int(round(dist_u(i, j) * TAU_S)) + (SERVICE_S if j > 0 else 0)

    def cb(fi, tj):
        return T[idx2node.get(fi, 0)][idx2node.get(tj, 0)]

    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    routing.AddDimension(transit, 0, 20 * HORIZON_S, True, 'Time')
    tdim = routing.GetDimensionOrDie('Time')
    tdim.SetGlobalSpanCostCoefficient(100)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.log_search = False
    return manager, routing, tdim, sp, idx2node


def solve(N, tlimit, seed):
    manager, routing, tdim, sp, idx2node = build(N)
    sp.time_limit.seconds = tlimit
    sp.sat_parameters.random_seed = seed
    sol = routing.SolveWithParameters(sp)
    if sol is None:
        return None
    return extract(manager, routing, tdim, sol, N, idx2node)


def extract(manager, routing, tdim, sol, N, idx2node):
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
        legs_u = [round(dist_u(a, b), 3) for a, b in zip(nodes[:-1], nodes[1:])]
        arr = [round((s['cumul'] - SERVICE_S) / 60, 2) for s in seq]   # 分钟
        dep = [round(s['cumul'] / 60, 2) for s in seq]
        routes.append(dict(veh=v + 1,
                           seq=[dict(point_id=int(tasks[s['node'] - 1][0]) + 1,
                                     visit_no=int(tasks[s['node'] - 1][1]) + 1) for s in seq],
                           arr_min=arr, dep_min=dep, legs_u=legs_u,
                           travel_units=round(sum(legs_u), 2), end_s=end_s,
                           busy_h=round(end_s / 3600, 4), n_tasks=len(seq)))
    span_s = max(r['end_s'] for r in routes)
    return dict(span_s=span_s, span_h=round(span_s / 3600, 4), routes=routes)


def try_feasible(N, tlimit, seeds):
    """返回 span≤9h 的最优解;找不到则 None"""
    best = None
    for seed in seeds:
        r = solve(N, tlimit, seed)
        if r is not None and r['span_s'] <= HORIZON_S:
            if best is None or r['span_s'] < best['span_s']:
                best = r
    return best


# ---------- Nmin 搜索 ----------
def main():
    global Nmin
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    N = N_LB[CASE]
    log = []
    while True:
        r = try_feasible(N, max(3, int(40 * TL)), [1, 2, 3])
        empty = r is not None and any(x['n_tasks'] == 0 for x in r['routes'])
        log.append(dict(N=N, feasible=r is not None, span_h=r['span_h'] if r else None))
        print(f'N={N}: feasible={r is not None}'
              + (f' span={r["span_h"]:.3f}h' if r else ''), flush=True)
        if r is None:
            N += 1
            continue
        if empty and N > N_LB[CASE]:
            N -= 1                      # 有空机 → 少一台也必然可行
            continue
        Nmin = N
        break

    print(f'Nmin = {Nmin}', flush=True)

    # ---------- 问题1.2:Nmin 下 min makespan(多种子取优) ----------
    best = None
    for seed in range(1, 7):
        r = solve(Nmin, max(5, int(90 * TL)), seed)
        if r is not None and (best is None or r['span_s'] < best['span_s']):
            best = r
            print(f'  seed {seed}: span={r["span_h"]:.3f}h (best so far)', flush=True)
    assert best is not None and best['span_s'] <= HORIZON_S, 'Nmin 解超出 9h,逻辑错误'

    # ---------- Nmin-1 的不可行证据 ----------
    ev = None
    if Nmin > N_LB[CASE]:
        best1 = None
        for seed in range(1, 4):
            r = solve(Nmin - 1, max(5, int(90 * TL)), seed)
            if r is not None and (best1 is None or r['span_s'] < best1['span_s']):
                best1 = r
        ev = dict(N=Nmin - 1, best_span_h=round(best1['span_h'], 3) if best1 else None,
                  feasible_9h=bool(best1 and best1['span_s'] <= HORIZON_S))
        print(f'证据: N={Nmin-1} 最优尝试 span={ev["best_span_h"]}h (>9h → 不可行证据,非严格证明)', flush=True)

    tmax_h = best['span_h']
    tmin_h = round(min(r['busy_h'] for r in best['routes']), 4)
    summary = dict(case=CASE, Nmin=Nmin, N_LB=N_LB[CASE], tmax_h=tmax_h, tmin_h=tmin_h,
                   evidence_Nminus1=ev, search_log=log)
    out = dict(summary=summary, best=best)
    os.makedirs(os.path.join(A_DIR, 'output'), exist_ok=True)
    path = os.path.join(A_DIR, 'output', f'p1_{CASE}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n[P1 {CASE}] Nmin={Nmin}  Tmax={tmax_h:.3f}h  Tmin={tmin_h:.3f}h  -> {path}')


if __name__ == '__main__':
    main()
