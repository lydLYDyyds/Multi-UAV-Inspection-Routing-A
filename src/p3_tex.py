# -*- coding: utf-8 -*-
"""问题3 求解器:以问题2 调度为初解,按动态航段代价重建合法调度并再优化。

Pass A:逐段用 nofly.arc_cost 选择直飞/等待/绕行,重建各机时刻;
Pass B:以当前调度为基准给任务加到达时间窗,OR-Tools 再优化(带窗 L*±1h 与
        自由 L 网格两种模式),修复后取 (Tmax, δ) 词典序更优者,最多两轮。
输出 output/p3tex_{case}.json(与 p3_*.json 同结构)。

用法:python src/p3_tex.py Case1   环境变量 P3TEX_NO_B=1 跳过 Pass B
"""
import sys, io, os, json
import numpy as np
from common import load_points, load_zones, A_DIR
import nofly

CASE = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
SERVICE_MIN = 5.0


def path_len(pts):
    pts = [np.asarray(q, float) for q in pts]
    return sum(float(np.linalg.norm(u - v)) for u, v in zip(pts[:-1], pts[1:]))


def build_all(routes_data, xy, zones, F):
    """按动态航段代价重建全部无人机调度(核心逻辑在 nofly.build_route)。"""
    out = []
    for r in routes_data:
        tasks = [(s['point_id'], s['visit_no']) for s in r['seq']]
        rr = nofly.build_route(tasks, xy, zones, F, veh=r['veh'])
        assert rr is not None, f'机 {r["veh"]} 路线不可行'
        out.append(rr)
    return out


def solve_vrp(N, L_sec, windows_sec, tlimit, seed, M, tasks, coords):
    """带任务到达时间窗 + 每机时长下界的 min-span VRP(与问题1/2 同机制)。"""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    TAU_S = 0.1 / 55.0 * 3600.0
    HORIZON_S = 9 * 3600
    manager = pywrapcp.RoutingIndexManager(M + 1, N, 0)
    routing = pywrapcp.RoutingModel(manager)
    node2idx = {n: manager.NodeToIndex(n) for n in range(M + 1)}
    idx2node = {i: n for n, i in node2idx.items()}
    T = [[0] * (M + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(M + 1):
            T[i][j] = int(round(float(np.hypot(*(coords[i] - coords[j]))) * TAU_S)) \
                + (300 if j > 0 else 0)

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
    for node, (lo, hi) in windows_sec.items():
        tdim.CumulVar(manager.NodeToIndex(node)).SetRange(int(lo), int(hi))
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.log_search = False
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
                seq.append(node)
            idx = sol.Value(routing.NextVar(idx))
        if not seq:
            continue
        routes.append(dict(veh=v + 1, n_tasks=len(seq),
                           seq=[dict(point_id=int(tasks[n - 1][0]) + 1,
                                      visit_no=int(tasks[n - 1][1]) + 1) for n in seq]))
    if len(routes) < N or any(len(r['seq']) == 0 for r in routes):
        return None
    return routes


def task_windows(routes_repaired, xy, zones, tasks):
    """以当前调度为基准,给每个任务节点取包含其当前到达时刻的最大可行区间(秒制)。
    返回 {node_index: (lo, hi)},node_index 为 RoutingIndexManager 的节点号。"""
    cur = {}
    for r in routes_repaired:
        for k, s in enumerate(r['seq']):
            cur[(s['point_id'], s['visit_no'])] = r['arr_min'][k] * 60
    win = {}
    for (pid, vno), t_cur in cur.items():
        pi = xy[pid - 1]
        banned = [(z['t0'] * 60, z['t1'] * 60) for z in zones if nofly.pt_inside(pi, z)]
        lo, hi = 0.0, float(9 * 3600)
        for (a, b) in banned:
            if t_cur >= b:
                lo = max(lo, b)
            elif t_cur + 300 <= a:
                hi = min(hi, a)
            else:
                lo, hi = max(lo, b), min(hi, float(9 * 3600))
        node = tasks.index((pid - 1, vno - 1)) + 1
        win[node] = (lo, hi)
    return win


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    TL = float(os.environ.get('P3_TL_SCALE', '1'))
    p = load_points(CASE)
    xy, visits = p['xy'], p['visits']
    zones = load_zones(CASE)
    tasks = [(i, k) for i in range(len(xy)) for k in range(visits[i])]
    M = len(tasks)
    coords = np.vstack([np.zeros((1, 2)), xy[[t[0] for t in tasks]]])
    with open(os.path.join(A_DIR, 'output', f'p2_{CASE}.json'), encoding='utf-8') as f:
        p2 = json.load(f)
    N = p2['N']
    base = p2['grid'][0]['routes']
    Lstar_h = p2['grid'][0]['L_h'] or 0.0
    pts_all = np.vstack([np.zeros((1, 2)), xy])
    F = nofly.build_forbidden_table(pts_all, zones)          # 预计算禁止出发区间
    print(f'F 表:非空弧 {len(F)} 条,区间段合计 '
          f'{sum(len(v) for v in F.values())}', flush=True)

    routes = build_all(base, xy, zones, F)
    viol = nofly.validate_schedule(routes, zones)
    assert not viol, f'PassA 违例: {viol[:3]}'
    ends = sorted(r['end_s'] for r in routes)
    passA = (round(max(ends) / 3600, 4),
             round((max(ends) - min(ends)) / 3600, 4))
    print(f'PassA: span={passA[0]:.3f}h δ={passA[1]:.3f}h 违例=0', flush=True)

    best = (passA[0], passA[1], routes, 'tex-A')
    if os.environ.get('P3TEX_NO_B'):
        win = {}
    else:
        win = task_windows(routes, xy, zones, tasks)
    for it in range(0 if os.environ.get('P3TEX_NO_B') else 2):
        tlimit = max(5, int(30 * TL))
        cands = []
        L_set = sorted({max(0.0, Lstar_h - 1), Lstar_h, Lstar_h + 1})
        for use_win, Ls in ((True, L_set),
                            (False, list(range(0, int(best[0]) + 2)))):
            for Lh in Ls:
                raw = solve_vrp(N, int(Lh * 3600), win if use_win else {},
                                tlimit, 7 + it + (10 if use_win else 0),
                                M, tasks, coords)
                if raw is None:
                    continue
                rr = build_all(raw, xy, zones, F)
                if nofly.validate_schedule(rr, zones):
                    continue
                e = sorted(x['end_s'] for x in rr)
                cands.append((round(max(e) / 3600, 4),
                              round((max(e) - min(e)) / 3600, 4), rr,
                              f'B{it}{"w" if use_win else "f"}'))
                print(f'  PassB it{it} {"窗" if use_win else "自由"} L={Lh:.0f}h: '
                      f'span={cands[-1][0]:.3f}h δ={cands[-1][1]:.3f}h', flush=True)
        if cands:
            cand = min(cands, key=lambda c: (c[0], c[1]))
            if (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
                win = task_windows(cand[2], xy, zones, tasks)
                continue
        break

    tmax_h, tmin_h, routes, src = best[0], best[0] - best[1], best[2], best[3]
    tmin_h = round(tmin_h, 4)
    n_det = sum(1 for r in routes for L in r['legs_wp'] if len(L) > 2)
    n_wait = sum(1 for r in routes for (k, t_s, w, kind, pos) in r['waits']
                 if kind.startswith('leg'))
    n_wait_srv = sum(1 for r in routes for (k, t_s, w, kind, pos) in r['waits']
                     if kind == 'service')
    tot_wait = round(sum(w for r in routes for (k, t_s, w, kind, pos) in r['waits']), 2)
    summary = dict(case=CASE, N=N, tmax_h=tmax_h, tmin_h=tmin_h,
                   delta_h=round(tmax_h - tmin_h, 4), source=src,
                   passA=dict(span_h=passA[0], delta_h=passA[1]),
                   violations=0, n_detour_legs=n_det,
                   n_wait_events=n_wait, n_service_waits=n_wait_srv,
                   total_wait_min=tot_wait)
    out = dict(summary=summary, routes=routes)
    os.makedirs(os.path.join(A_DIR, 'output'), exist_ok=True)
    path = os.path.join(A_DIR, 'output', f'p3tex_{CASE}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'[P3TEX {CASE}] source={src} N={N} Tmax={tmax_h}h Tmin={tmin_h}h '
          f'δ={summary["delta_h"]}h 违例=0 绕行段={n_det} 等待事件={n_wait}'
          f'(+服务{n_wait_srv}) 总等待={tot_wait}min -> {path}', flush=True)

    old_path = os.path.join(A_DIR, 'output', f'p3_{CASE}.json')
    if os.path.exists(old_path):
        with open(old_path, encoding='utf-8') as f:
            old = json.load(f)
        os_ = old['summary']
        print(f'对比旧p3: Tmax {os_["tmax_h"]}h -> {tmax_h}h | '
              f'Tmin {os_["tmin_h"]}h -> {tmin_h}h | δ {os_["delta_h"]}h -> {summary["delta_h"]}h '
              f'(旧版基于旧p2初解+未绕行)', flush=True)


if __name__ == '__main__':
    main()
