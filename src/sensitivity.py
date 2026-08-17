# -*- coding: utf-8 -*-
"""敏感性分析(全部基于真实重跑,不编造)。

S1 速度 55→50/60 km/h:对问题1 最优路径重算时长 + Nmin-1 在 60km/h 下是否可行(60s 重解);
S2 服务时间 5→4/6 min:同一路径重算时长;
S3 禁飞半径 ×1.1/1.2:问题3 修复+校验(仅修复,不重优化);
S4 Case4 Z8 解读为"17:00 起持续生效":重跑问题3 修复;
S5 同点多巡检必须间隔(禁止相邻):重解问题1(Nmin, 2 种子×60s);
S6 随机删 5% 任务(3 种子):重解问题1(Nmin, 45s)。

输出:output/sensitivity.json + 控制台报告。
"""
import sys, io, os, json, random
import numpy as np
from common import load_points, load_zones, A_DIR
from p3_case import build_schedule, verify

CASES = ['Case1', 'Case2', 'Case3', 'Case4']
TAU_MIN = 0.1 / 55.0 * 60.0
SERVICE_MIN = 5.0


def load_routes(prob, case):
    with open(os.path.join(A_DIR, 'output', f'{prob}_{case}.json'), encoding='utf-8') as f:
        return json.load(f)


def reschedule_times(case, routes, vkmh=55.0, smin=5.0):
    """同一路径在不同速度/服务时长下的 Tmax(h)"""
    p = load_points(case)
    xy = p['xy']
    tau = 0.1 / vkmh * 60
    ends = []
    for r in routes:
        prev, dist_u = 0, 0.0
        for s in r['seq']:
            pxy = xy[s['point_id'] - 1]
            dist_u += float(np.hypot(*(pxy - (xy[prev - 1] if prev else [0, 0]))))
            prev = s['point_id']
        dist_u += float(np.hypot(*(xy[prev - 1] if prev else [0, 0])))
        ends.append(dist_u * tau + r['n_tasks'] * smin)
    return round(max(ends) / 60, 3)


def solve_vrp_tasks(case, tasks_sub, N, tlimit, seed, forbid_consec=False):
    """通用 min-span VRP(任务子集),返回 Tmax(h) 或 None"""
    xy_all = load_points(case)['xy']
    M = len(tasks_sub)
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    coords = np.vstack([np.zeros((1, 2)), xy_all[[t[0] for t in tasks_sub]]])
    manager = pywrapcp.RoutingIndexManager(M + 1, N, 0)
    routing = pywrapcp.RoutingModel(manager)
    node2idx = {n: manager.NodeToIndex(n) for n in range(M + 1)}
    idx2node = {i: n for n, i in node2idx.items()}
    T = [[0] * (M + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(M + 1):
            T[i][j] = int(round(float(np.hypot(*(coords[i] - coords[j]))) * 0.1 / 55 * 3600)) \
                + (300 if j > 0 else 0)
    if forbid_consec:
        for a in range(1, M + 1):
            for b in range(1, M + 1):
                if a != b and tasks_sub[a - 1][0] == tasks_sub[b - 1][0]:
                    routing.NextVar(node2idx[a]).RemoveValue(node2idx[b])

    def cb(fi, tj):
        return T[idx2node.get(fi, 0)][idx2node.get(tj, 0)]
    transit = routing.RegisterTransitCallback(cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    routing.AddDimension(transit, 0, 20 * 9 * 3600, True, 'Time')
    tdim = routing.GetDimensionOrDie('Time')
    tdim.SetGlobalSpanCostCoefficient(100)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.log_search = False
    sp.time_limit.seconds = tlimit
    sp.sat_parameters.random_seed = seed
    sol = routing.SolveWithParameters(sp)
    if sol is None:
        return None
    ends = [sol.Value(tdim.CumulVar(manager.GetEndIndex(v))) for v in range(N)]
    return round(max(ends) / 3600, 3)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    report = {}
    # ---------- S1 / S2 ----------
    print('== S1 速度扰动 / S2 服务时长扰动 ==', flush=True)
    for case in CASES:
        d = load_routes('p1', case)
        routes = d['best']['routes']
        row = {}
        for vkmh in (50, 55, 60):
            row[f'v{vkmh}_Tmax'] = reschedule_times(case, routes, vkmh=vkmh)
        for smin in (4, 5, 6):
            row[f's{smin}_Tmax'] = reschedule_times(case, routes, smin=smin)
        report[f'S1S2_{case}'] = row
        print(case, row, flush=True)

    # S1 补充:Nmin-1 在 60 km/h 下重解
    print('== S1b: v=60 时 Nmin-1 重解 ==', flush=True)
    for case in CASES:
        Nmin = load_routes('p1', case)['summary']['Nmin']
        p = load_points(case)
        tasks = [(i, k) for i in range(len(p['xy'])) for k in range(p['visits'][i])]
        r = None
        # 60 km/h 等价于把时间缩小 55/60:用 55km/h 求解、比较 span ≤ 9h×60/55
        span = solve_vrp_tasks(case, tasks, Nmin - 1, 60, 11)
        if span is not None:
            span60 = round(span * 55 / 60, 3)     # 换算回 60km/h 的真实时长
        else:
            span60 = None
        report[f'S1b_{case}'] = dict(N=Nmin - 1, span60_h=span60, feasible_9h=bool(span60 and span60 <= 9))
        print(case, f'N={Nmin-1}: v=60 span={span60}h', flush=True)

    # ---------- S3 / S4 ----------
    print('== S3 禁飞半径膨胀 / S4 Z8 持续生效 ==', flush=True)
    for case in CASES:
        p3_path = os.path.join(A_DIR, 'output', f'p3_{case}.json')
        if not os.path.exists(p3_path):
            print(case, 'p3 结果未就绪,跳过 S3/S4', flush=True)
            continue
        d = load_routes('p3', case)
        zones = load_zones(case)
        base_routes = d['routes']
        row = {}
        for eps in (1.0, 1.1, 1.2):
            zz = [dict(zone_id=z['zone_id'], c=z['c'], r=z['r'] * eps,
                       t0=z['t0'], t1=z['t1']) for z in zones]
            routes, span, delta = build_schedule(base_routes, zz)
            viol = verify(routes, zz)
            row[f'r{eps}'] = dict(Tmax=round(span, 3), delta=round(delta, 3), viol=len(viol))
        if case == 'Case4':
            zz = [dict(zone_id=z['zone_id'], c=z['c'], r=z['r'],
                       t0=(540.0 if z['zone_id'] == 'Z8' else z['t0']),
                       t1=(720.0 if z['zone_id'] == 'Z8' else z['t1'])) for z in zones]
            routes, span, delta = build_schedule(base_routes, zz)
            viol = verify(routes, zz)
            row['Z8_persist'] = dict(Tmax=round(span, 3), delta=round(delta, 3), viol=len(viol))
        report[f'S3S4_{case}'] = row
        print(case, row, flush=True)

    # ---------- S5 ----------
    print('== S5 同点连续巡检禁止(重解) ==', flush=True)
    for case in CASES:
        Nmin = load_routes('p1', case)['summary']['Nmin']
        p = load_points(case)
        tasks = [(i, k) for i in range(len(p['xy'])) for k in range(p['visits'][i])]
        vals = [solve_vrp_tasks(case, tasks, Nmin, 60, s, forbid_consec=True) for s in (21, 22)]
        report[f'S5_{case}'] = dict(N=Nmin, spans=vals)
        print(case, '禁止连续重解 spans=', vals, flush=True)

    # ---------- S6 ----------
    print('== S6 随机删 5% 任务 ==', flush=True)
    for case in CASES:
        Nmin = load_routes('p1', case)['summary']['Nmin']
        p = load_points(case)
        tasks = [(i, k) for i in range(len(p['xy'])) for k in range(p['visits'][i])]
        spans = []
        for seed in (31, 32, 33):
            rng = random.Random(seed)
            keep = set(rng.sample(range(len(tasks)), int(len(tasks) * 0.95)))
            sub = [t for idx, t in enumerate(tasks) if idx in keep]
            spans.append(solve_vrp_tasks(case, sub, Nmin, 45, seed))
        report[f'S6_{case}'] = dict(spans=spans)
        print(case, '删5%任务重解 spans=', spans, flush=True)

    with open(os.path.join(A_DIR, 'output', 'sensitivity.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print('\n已保存 output/sensitivity.json', flush=True)


if __name__ == '__main__':
    main()
