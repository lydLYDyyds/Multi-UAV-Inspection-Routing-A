# -*- coding: utf-8 -*-
"""问题3:临时禁飞区(圆+时间窗)约束下的多机协同巡检调度。

约束语义(基于题目与假设 A3/边界约定):
- 无人机 8:00 起飞;禁飞区 z 在 [t0_z, t1_z](分钟,8:00 起算)生效。
- 飞行段(折线)在其**实际穿越时段**不得与"当时生效"的禁飞圆内部相交(边界允许通过)。
- 巡检服务若位于某禁飞圆内部,其 5 min 服务时段不得与该区生效时段重叠
  (可等待至解除后再服务,无人机允许空中等待)。

方法:
  Pass A:以问题2 的最优调度为初解 → 修复(几何绕行:双切线+圆弧;或等待至禁飞解除)
          → 独立校验(逐段×逐区×逐时刻),修复迭代至零违例。
  Pass B:带任务时间窗的 OR-Tools 再优化(窗取当前服务时刻所在的最大可行区间,
          L 网格围绕当前最优 L)→ 修复 → 校验;若 (makespan, δ) 更优则接受,迭代 ≤2 轮。
  输出 output/p3_{case}.json(含逐段航路点、等待记录、校验报告)。
"""
import sys, io, os, json
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from common import load_points, load_zones, A_DIR

CASE = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
TL = float(os.environ.get('P3_TL_SCALE', '1'))
SERVICE_S = 300
HORIZON_S = 9 * 3600
TAU_S = 0.1 / 55.0 * 3600.0
SERVICE_MIN = 5.0
TAU_MIN = 0.1 / 55.0 * 60.0

p = load_points(CASE)
xy, visits = p['xy'], p['visits']
zones = load_zones(CASE)                       # list of dict(c, r, t0, t1)
tasks = [(i, k) for i in range(len(xy)) for k in range(visits[i])]
M = len(tasks)
coords = np.vstack([np.zeros((1, 2)), xy[[t[0] for t in tasks]]])


def load_case(case):
    """切换当前算例数据(供外部模块如 sensitivity 调用)"""
    global xy, visits, zones, tasks, M, coords
    p = load_points(case)
    xy, visits = p['xy'], p['visits']
    zones = load_zones(case)
    tasks = [(i, k) for i in range(len(xy)) for k in range(visits[i])]
    M = len(tasks)
    coords = np.vstack([np.zeros((1, 2)), xy[[t[0] for t in tasks]]])

# ---------------- 几何工具 ----------------
def pt_inside(P, z):
    return float(np.linalg.norm(np.asarray(P, float) - z['c'])) < z['r'] - 1e-9


def seg_hits_disk(a, b, z):
    """线段 ab 与禁飞圆内部是否相交(相切/边界不算)"""
    a = np.asarray(a, float); b = np.asarray(b, float); c = z['c']; r = z['r']
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0:
        return pt_inside(a, z)
    tt = float(np.clip((c - a) @ ab / L2, 0, 1))
    return float(np.linalg.norm(a + tt * ab - c)) < r - 1e-9


def poly_hits_disk(pts, z):
    return any(seg_hits_disk(u, v, z) for u, v in zip(pts[:-1], pts[1:]))


def tangents(P, z):
    c, r = z['c'], z['r']
    d = np.asarray(P, float) - c
    D = float(np.linalg.norm(d))
    if D <= r:
        return None
    u = d / D
    t = np.sqrt(max(D * D - r * r, 0.0))
    base = c + (r * r / D) * u
    off = (r * t / D) * np.array([-u[1], u[0]])
    return base + off, base - off


def arc_pts(z, Ta, Tb, ccw):
    c, r = z['c'], z['r']
    a0 = np.arctan2(*(Ta - c)[::-1])
    a1 = np.arctan2(*(Tb - c)[::-1])
    if ccw:
        d = (a1 - a0) % (2 * np.pi)
    else:
        d = (a0 - a1) % (2 * np.pi)
    if d < 1e-9:
        return [Ta, Tb], 0.0
    n = max(2, int(np.ceil(d / (np.pi / 12))))     # ~15° 步长
    theta = d / n
    R = r / np.cos(theta / 2.0) + 0.05       # 向外偏移:弦不进入圆内(min-distance 校验兼容)
    if ccw:
        ang = a0 + d * np.linspace(0, 1, n + 1)
    else:
        ang = a0 - d * np.linspace(0, 1, n + 1)
    pts = [c + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
    return pts, d * R


def detour_around(pts, z):
    """折线 pts 绕过禁飞圆 z(要求端点均在圆外);返回新折线或 None"""
    a, b = pts[0], pts[-1]
    if pt_inside(a, z) or pt_inside(b, z):
        return None
    if not poly_hits_disk(pts, z):
        return pts
    Ta = tangents(a, z); Tb = tangents(b, z)
    if Ta is None or Tb is None:
        return None
    best, best_len = None, np.inf
    for ta in Ta:
        for tb in Tb:
            for ccw in (True, False):
                arc, alen = arc_pts(z, ta, tb, ccw)
                cand = [a] + [ta] + arc[1:-1] + [tb] + [b]
                L = float(np.linalg.norm(a - ta)) + alen + float(np.linalg.norm(tb - b))
                if L < best_len:
                    best, best_len = cand, L
    if best is None:
        return None
    return best


def path_len(pts):
    return sum(float(np.linalg.norm(np.asarray(u, float) - np.asarray(v, float)))
               for u, v in zip(pts[:-1], pts[1:]))


def active_overlap(t0a, t1a, t0b, t1b):
    return max(t0a, t0b) < min(t1a, t1b)     # 开区间重叠(瞬时接触不算)


# ---------------- 修复与校验 ----------------
def process_leg(u, v, t, zones, k_idx, wait_log):
    """处理一段 u→v:返回 (path, new_t)。必要时绕行或等待;等待记入 wait_log。"""
    path = [u, v]
    for _ in range(40):                          # 迭代上限
        L = path_len(path)
        t_arr = t + L * TAU_MIN
        conf = [z for z in zones
                if poly_hits_disk(path, z) and active_overlap(t, t_arr, z['t0'], z['t1'])]
        if not conf:
            break
        need_wait = 0.0
        for z in conf:
            if pt_inside(u, z) or pt_inside(v, z):
                need_wait = max(need_wait, z['t1'] - t)
        if need_wait > 1e-9:
            t += need_wait
            wait_log.append((k_idx, round(need_wait, 2), 'leg'))
            continue
        # 两端点均在圆外 → 逐区绕行
        newpath, ok = path, True
        for z in conf:
            np2 = detour_around(newpath, z)
            if np2 is None:
                ok = False
                break
            newpath = np2
        if ok and not any(poly_hits_disk(newpath, z) for z in conf):
            path = newpath
            continue
        # 兜底:等待全部冲突区解除
        wt = max(z['t1'] for z in conf) - t
        t += max(wt, 0.0)
        wait_log.append((k_idx, round(max(wt, 0.0), 2), 'leg-fallback'))
    t += path_len(path) * TAU_MIN
    return path, t


def repair_route(seq_pts, zones, t_start=0.0):
    """给定任务点序列(含起点基地),返回 (legs, wait_log, end_min)。
    legs 含返航段(共 n_tasks+1 段);时刻含等待与服务。"""
    t = float(t_start)
    legs, wait_log = [], []
    for k in range(1, len(seq_pts)):             # 去程段 1..n_tasks
        path, t = process_leg(seq_pts[k - 1], seq_pts[k], t, zones, k, wait_log)
        legs.append(path)
        # 服务时间窗口(服务点若在生效禁飞区内 → 等待至解除后服务)
        v = seq_pts[k]
        for z in zones:
            if pt_inside(v, z) and active_overlap(t, t + SERVICE_MIN, z['t0'], z['t1']):
                wt = z['t1'] - t
                wait_log.append((k, round(wt, 2), 'service'))
                t += wt
        t += SERVICE_MIN
    # 返航段
    k_last = len(seq_pts)
    path, t = process_leg(seq_pts[-1], seq_pts[0], t, zones, k_last, wait_log)
    legs.append(path)
    return legs, wait_log, t


def build_schedule(routes_data, zones):
    """把 p2 风格 routes 修复为带航路点/等待/时刻的调度"""
    out = []
    for r in routes_data:
        pts = [np.array([0., 0.])] + [xy[s['point_id'] - 1] for s in r['seq']]
        legs, waits, end_min = repair_route(pts, zones)
        # 由等待日志精确重建时刻:leg_dep=起飞(等待后),arr=到达,dep=离开(服务后)
        t = 0.0
        arr, dep, leg_dep = [], [], []
        for k in range(len(r['seq'])):
            w_leg = sum(wt for (kk, wt, kind) in waits if kk == k + 1 and kind.startswith('leg'))
            leg_dep.append(round(t + w_leg, 2))
            t = t + w_leg + path_len(legs[k]) * TAU_MIN
            arr.append(round(t, 2))
            w_srv = sum(wt for (kk, wt, kind) in waits if kk == k + 1 and kind == 'service')
            t += w_srv
            dep.append(round(t + SERVICE_MIN, 2))
            t += SERVICE_MIN
        legs_u = [round(path_len(L), 3) for L in legs]
        n_t = len(r['seq'])
        w_ret = sum(wt for (kk, wt, kind) in waits if kk == n_t + 1 and kind.startswith('leg'))
        return_dep = t + w_ret
        out.append(dict(veh=r['veh'], seq=r['seq'], arr_min=arr, dep_min=dep,
                        leg_dep_min=leg_dep, return_dep_min=round(return_dep, 2),
                        legs_u=legs_u, legs_wp=[[list(map(float, q)) for q in L] for L in legs],
                        travel_units=round(sum(legs_u), 2),
                        end_s=int(round(end_min * 60)),
                        busy_h=round(end_min / 60, 4), n_tasks=len(r['seq']),
                        waits=waits))
    span = max(x['end_s'] for x in out) / 3600
    delta = round(span - min(x['busy_h'] for x in out), 4)
    return out, span, delta


def verify(routes, zones):
    """返回违例列表;空列表=通过。逐段×逐区×逐时刻交叉核对。
    时间基准以调度自身记录的 leg_dep/arr/dep(已含等待)为准,只检真实飞行区间。"""
    viol = []
    for r in routes:
        t = 0.0
        pts = [np.array([0., 0.])] + [xy[s['point_id'] - 1] for s in r['seq']]
        for k, (leg, s) in enumerate(zip(r['legs_wp'][:-1], r['seq'])):
            t_dep = r['leg_dep_min'][k]
            t_arr = r['arr_min'][k]
            for z in zones:
                if poly_hits_disk(leg, z) and active_overlap(t_dep, t_arr, z['t0'], z['t1']):
                    viol.append(dict(veh=r['veh'], kind='leg', k=k + 1, zone=z['zone_id'],
                                     t=[round(t_dep, 2), round(t_arr, 2)]))
            for z in zones:
                if pt_inside(pts[k + 1], z) and \
                        active_overlap(r['dep_min'][k] - SERVICE_MIN, r['dep_min'][k], z['t0'], z['t1']):
                    viol.append(dict(veh=r['veh'], kind='service', k=k + 1, zone=z['zone_id'],
                                     t=[round(r['dep_min'][k] - SERVICE_MIN, 2), round(r['dep_min'][k], 2)]))
            t = r['dep_min'][k]
        # 返航段
        t_dep = r.get('return_dep_min', r['dep_min'][-1])
        t_arr = r['end_s'] / 60.0
        for z in zones:
            if poly_hits_disk(r['legs_wp'][-1], z) and active_overlap(t_dep, t_arr, z['t0'], z['t1']):
                viol.append(dict(veh=r['veh'], kind='leg', k='return', zone=z['zone_id'],
                                 t=[round(t_dep, 2), round(t_arr, 2)]))
    return viol


# ---------------- Pass B:带时间窗的 OR-Tools 再优化 ----------------
def solve_vrp(N, L_sec, windows_sec, tlimit, seed):
    """windows_sec: {task_node: (lo, hi)} 到达时刻窗;L_sec: 每机工作时长下界"""
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


def task_windows(routes_repaired):
    """以当前调度为基准,给每个任务节点取包含其当前到达时刻的最大可行区间"""
    cur = {}
    for r in routes_repaired:
        for k, s in enumerate(r['seq']):
            cur[(s['point_id'], s['visit_no'])] = r['arr_min'][k] * 60
    win = {}
    for (pid, vno), t_cur in cur.items():
        i = pid - 1                                 # 0 基
        pi = xy[i]
        banned = [(z['t0'] * 60, z['t1'] * 60) for z in zones if pt_inside(pi, z)]
        lo, hi = 0.0, float(HORIZON_S)
        for (a, b) in banned:
            if t_cur >= b:
                lo = max(lo, b)
            elif t_cur + SERVICE_S <= a:
                hi = min(hi, a)
            else:
                lo, hi = max(lo, b), min(hi, float(HORIZON_S))
        node = tasks.index((i, vno - 1)) + 1
        win[node] = (lo, hi)
    return win


# ---------------- 主流程 ----------------
def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    with open(os.path.join(A_DIR, 'output', f'p2_{CASE}.json'), encoding='utf-8') as f:
        p2 = json.load(f)
    N = p2['N']
    base_routes = p2['grid'][0]['routes']
    Lstar_h = p2['grid'][0]['L_h'] or 0.0

    # Pass A(单次确定性修复,修复后必须零违例)
    routesA, spanA, deltaA = build_schedule(base_routes, zones)
    violA = verify(routesA, zones)
    print(f'PassA: span={spanA:.3f}h δ={deltaA:.3f}h 违例={len(violA)}', flush=True)
    if violA:
        raise RuntimeError(f'PassA 修复后仍有违例(修复器缺陷): {violA[:5]}')

    best = (spanA, deltaA, routesA, 'A')
    print(f'PassA 终态: span={spanA:.3f}h δ={deltaA:.3f}h 违例={len(violA)}', flush=True)

    # Pass B(两模式:带时间窗窄网格 / 无时间窗宽网格)
    win = task_windows(routesA)
    for it in range(2):
        tlimit = max(5, int(30 * TL))
        cands = []
        for use_win, L_set in ((True, sorted({max(0, Lstar_h - 1), Lstar_h, Lstar_h + 1})),
                               (False, list(range(0, int(spanA) + 2)))):
            for Lh in L_set:
                L_sec = int(Lh * 3600)
                raw = solve_vrp(N, L_sec, win if use_win else {}, tlimit, 7 + it + (10 if use_win else 0))
                if raw is None:
                    continue
                routesB, spanB, deltaB = build_schedule(raw, zones)
                violB = verify(routesB, zones)
                if not violB:
                    cands.append((spanB, deltaB, routesB, f'B{it}{"w" if use_win else "f"}'))
                    print(f'  PassB it{it} {"窗" if use_win else "自由"} L={Lh:.0f}h: '
                          f'span={spanB:.3f}h δ={deltaB:.3f}h', flush=True)
        if cands:
            cand = min(cands, key=lambda c: (c[0], c[1]))
            if (cand[0], cand[1]) < (best[0], best[1]):
                best = cand
                win = task_windows(cand[2])
                continue
        break

    span, delta, routes, src = best
    viol = verify(routes, zones)
    assert not viol, f'最终调度仍有违例: {viol[:3]}'
    ends = sorted(r['end_s'] for r in routes)
    tmax_h, tmin_h = round(max(ends) / 3600, 4), round(min(ends) / 3600, 4)
    summary = dict(case=CASE, N=N, tmax_h=tmax_h, tmin_h=tmin_h,
                   delta_h=round(tmax_h - tmin_h, 4), source=src,
                   passA=dict(span_h=round(spanA, 4), delta_h=round(deltaA, 4)),
                   violations=len(viol))
    out = dict(summary=summary, routes=routes)
    os.makedirs(os.path.join(A_DIR, 'output'), exist_ok=True)
    path = os.path.join(A_DIR, 'output', f'p3_{CASE}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'[P3 {CASE}] source={src} N={N} Tmax={tmax_h}h Tmin={tmin_h}h δ={delta}h 违例=0 -> {path}')


if __name__ == '__main__':
    main()
