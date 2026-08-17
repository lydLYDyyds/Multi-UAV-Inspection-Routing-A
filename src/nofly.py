# -*- coding: utf-8 -*-
"""动态禁飞区航段级硬约束模型。

对任意航段 i→j,将直线航迹参数化为 p(λ)=p_i+λ(p_j−p_i)、t=s+λτ_ij,
与禁飞圆联立解二次方程得到进入/离开参数 λ_in/λ_out,由此把"空间障碍+时间窗"
约束解析投影为该航段的禁止出发时间集合 F_ij;在此基础上构造直飞、等待(含圆外
悬停)与绕行三种候选策略,取最短合法时间为动态航段代价 τ̃_ij(s),并按此递推整机
调度时刻。飞行、服务、等待三类状态均按统一时空硬约束验证。

约定:时间单位分钟(8:00 起算);开圆(仅边界接触不算进入);时间窗重叠按开区间。
"""
import numpy as np

TAU_MIN = 0.1 / 55.0 * 60.0           # 分钟/坐标单位
MARGIN = 0.03                          # 窗口边界安全裕度(分钟),抵消 0.01 min 舍入
SERVICE_MIN = 5.0


def seg_zone_params(A, B, z):
    """线段 AB 与禁飞圆 z 的解析求交。返回 lam_clip=(λ_in,λ_out)∩[0,1],
    None 表示不相交或仅边界接触(开圆约定)。"""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d = B - A
    c = z['c']
    R = z['r']
    a = float(d @ d)
    b = 2.0 * float((A - c) @ d)
    cc = float((A - c) @ (A - c)) - R * R
    if a <= 1e-9:
        inside = cc < -1e-9
        return dict(lam=None, lam_clip=(0.0, 0.0) if inside else None)
    disc = b * b - 4.0 * a * cc
    if disc <= 1e-12 * max(1.0, a * a):
        return dict(lam=None, lam_clip=None)
    sq = np.sqrt(disc)
    l1, l2 = (-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)
    lo, hi = max(0.0, l1), min(1.0, l2)
    if lo > hi + 1e-12 or hi - lo <= 1e-7:
        return dict(lam=(l1, l2), lam_clip=None)
    return dict(lam=(l1, l2), lam_clip=(lo, hi))


def overlap(a0, a1, b0, b1):
    """开区间重叠判定。"""
    return max(a0, b0) < min(a1, b1)


def leg_zone_conflict(A, B, z, t_dep, tau):
    """出发时刻 t_dep 下航段 AB 与区 z 是否冲突;冲突返回穿越时段 (t1,t2)。"""
    r = seg_zone_params(A, B, z)
    if r['lam_clip'] is None:
        return None
    lo, hi = r['lam_clip']
    t1 = t_dep + lo * tau
    t2 = t_dep + hi * tau
    return (t1, t2) if overlap(t1, t2, z['t0'], z['t1']) else None


def conflicting_zones(A, B, zs, t_dep, tau):
    return [z for z in zs if leg_zone_conflict(A, B, z, t_dep, tau) is not None]


def pt_inside(P, z):
    return float(np.linalg.norm(np.asarray(P, float) - z['c'])) < z['r'] - 1e-9


def seg_len(A, B):
    return float(np.linalg.norm(np.asarray(B, float) - np.asarray(A, float)))


def flight_min(L_units):
    return L_units * TAU_MIN


def path_len(pts):
    pts = [np.asarray(q, float) for q in pts]
    return sum(float(np.linalg.norm(u - v)) for u, v in zip(pts[:-1], pts[1:]))


def containing_zones(P, zs):
    return [z for z in zs if pt_inside(P, z)]


def _exit_time(P, zones):
    """点 P 到所有包含它的禁飞圆边界外的最小飞行时间(按最远边界,含外推量)。"""
    cz = containing_zones(P, zones)
    if not cz:
        return 0.0
    return max(z['r'] - float(np.linalg.norm(np.asarray(P, float) - z['c'])) for z in cz) \
        * TAU_MIN + 0.05 * TAU_MIN


def leg_forbidden_intervals(u, v, z, tau, zones):
    """航段 u→v 关于区 z 的禁止出发区间,含三类:
    1) 飞行穿越:s ∈ (a−λ_out·τ, b−λ_in·τ);
    2) 到达服务:v 在圆内,服务与离场须在激活前完成,即 s ∈ (a−5−t_exit−τ, b−τ);
    3) 离场时限:u 在圆内,须在激活前退出边界,即 s ∈ (a−t_exit, b)。"""
    ivs = []
    r = seg_zone_params(u, v, z)
    if r['lam_clip'] is not None:
        lo, hi = r['lam_clip']
        ivs.append((z['t0'] - hi * tau, z['t1'] - lo * tau))
    if pt_inside(v, z):
        t_exit = _exit_time(v, zones)
        ivs.append((z['t0'] - SERVICE_MIN - t_exit - MARGIN - tau, z['t1'] - tau))
    if pt_inside(u, z):
        t_exit = _exit_time(u, zones)
        ivs.append((z['t0'] - t_exit - MARGIN, z['t1']))
    return ivs


def merge_intervals(ivs, eps=1e-6):
    """区间并集:排序后合并重叠项。"""
    ivs = sorted((l, u) for l, u in ivs if u > l)
    out = []
    for l, u in ivs:
        if out and l <= out[-1][1] + eps:
            out[-1] = (out[-1][0], max(out[-1][1], u))
        else:
            out.append((l, u))
    return out


def build_forbidden_table(points, zones):
    """预计算全部 (i,j) 航段的合并禁止出发区间 F_ij。
    points 第 0 行为基地,其余为巡检点坐标;返回 {(i,j): [(L,U),...]}。"""
    n = len(points)
    table = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tau = flight_min(seg_len(points[i], points[j]))
            ivs = []
            for z in zones:
                ivs += leg_forbidden_intervals(points[i], points[j], z, tau, zones)
            F = merge_intervals(ivs)
            if F:
                table[(i, j)] = F
    return table


def next_legal_departure(s, F):
    """最早合法出发时刻 s'=inf{u≥s: u∉F};越过区间右端时加安全裕度。"""
    for (l, u) in F:
        if s < l:
            return s
        if s < u:
            s = u + MARGIN
    return s


# ---------------- 绕行几何:切线 + 圆弧 + 切线 ----------------
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
    """圆弧离散与弧长。离散半径向外偏移 R=r/cos(θ/2)+0.05,
    使折线严格位于圆外,便于 min-distance 类校验。"""
    c, r = z['c'], z['r']
    a0 = np.arctan2(*(Ta - c)[::-1])
    a1 = np.arctan2(*(Tb - c)[::-1])
    d = ((a1 - a0) if ccw else (a0 - a1)) % (2 * np.pi)
    if d < 1e-9:
        return [Ta, Tb], 0.0
    n = max(2, int(np.ceil(d / (np.pi / 12))))
    theta = d / n
    R = r / np.cos(theta / 2.0) + 0.05
    ang = a0 + d * np.linspace(0, 1, n + 1) if ccw else a0 - d * np.linspace(0, 1, n + 1)
    pts = [c + R * np.array([np.cos(a), np.sin(a)]) for a in ang]
    return pts, d * R


def detour_path(A, B, z):
    """单圆最短绕行折线(两切线×两切线×两弧向共 8 候选取最短)。端点须在圆外。"""
    a, b = np.asarray(A, float), np.asarray(B, float)
    if pt_inside(a, z) or pt_inside(b, z):
        return None
    Ta, Tb = tangents(a, z), tangents(b, z)
    if Ta is None or Tb is None:
        return None
    best, best_len = None, np.inf
    for ta in Ta:
        for tb in Tb:
            for ccw in (True, False):
                arc, alen = arc_pts(z, ta, tb, ccw)
                cand = [a] + [ta] + arc[1:-1] + [tb] + [b]
                L = seg_len(a, ta) + alen + seg_len(tb, b)
                if L < best_len:
                    best, best_len = cand, L
    return best


def poly_legal_at(pts, times, zs):
    """折线逐段(按各自飞行时段)解析验证全部禁飞区。返回 (legal, viol)。"""
    viol = []
    for (u, v), (td, ta) in zip(zip(pts[:-1], pts[1:]), times):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        for z in zs:
            c_ = leg_zone_conflict(u, v, z, td, ta - td)
            if c_ is not None:
                viol.append(dict(zone=z['zone_id'], t=(round(c_[0], 2), round(c_[1], 2))))
    return not viol, viol


# ---------------- 等待与圆外悬停 ----------------
def plan_hop(u, v, s, s1, zones):
    """等待期间起点 u 位于某生效区内时,转移至圆外悬停点 w 再等待。
    只考虑窗口与等待区间重叠的包含区;候选 w 为沿航向离场边界点或最近边界点,
    均沿径向外推至所有包含区之外。返回 (path=[u,w,v], hover_wait) 或 None。"""
    cz = [z for z in containing_zones(u, zones) if overlap(s, s1, z['t0'], z['t1'])]
    if not cz:
        return None
    a_min = min(z['t0'] for z in cz)
    u = np.asarray(u, float)
    v = np.asarray(v, float)

    def outside_all(w, step_dir):
        for _ in range(100):
            if not any(pt_inside(w, z) for z in cz):
                return w
            w = np.asarray(w) + 5.0 * step_dir
        return None

    cands = []
    lam_max, z_far = -1.0, None
    for z in cz:
        r = seg_zone_params(u, v, z)
        if r['lam_clip'] is not None and r['lam_clip'][1] > lam_max:
            lam_max, z_far = r['lam_clip'][1], z
    if z_far is not None:
        w1 = u + lam_max * (v - u)
        d1 = np.asarray(w1) - z_far['c']
        d1 = d1 / max(float(np.linalg.norm(d1)), 1e-9)
        w1 = outside_all(w1, d1)
        if w1 is not None:
            cands.append(w1)
    z_near = min(cz, key=lambda z: z['r'] - float(np.linalg.norm(u - z['c'])))
    d2 = u - z_near['c']
    d2 = d2 / max(float(np.linalg.norm(d2)), 1e-9)
    w2 = z_near['c'] + (z_near['r'] + 0.05) * d2
    w2 = outside_all(w2, d2)
    if w2 is not None:
        cands.append(w2)
    for w in cands:
        t_hop = flight_min(seg_len(u, w))
        if s + t_hop <= a_min + 1e-6:
            hover = max(0.0, s1 - s - t_hop)
            return [u, w, v], hover
    return None


def _hop_plan(u, v, s, zones, F, tau):
    """等待候选:s'=最早合法出发时刻;等待位置非法时经 plan_hop 圆外悬停。
    以实际路径校验飞行冲突与到达服务约束,违例则延长悬停。"""
    s1 = next_legal_departure(s, F)
    if s1 <= s + 1e-12:
        return None
    if not any(overlap(s, s1, z['t0'], z['t1']) for z in containing_zones(u, zones)):
        return dict(mode='wait', wait=s1 - s, path=[np.asarray(u, float), np.asarray(v, float)],
                    t_new=float(s))
    hp = plan_hop(u, v, s, s1, zones)
    if hp is None:
        return None
    path, hover = hp
    t_hop = flight_min(seg_len(path[0], path[1]))
    tau_wv = flight_min(seg_len(path[1], path[2]))
    t_arr = s + t_hop + hover + tau_wv
    for z in containing_zones(v, zones):
        t_exit = _exit_time(v, zones)
        if z['t0'] - SERVICE_MIN - t_exit < t_arr < z['t1']:
            hover = max(hover, z['t1'] + MARGIN - s - t_hop - tau_wv)
            t_arr = s + t_hop + hover + tau_wv
    for _ in range(12):
        times = [(s, s + t_hop), (s + t_hop + hover, t_arr)]
        ok, viol = poly_legal_at(path, times, zones)
        if ok:
            break
        stuck = False
        for x in viol:
            if x['t'][0] < s + t_hop - 1e-9:
                stuck = True
                break
            zz = next((z for z in zones if z['zone_id'] == x['zone']), None)
            if zz is None:
                continue
            r = seg_zone_params(path[1], path[2], zz)
            if r['lam_clip'] is None:
                continue
            hover = max(hover, zz['t1'] + MARGIN - s - t_hop - r['lam_clip'][0] * tau_wv)
        if stuck:
            return None
        t_arr = s + t_hop + hover + tau_wv
    return dict(mode='wait', wait=hover, path=path, t_new=float(s))


# ---------------- 动态航段代价 ----------------
def tau_detour(u, v, zs, t_dep):
    """绕行候选:对每个冲突区构造绕行折线,按实际路径逐段验证。"""
    tau = flight_min(seg_len(u, v))
    conf = conflicting_zones(u, v, zs, t_dep, tau)
    if not conf:
        return None
    best = None
    for z in conf:
        pts = detour_path(u, v, z)
        if pts is None:
            continue
        t, times = float(t_dep), []
        for a, b in zip(pts[:-1], pts[1:]):
            dt = flight_min(seg_len(a, b))
            times.append((t, t + dt))
            t += dt
        ok, _ = poly_legal_at(pts, times, zs)
        if not ok:
            continue
        bad = False
        for zz in zs:
            if pt_inside(v, zz):
                t_exit = _exit_time(v, zs)
                if zz['t0'] - SERVICE_MIN - t_exit < t < zz['t1']:
                    bad = True
                    break
        if bad:
            continue
        if best is None or t - t_dep < best[0]:
            best = (t - t_dep, pts)
    return best


def arc_cost(u, v, s, zones, F):
    """动态航段代价 τ̃_ij(s) = min{直飞, 等待, 绕行};违反硬约束的候选为 +∞。
    返回 dict(mode, cost, path, wait, t_new)。"""
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    tau = flight_min(seg_len(u, v))
    s1 = next_legal_departure(s, F)
    if s1 <= s + 1e-12:
        return dict(mode='direct', cost=tau, path=[u, v], wait=0.0, t_new=float(s))
    cands = []
    wp = _hop_plan(u, v, s, zones, F, tau)
    if wp is not None:
        cands.append((wp['wait'] + flight_min(path_len(wp['path'])), 'wait', wp['wait'],
                      wp['path'], s))
    dt = tau_detour(u, v, zones, s)
    if dt is not None:
        cands.append((dt[0], 'detour', 0.0, dt[1], s))
    if not cands:
        return dict(mode='fail', cost=None, path=[u, v], wait=0.0, t_new=float(s))
    cost, mode, w, path, t0 = min(cands, key=lambda c: c[0])
    return dict(mode=mode, cost=cost, path=path, wait=w, t_new=float(t0) + w)


# ---------------- 单机路线时间递推 ----------------
def _process_leg(u, v, t, zones, Fk, k, waits):
    """处理一条航段,返回 (ac, depart);无法合法执行时返回 (None, None)。
    等待事件以 (k, t_s, wt, kind, pos) 记入 waits,pos=0 原地、1 圆外悬停点。"""
    ac = arc_cost(u, v, t, zones, Fk)
    if ac['mode'] == 'fail':
        s1 = next_legal_departure(t, Fk)
        if s1 <= t + 1e-9:
            s1 = max((z['t1'] for z in conflicting_zones(
                u, v, zones, t, flight_min(seg_len(u, v)))), default=t) + MARGIN
        if not any(overlap(t, s1, z['t0'], z['t1'])
                   for z in containing_zones(u, zones)):
            path, hs, hw = [u, v], t, s1 - t
        else:
            hp = plan_hop(u, v, t, s1, zones)
            if hp is None:
                return None, None
            path, hw = hp
            t_hop = flight_min(seg_len(path[0], path[1]))
            hs = t + t_hop
        ac = dict(mode='wait', path=path, wait=hw, hover_start=hs)
        if hw > 1e-6:
            waits.append((k, round(hs, 2), round(hw, 2), 'leg-fallback',
                          0 if len(path) == 2 else 1))
    wait_w = ac.get('wait', 0.0)
    hop = len(ac['path']) > 2
    depart = t + (wait_w if not hop else 0.0)
    if ac['mode'] == 'wait' and wait_w > 1e-6 and 'hover_start' not in ac:
        if hop:
            t_hop = flight_min(seg_len(ac['path'][0], ac['path'][1]))
            waits.append((k, round(depart + t_hop, 2), round(wait_w, 2), 'leg', 1))
        else:
            waits.append((k, round(depart - wait_w, 2), round(wait_w, 2), 'leg', 0))
    return ac, depart


def build_route(tasks, xy, zones, F, veh=1):
    """按 τ̃ 递推单机调度的全部时刻。tasks 为 [(point_id, visit_no), ...];
    返回 route 字典,或 None 表示该路线不存在合法调度。"""
    pids = [t[0] for t in tasks]
    pts = [np.array([0.0, 0.0])] + [xy[p - 1].astype(float) for p in pids]
    t = 0.0
    legs, waits, arr, dep, leg_dep = [], [], [], [], []
    for k in range(1, len(pts)):
        u, v = pts[k - 1], pts[k]
        pid_u = pids[k - 2] if k >= 2 else 0
        Fk = F.get((pid_u, pids[k - 1]), [])
        ac, depart = _process_leg(u, v, t, zones, Fk, k, waits)
        if ac is None:
            return None
        wait_w = ac.get('wait', 0.0)
        hop = len(ac['path']) > 2
        leg_dep.append(round(depart, 2))
        t = depart + (wait_w if hop else 0.0) + flight_min(path_len(ac['path']))
        arr.append(round(t, 2))
        for z in zones:
            if pt_inside(v, z) and overlap(t, t + SERVICE_MIN, z['t0'], z['t1']):
                return None
        dep.append(round(t + SERVICE_MIN, 2))
        t += SERVICE_MIN
        legs.append([list(map(float, q)) for q in ac['path']])
    u, v = pts[-1], pts[0]
    Fk = F.get((pids[-1], 0), [])
    k_ret = len(pids) + 1
    ac, depart = _process_leg(u, v, t, zones, Fk, k_ret, waits)
    if ac is None:
        return None
    wait_w = ac.get('wait', 0.0)
    hop = len(ac['path']) > 2
    return_dep = round(depart, 2)
    t = depart + (wait_w if hop else 0.0) + flight_min(path_len(ac['path']))
    legs.append([list(map(float, q)) for q in ac['path']])
    end_min = t
    legs_u = [round(path_len(L), 3) for L in legs]
    seq = [dict(point_id=int(t_[0]), visit_no=int(t_[1])) for t_ in tasks]
    return dict(veh=veh, seq=seq, arr_min=arr, dep_min=dep,
                leg_dep_min=leg_dep, return_dep_min=return_dep,
                legs_u=legs_u, legs_wp=legs,
                travel_units=round(sum(legs_u), 2),
                end_s=int(round(end_min * 60)),
                busy_h=round(end_min / 60, 4), n_tasks=len(pids),
                waits=waits)


def leg_segment_times(leg, td, leg_waits):
    """折线各子段的真实飞行时段。pos≥1 的悬停插在对应航路点之后,
    pos=0 的原地等待已含于 td,pos=-1 的服务等待在到达之后。"""
    lens = [seg_len(u, v) for u, v in zip(leg[:-1], leg[1:])]
    cum = float(td)
    times = []
    for i, L in enumerate(lens):
        if i >= 1:
            cum += sum(wt for (t_s, wt, pos) in leg_waits if pos == i)
        times.append((cum, cum + flight_min(L)))
        cum += flight_min(L)
    return times


# ---------------- 独立验证:飞行 + 服务 + 等待 ----------------
def validate_schedule(routes, zs):
    """对完整调度逐段解析验证三类状态,返回违例列表(空为通过)。"""
    viol = []
    for r in routes:
        n = len(r['seq'])
        for k in range(n):
            leg = r['legs_wp'][k]
            td = r['leg_dep_min'][k]
            leg_waits = [(t_s, wt, pos) for (k_, t_s, wt, kind, pos) in r.get('waits', [])
                         if k_ == k + 1]
            ok, v = poly_legal_at(leg, leg_segment_times(leg, td, leg_waits), zs)
            for x in v:
                viol.append(dict(veh=r['veh'], kind='leg', k=k + 1,
                                 zone=x['zone'], t=x['t']))
            vpt = leg[-1]
            for z in zs:
                if pt_inside(vpt, z) and overlap(r['dep_min'][k] - SERVICE_MIN,
                                                 r['dep_min'][k], z['t0'], z['t1']):
                    viol.append(dict(veh=r['veh'], kind='service', k=k + 1,
                                     zone=z['zone_id'],
                                     t=(round(r['dep_min'][k] - SERVICE_MIN, 2),
                                        round(r['dep_min'][k], 2))))
            for (k_, t_s, wt, kind, pos) in r.get('waits', []):
                if k_ != k + 1:
                    continue
                wp = leg[pos]
                for z in zs:
                    if pt_inside(wp, z) and overlap(t_s, t_s + wt, z['t0'], z['t1']):
                        viol.append(dict(veh=r['veh'], kind='wait', k=k + 1,
                                         zone=z['zone_id'],
                                         t=(round(t_s, 2), round(t_s + wt, 2))))
        td = r.get('return_dep_min', r['dep_min'][-1])
        leg_waits = [(t_s, wt, pos) for (k_, t_s, wt, kind, pos) in r.get('waits', [])
                     if k_ == n + 1]
        ok, v = poly_legal_at(r['legs_wp'][-1], leg_segment_times(r['legs_wp'][-1], td,
                                                                  leg_waits), zs)
        for x in v:
            viol.append(dict(veh=r['veh'], kind='leg', k='return',
                             zone=x['zone'], t=x['t']))
        for (k_, t_s, wt, kind, pos) in r.get('waits', []):
            if k_ != n + 1:
                continue
            wp = r['legs_wp'][-1][pos]
            for z in zs:
                if pt_inside(wp, z) and overlap(t_s, t_s + wt, z['t0'], z['t1']):
                    viol.append(dict(veh=r['veh'], kind='wait', k='return',
                                     zone=z['zone_id'],
                                     t=(round(t_s, 2), round(t_s + wt, 2))))
    return viol


if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    z = dict(zone_id='T', c=np.array([5.0, 0.0]), r=2.0, t0=0.0, t1=100.0)
    r = seg_zone_params([0, 0], [10, 0], z)
    print('穿越参数:', [round(x, 4) for x in r['lam']])
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    F = build_forbidden_table(pts, [z])
    print('禁止出发区间:', [(round(l, 2), round(u, 2)) for l, u in F[(0, 1)]])
    ac = arc_cost(np.array([0.0, 0.0]), np.array([10.0, 0.0]), 0.0, [z], F[(0, 1)])
    print('动态代价:', ac['mode'], round(ac['cost'], 2), 'min')
