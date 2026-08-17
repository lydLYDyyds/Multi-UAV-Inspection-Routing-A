# -*- coding: utf-8 -*-
"""问题3 全局优化(ALNS):破坏-修复 + 局部搜索 + 模拟退火接受。

所有候选方案用 nofly.build_route(动态航段代价 + 三态硬约束)真实评估,
不可行方案直接排除;目标 (Tmax, δ) 按词典序比较。
破坏算子:最长路线任务 / 禁飞敏感航段邻近任务 / 随机任务 / 空间相近任务;
修复:对每个被删任务枚举全部插入位置,取词典序最优合法插入;
局部搜索:relocate / swap / 2-opt(段反转) / cross-exchange(跨机换尾);
接受:词典序改进必收,否则按模拟退火概率接受。
输出 output/p3alns_{case}.json。

用法:python src/alns3.py Case1   环境变量 ALNS_ITERS / ALNS_SEED
"""
import sys, io, os, json, random
import numpy as np
from common import load_points, load_zones, A_DIR
import nofly

CASE = sys.argv[1] if len(sys.argv) > 1 else 'Case1'
ITERS = int(os.environ.get('ALNS_ITERS', '300'))
SEED = int(os.environ.get('ALNS_SEED', '1'))
R_ATTEMPTS = 60                      # 每个局部搜索算子的随机尝试次数
NO_IMPROVE_LIMIT = 80
T0 = 0.05                            # 模拟退火初始温度(h)
COOL = 0.98
SENS_KINDS = ('leg', 'leg-fallback')


def end_of(tasks, xy, zones, F, veh):
    rr = nofly.build_route(tasks, xy, zones, F, veh=veh)
    return rr


def full_eval(sol, xy, zones, F):
    """全方案真实评估;返回 (routes, ends_h) 或 None。"""
    routes, ends = [], []
    for v, tasks in enumerate(sol):
        rr = end_of(tasks, xy, zones, F, v + 1)
        if rr is None:
            return None
        routes.append(rr)
        ends.append(rr['end_s'] / 3600.0)
    return routes, ends


def key_of(ends):
    return (max(ends), max(ends) - min(ends))


# ---------------- 破坏算子 ----------------
def destroy(sol, xy, zones, F, rng):
    d = rng.randint(4, 8)
    flat = [(v, i, t) for v, rt in enumerate(sol) for i, t in enumerate(rt)]
    if not flat:
        return []
    op = rng.random()
    picks = []
    if op < 0.25:                                    # ①最长路线任务
        _, ends = full_eval(sol, xy, zones, F)
        v = int(np.argmax(ends))
        idx = rng.sample(range(len(sol[v])), min(d, len(sol[v])))
        picks = [(v, i, sol[v][i]) for i in idx]
    elif op < 0.5:                                   # ②禁飞敏感航段附近任务
        sens = set()
        for v, rt in enumerate(sol):
            rr = end_of(rt, xy, zones, F, v + 1)
            if rr is None:
                continue
            for k in range(1, len(rt) + 1):
                if any(w[0] == k and w[3] in SENS_KINDS for w in rr['waits']) \
                        or (k - 1 < len(rr['legs_wp']) and len(rr['legs_wp'][k - 1]) > 2):
                    sens.add((v, k - 1))
        cand = [f for f in flat if (f[0], f[1]) in sens]
        if not cand:
            cand = flat
        picks = rng.sample(cand, min(d, len(cand)))
    elif op < 0.75:                                  # ③随机任务
        picks = rng.sample(flat, min(d, len(flat)))
    else:                                            # ④空间相近任务
        seed = rng.choice(flat)
        c = xy[seed[2][0] - 1]
        cand = sorted(flat, key=lambda f: float(np.linalg.norm(xy[f[2][0] - 1] - c)))
        picks = cand[: min(d, len(cand))]
    return picks


# ---------------- 修复(最优插入) ----------------
def repair(sol, picks, ends_base, xy, zones, F, rng):
    """从 sol 中摘除 picks 并逐个最优插入;返回 (new_sol, ends_new) 或 None。"""
    new_sol = [list(rt) for rt in sol]
    removed = []
    for (v, i, t) in sorted(picks, key=lambda p: -p[1]):
        removed.append(new_sol[v].pop(i))
    rng.shuffle(removed)
    for task in removed:
        best_key, best_v, best_pos = None, None, None
        for v in range(len(new_sol)):
            route = new_sol[v]
            others = [e for (vv, e) in enumerate(ends_base) if vv != v]
            for pos in range(len(route) + 1):
                rr = end_of(route[:pos] + [task] + route[pos:], xy, zones, F, v + 1)
                if rr is None:
                    continue
                new_end = rr['end_s'] / 3600.0
                Tmax = max([new_end] + others)
                Tmin = min([new_end] + others) if others else new_end
                k = (Tmax, Tmax - Tmin)
                if best_key is None or k < best_key:
                    best_key, best_v, best_pos = k, v, pos
        if best_v is None:                           # 无处可插 → 随机车尾兜底
            best_v = rng.randrange(len(new_sol))
            best_pos = len(new_sol[best_v])
        new_sol[best_v].insert(best_pos, task)
        ends_base[best_v] = end_of(new_sol[best_v], xy, zones, F, best_v + 1)['end_s'] / 3600.0
    return new_sol


# ---------------- 局部搜索 ----------------
def try_moves(sol, xy, zones, F, rng):
    routes, ends = full_eval(sol, xy, zones, F)
    if routes is None:
        return sol
    cur_key = key_of(ends)
    N = len(sol)
    # relocate
    for _ in range(R_ATTEMPTS):
        v = rng.randrange(N)
        if not sol[v]:
            continue
        i = rng.randrange(len(sol[v]))
        t = sol[v][i]
        v2 = rng.randrange(N)
        pos = rng.randrange(len(sol[v2]) + 1) if v2 != v else rng.randrange(len(sol[v2]) + 1)
        if v2 == v and (pos == i or pos == i + 1):
            continue
        cand = [list(rt) for rt in sol]
        cand[v].pop(i)
        cand[v2].insert(pos, t)
        res = full_eval(cand, xy, zones, F)
        if res is None:
            continue
        k = key_of(res[1])
        if k < cur_key:
            sol, cur_key = cand, k
            break
    # swap
    for _ in range(R_ATTEMPTS):
        a = [(v, i) for v in range(N) for i in range(len(sol[v]))]
        if len(a) < 2:
            break
        (v1, i1), (v2, i2) = rng.sample(a, 2)
        cand = [list(rt) for rt in sol]
        cand[v1][i1], cand[v2][i2] = cand[v2][i2], cand[v1][i1]
        res = full_eval(cand, xy, zones, F)
        if res is None:
            continue
        k = key_of(res[1])
        if k < cur_key:
            sol, cur_key = cand, k
            break
    # 2-opt(段反转,段长 2..6)
    for _ in range(R_ATTEMPTS):
        v = rng.randrange(N)
        if len(sol[v]) < 4:
            continue
        i = rng.randrange(len(sol[v]) - 1)
        j = rng.randrange(i + 2, min(len(sol[v]) + 1, i + 8))
        cand = [list(rt) for rt in sol]
        cand[v][i:j] = list(reversed(cand[v][i:j]))
        res = full_eval(cand, xy, zones, F)
        if res is None:
            continue
        k = key_of(res[1])
        if k < cur_key:
            sol, cur_key = cand, k
            break
    # cross-exchange(换尾)
    for _ in range(R_ATTEMPTS):
        if N < 2:
            break
        v1, v2 = rng.sample(range(N), 2)
        if not sol[v1] or not sol[v2]:
            continue
        i1 = rng.randrange(len(sol[v1]))
        i2 = rng.randrange(len(sol[v2]))
        cand = [list(rt) for rt in sol]
        cand[v1][i1:], cand[v2][i2:] = cand[v2][i2:], cand[v1][i1:]
        res = full_eval(cand, xy, zones, F)
        if res is None:
            continue
        k = key_of(res[1])
        if k < cur_key:
            sol, cur_key = cand, k
            break
    return sol


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    rng = random.Random(SEED)
    p = load_points(CASE)
    xy = p['xy']
    zones = load_zones(CASE)
    pts_all = np.vstack([np.zeros((1, 2)), xy])
    F = nofly.build_forbidden_table(pts_all, zones)
    with open(os.path.join(A_DIR, 'output', f'p3_{CASE}.json'), encoding='utf-8') as f:
        p3 = json.load(f)
    N = p3['summary']['N']
    sol = [[(s['point_id'], s['visit_no']) for s in r['seq']] for r in p3['routes']]
    while len(sol) < N:
        sol.append([])

    res0 = full_eval(sol, xy, zones, F)
    assert res0 is not None, '初解不可行'
    _, ends = res0
    cur_key = key_of(ends)
    best_sol, best_key = [list(rt) for rt in sol], cur_key
    T = T0
    no_imp = 0
    print(f'{CASE}: 初解 Tmax={cur_key[0]:.4f}h δ={cur_key[1]:.4f}h '
          f'ITERS={ITERS} SEED={SEED}', flush=True)
    for it in range(1, ITERS + 1):
        picks = destroy(sol, xy, zones, F, rng)
        cand = repair(sol, picks, ends, xy, zones, F, rng)
        cand = try_moves(cand, xy, zones, F, rng)
        res = full_eval(cand, xy, zones, F)
        if res is None:
            continue
        _, ends_c = res
        k = key_of(ends_c)
        accept = False
        if k < cur_key:
            accept = True
        elif k[0] == cur_key[0] and k[1] < cur_key[1]:
            accept = True
        elif rng.random() < np.exp(-(k[0] - cur_key[0]) * 60.0 / T):
            accept = True
        if accept:
            sol, ends, cur_key = cand, ends_c, k
            if k < best_key:
                best_sol, best_key = [list(rt) for rt in sol], k
                no_imp = 0
            else:
                no_imp += 1
        else:
            no_imp += 1
        T = max(T * COOL, 0.0005)
        if it % 25 == 0 or it == ITERS:
            print(f'  it={it:4d} cur=({cur_key[0]:.4f},{cur_key[1]:.4f}) '
                  f'best=({best_key[0]:.4f},{best_key[1]:.4f}) T={T:.4f}', flush=True)
        if no_imp >= NO_IMPROVE_LIMIT:
            print(f'  连续无改进 {no_imp} 次,提前终止', flush=True)
            break

    routes, ends_b = full_eval(best_sol, xy, zones, F)
    assert routes is not None
    viol = nofly.validate_schedule(routes, zones)
    assert not viol, f'最优解违例: {viol[:3]}'
    tmax_h = round(max(ends_b), 4)
    tmin_h = round(min(ends_b), 4)
    n_det = sum(1 for r in routes for L in r['legs_wp'] if len(L) > 2)
    n_wait = sum(1 for r in routes for w in r['waits'] if w[3].startswith('leg'))
    tot_wait = round(sum(w[2] for r in routes for w in r['waits']), 2)
    summary = dict(case=CASE, N=N, tmax_h=tmax_h, tmin_h=tmin_h,
                   delta_h=round(tmax_h - tmin_h, 4), source='alns',
                   violations=0, n_detour_legs=n_det, n_wait_events=n_wait,
                   total_wait_min=tot_wait, iters=ITERS, seed=SEED)
    out = dict(summary=summary, routes=routes)
    os.makedirs(os.path.join(A_DIR, 'output'), exist_ok=True)
    path = os.path.join(A_DIR, 'output', f'p3alns_{CASE}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'[ALNS {CASE}] Tmax={tmax_h}h Tmin={tmin_h}h δ={summary["delta_h"]}h '
          f'违例=0 绕行段={n_det} 等待={n_wait} -> {path}', flush=True)
    os_ = p3['summary']
    print(f'对比当前p3: Tmax {os_["tmax_h"]}h -> {tmax_h}h | '
          f'δ {os_["delta_h"]}h -> {summary["delta_h"]}h', flush=True)


if __name__ == '__main__':
    main()
