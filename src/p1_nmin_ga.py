# -*- coding: utf-8 -*-
"""问题1 第一部分:Nmin 的遗传算法求解(4 算例并行)。

流程(与"问题一第一部分.pdf"模型 min N s.t. max_k T_k ≤ 9 h 对应):
1. 等级展开:点 i 的巡检次数 k_i 展开为 k_i 个任务(同坐标),总任务数 M=Σk_i;
2. 理论下界 N_LB(公式同 src/p1_lb.py:服务总时间 + 飞行时间有效下界 LB1~LB5);
3. 自 N_LB 起对每个 N 用 GA(巨型路线+最优分割+局部搜索)求 min-makespan 调度,
   首次出现 best ≤ 540 min 的 N 即 Nmin(构造可行性证明);
   对 Nmin-1 记录 GA 最优 makespan 作为不可行性的经验证据(论文中如实声明);
4. 在 Nmin 上做加长抛光,取 Tmax 最小者为最终调度;
5. 独立校验(任务覆盖恰一次、逐机时间按坐标重算、Tmax≤540)后导出。

输出:A 题/output/ga_p1_Case*.json、ga_p1_Case*.log、figs/ga_p1_Case*_路径图.png、
     ga_p1_nmin.xlsx(表2 格式)。随机种子固定,全程可复现。
"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import (A_DIR, HORIZON_MIN, SERVICE_MIN, TAU_MIN, load_points,
                    travel_min)
from ga_minmax import GA, validate_schedule

OUT = os.path.join(A_DIR, 'output')
FIG = os.path.join(OUT, 'figs')
os.makedirs(FIG, exist_ok=True)

# ---------------- 参数 ----------------
POP_SIZE = 80
MAX_GENS = 1000
PATIENCE = 250
RESTART_EVERY = 120
N_SEEDS_EVIDENCE = 3      # 不可行证据 N 的种子数
N_SEEDS_CANDIDATE = 6     # 候选可行 N 的种子数
POLISH_GENS = 1500
POLISH_SEEDS = 2
HORIZON = HORIZON_MIN     # 540 min


def expand_tasks(p):
    """点 -> 任务列表 [(点下标, 第几次巡检), ...],共 M 项。"""
    tasks = []
    for i in range(len(p['ids'])):
        for k in range(int(p['visits'][i])):
            tasks.append((i, k + 1))
    return tasks


def lower_bound_N(p):
    """理论下界 N_LB:与 src/p1_lb.py 相同公式(独立复现)。"""
    xy, visits = p['xy'], p['visits']
    d = np.hypot(xy[:, 0], xy[:, 1])
    n = len(d)
    M = int(visits.sum())
    t_srv = SERVICE_MIN * M
    order = np.argsort(-d)
    ds, ps = d[order], xy[order]
    w = ds[0] + ds[1] + float(np.linalg.norm(ps[0] - ps[1]))
    lb1 = 2 * ds[0]
    lb2 = 2 * d.sum() / n

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
            v.append(2 * (ds[0] + d[np.argsort(d)[:N - 1]].sum()))
        return t_srv + travel_min(max(v))

    return next(N for N in range(1, 80) if lb(N) <= HORIZON * N)


def run_seed(tasks, xy, N, seed, max_gens, patience, stop_if_le, pop_size):
    ga = GA(tasks, xy, N=N, seed=seed, pop_size=pop_size)
    res = ga.run(max_gens=max_gens, patience=patience, stop_if_le=stop_if_le)
    return ga, res


def schedule_times(routes, D_ext, M):
    """按路线顺序计算各任务到达/离开时刻(8:00 为 0)。返回逐机明细。"""
    out = []
    for rid, r in enumerate(routes):
        t = 0.0
        prev = M
        tasks_t = []
        for x in r:
            t += TAU_MIN * D_ext[prev, x]
            arr = t
            t += SERVICE_MIN
            tasks_t.append((int(x), float(arr), float(arr) + SERVICE_MIN))
            prev = x
        if r:
            t += TAU_MIN * D_ext[prev, M]
            ra = np.asarray(r, dtype=np.int64)
            dist_u = float(D_ext[M, r[0]] + D_ext[r[-1], M] +
                           (D_ext[ra[:-1], ra[1:]].sum() if len(r) > 1 else 0.0))
        else:
            dist_u = 0.0
        out.append(dict(drone=rid + 1, total_min=float(t), dist_u=dist_u,
                        n_tasks=len(r), tasks=tasks_t))
    return out


def export_case(case, p, tasks, n_lb, search, nmin, res, wall):
    """导出 JSON 与路径图。res 为最终调度的 (ga, result) 元组。"""
    ga, r = res
    M = len(tasks)
    rep = validate_schedule(M, r['routes'], ga.D_ext, horizon=HORIZON)
    det = schedule_times(r['routes'], ga.D_ext, M)
    sched = dict(
        Tmax_min=rep['Tmax_min'], Tmax_h=rep['Tmax_min'] / 60,
        Tmin_min=rep['Tmin_min'], Tmin_h=rep['Tmin_min'] / 60,
        validated=dict(coverage_ok=rep['coverage_ok'], horizon_ok=rep['horizon_ok'],
                       route_times=rep['route_times']),
        seed=r['seed'], gens=r['gens'], best_fit_min=r['best_fit'],
        routes=[dict(
            drone=d['drone'], total_min=d['total_min'], dist_u=d['dist_u'],
            n_tasks=d['n_tasks'],
            tasks=[dict(task_id=t[0], point_id=int(p['ids'][tasks[t[0]][0]]),
                        point_idx=int(tasks[t[0]][0]), visit=int(tasks[t[0]][1]),
                        x=float(p['xy'][tasks[t[0]][0], 0]),
                        y=float(p['xy'][tasks[t[0]][0], 1]),
                        arr_min=t[1], dep_min=t[2]) for t in d['tasks']])
            for d in det])
    out = dict(case=case, n=int(len(p['ids'])), M=M, N_LB=n_lb, Nmin=nmin,
               wall_s=round(wall, 1),
               params=dict(pop_size=POP_SIZE, max_gens=MAX_GENS, patience=PATIENCE,
                           restart_every=RESTART_EVERY, polish_gens=POLISH_GENS,
                           horizon_min=HORIZON),
               search=search, schedule=sched)
    with open(os.path.join(OUT, 'ga_p1_%s.json' % case), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    plot_case(case, p, tasks, r['routes'], rep['Tmax_min'])
    return out


def plot_case(case, p, tasks, routes, tmax, fname=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    xy = p['xy']
    ax.scatter(xy[:, 0], xy[:, 1], c='0.75', s=6, label='巡检点 (n=%d)' % len(xy))
    ax.scatter([0], [0], c='k', marker='s', s=70, zorder=5, label='基地 (0,0)')
    colors = plt.cm.tab10.colors
    for rid, r in enumerate(routes):
        pts = np.vstack([[0, 0], xy[[tasks[t][0] for t in r]], [0, 0]])
        ax.plot(pts[:, 0], pts[:, 1], '-', color=colors[rid % 10], lw=1.3,
                alpha=0.9, label='无人机 %d' % (rid + 1))
    ax.set_title('%s: N=%d, Tmax=%.2f h' % (case, len(routes), tmax / 60))
    ax.set_xlabel('X (1 u = 100 m)')
    ax.set_ylabel('Y (1 u = 100 m)')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname or 'ga_p1_%s_路径图.png' % case), dpi=130)
    plt.close(fig)


def solve_case(args):
    case, ci = args
    logf = open(os.path.join(OUT, 'ga_p1_%s.log' % case), 'w', encoding='utf-8')

    def log(*a):
        print(*a, file=logf, flush=True)

    t0 = time.time()
    p = load_points(case)
    tasks = expand_tasks(p)
    M = len(tasks)
    n_lb = lower_bound_N(p)
    log('=== %s: n=%d M=%d N_LB=%d ===' % (case, len(p['ids']), M, n_lb))

    search = []
    nmin, final_res = None, None
    for N in range(n_lb, n_lb + 8):
        n_seeds = N_SEEDS_EVIDENCE if N < n_lb + 2 else N_SEEDS_CANDIDATE
        seeds = [ci * 1000 + N * 10 + s for s in range(n_seeds)]
        best_for_N, feasible = None, False
        for s in seeds:
            ga, r = run_seed(tasks, p['xy'], N, s, MAX_GENS, PATIENCE,
                             stop_if_le=HORIZON, pop_size=POP_SIZE)
            if best_for_N is None or r['mk'] < best_for_N['mk'] - 1e-9:
                best_for_N = dict(mk=r['mk'], seed=s, gens=r['gens'])
                best_ga, best_r = ga, r
            log('  N=%d seed=%d: mk=%.2f min (%.3f h) gens=%d'
                % (N, s, r['mk'], r['mk'] / 60, r['gens']))
            if r['mk'] <= HORIZON + 1e-9:
                feasible = True
                break
        search.append(dict(N=N, best_min=round(best_for_N['mk'], 2),
                           best_h=round(best_for_N['mk'] / 60, 3),
                           seed=best_for_N['seed'], feasible=feasible))
        log('  => N=%d best=%.2f min (%.3f h) feasible=%s'
            % (N, best_for_N['mk'], best_for_N['mk'] / 60, feasible))
        if feasible:
            nmin = N
            final_res = (best_ga, best_r)
            break

    # 抛光:Nmin 上加长运行取更优调度
    if nmin is not None:
        for s in range(POLISH_SEEDS):
            ga, r = run_seed(tasks, p['xy'], nmin, ci * 9000 + s, POLISH_GENS,
                             PATIENCE, stop_if_le=None, pop_size=POP_SIZE)
            log('  polish seed=%d: mk=%.2f min (%.3f h) gens=%d'
                % (s, r['mk'], r['mk'] / 60, r['gens']))
            if r['mk'] < final_res[1]['mk'] - 1e-9:
                final_res = (ga, r)

    wall = time.time() - t0
    rep = validate_schedule(M, final_res[1]['routes'], final_res[0].D_ext, horizon=HORIZON)
    log('=== %s 完成: Nmin=%s, Tmax=%.2f min (%.3f h), Tmin=%.2f min (%.3f h), '
        'coverage_ok=%s, horizon_ok=%s, wall=%.1f s'
        % (case, nmin, rep['Tmax_min'], rep['Tmax_min'] / 60,
           rep['Tmin_min'], rep['Tmin_min'] / 60, rep['coverage_ok'],
           rep['horizon_ok'], wall))
    logf.close()
    out = export_case(case, p, tasks, n_lb, search, nmin, final_res, wall)
    return out


def main():
    cases = ['Case1', 'Case2', 'Case3', 'Case4']
    mp.set_start_method('spawn', force=True)
    with mp.Pool(len(cases)) as pool:
        outs = pool.map(solve_case, [(c, i + 1) for i, c in enumerate(cases)])

    print('\n===== 问题一第一部分 GA 结果汇总 (9h 约束) =====')
    print('%-8s %6s %8s %8s %8s' % ('算例', 'Nmin', 'Tmax(h)', 'Tmin(h)', 'N_LB'))
    for o in outs:
        s = o['schedule']
        print('%-8s %6d %8.3f %8.3f %8d'
              % (o['case'], o['Nmin'], s['Tmax_h'], s['Tmin_h'], o['N_LB']))

    # 导出表2 格式 xlsx
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表2_GA'
    ws.append(['测试算例', '无人机数量N', '单架无人机最长工作时间Tmax(h)',
               '单架无人机最短工作时间Tmin(h)'])
    for o in outs:
        s = o['schedule']
        ws.append([o['case'], o['Nmin'], round(s['Tmax_h'], 4), round(s['Tmin_h'], 4)])
    ws2 = wb.create_sheet('逐N搜索')
    ws2.append(['测试算例', 'N', 'GA最优Tmax(h)', '是否≤9h'])
    for o in outs:
        for row in o['search']:
            ws2.append([o['case'], row['N'], round(row['best_h'], 4),
                        '是' if row['feasible'] else '否'])
    wb.save(os.path.join(OUT, 'ga_p1_nmin.xlsx'))
    print('\n已导出: A 题/output/ga_p1_nmin.xlsx 与各算例 ga_p1_*.json / 路径图')


if __name__ == '__main__':
    main()
