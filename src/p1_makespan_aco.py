# -*- coding: utf-8 -*-
"""问题1 第二层:蚁群算法在固定 N=Nmin 下最小化 makespan(4 算例并行)。

流程:读第二层 GA 解(ga_p1_layer2_*.json)的 Nmin 与路线作为信息素热启动 →
ACO(巨型路线+最优分割+deep_ls)多种子求解 → 独立校验 → 导出。

输出:A 题/output/ga_p1_aco_<Case>.json、ga_p1_aco_<Case>.log、
     figs/ga_p1_aco_<Case>_路径图.png、ga_p1_aco.xlsx(表2 + 三算法对比)。
"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import A_DIR, HORIZON_MIN, load_points
from aco_minmax import ACO
from ga_minmax import validate_schedule
from p1_nmin_ga import (OUT, FIG, expand_tasks, plot_case, schedule_times)

os.makedirs(FIG, exist_ok=True)

# 参数
N_ANTS = 30
ITERS = 400
N_SEEDS = 4
HORIZON = HORIZON_MIN

REF_TMAX_H = {'Case1': 7.883, 'Case2': 7.178, 'Case3': 8.373, 'Case4': 8.020}


def solve_case(args):
    case, ci = args
    logf = open(os.path.join(OUT, 'ga_p1_aco_%s.log' % case), 'w', encoding='utf-8')

    def log(*a):
        print(*a, file=logf, flush=True)

    t0 = time.time()
    p = load_points(case)
    tasks = expand_tasks(p)
    M = len(tasks)
    d2 = json.load(open(os.path.join(OUT, 'ga_p1_layer2_%s.json' % case), encoding='utf-8'))
    nmin = d2['Nmin']
    seed_routes = [[t['task_id'] for t in r['tasks']] for r in d2['schedule']['routes']]
    ga2_tmax = d2['layer2']['Tmax_h']
    log('=== %s: M=%d Nmin=%d, 簇编码GA Tmax=%.3f h(信息素热启动)==='
        % (case, M, nmin, ga2_tmax))

    best = None
    for s in range(N_SEEDS):
        aco = ACO(tasks, p['xy'], N=nmin, seed=ci * 5000 + s, n_ants=N_ANTS,
                  seed_routes=seed_routes)
        r = aco.run(iters=ITERS, log=log)
        log('  seed=%d: mk=%.2f min (%.3f h) tsum=%.2f min'
            % (s, r['mk'], r['mk'] / 60, r['tsum']))
        if best is None or r['mk'] < best[1]['mk'] - 1e-9:
            best = (aco, r)
    aco, r = best
    rep = validate_schedule(M, r['routes'], aco.D_ext, horizon=HORIZON)
    assert rep['coverage_ok'] and rep['horizon_ok'], 'ACO 解未通过校验'
    det = schedule_times(r['routes'], aco.D_ext, M)

    out = dict(case=case, n=int(len(p['ids'])), M=M, Nmin=nmin,
               wall_s=round(time.time() - t0, 1),
               params=dict(n_ants=N_ANTS, iters=ITERS, n_seeds=N_SEEDS,
                           alpha=1.0, beta=2.0, rho=0.1, q0=0.9, nn_size=25),
               cluster_ga=dict(Tmax_h=ga2_tmax),
               aco=dict(Tmax_min=rep['Tmax_min'], Tmax_h=rep['Tmax_min'] / 60,
                        Tmin_min=rep['Tmin_min'], Tmin_h=rep['Tmin_min'] / 60,
                        improvement_h=ga2_tmax - rep['Tmax_min'] / 60,
                        tsum_min=r['tsum'], seed=r['seed']),
               schedule=dict(Tmax_min=rep['Tmax_min'], Tmin_min=rep['Tmin_min'],
                             route_times=rep['route_times'],
                             routes=[dict(
                                 drone=d['drone'], total_min=d['total_min'],
                                 dist_u=d['dist_u'], n_tasks=d['n_tasks'],
                                 tasks=[dict(task_id=t[0],
                                             point_id=int(p['ids'][tasks[t[0]][0]]),
                                             point_idx=int(tasks[t[0]][0]),
                                             visit=int(tasks[t[0]][1]),
                                             x=float(p['xy'][tasks[t[0]][0], 0]),
                                             y=float(p['xy'][tasks[t[0]][0], 1]),
                                             arr_min=t[1], dep_min=t[2])
                                        for t in d['tasks']])
                                 for d in det]),
               validated=dict(coverage_ok=rep['coverage_ok'], horizon_ok=rep['horizon_ok']))
    with open(os.path.join(OUT, 'ga_p1_aco_%s.json' % case), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    plot_case(case, p, tasks, r['routes'], rep['Tmax_min'],
              fname='ga_p1_aco_%s_路径图.png' % case)
    log('=== %s ACO 完成: Tmax %.3f -> %.3f h, Tmin=%.3f h, wall=%.1f s ==='
        % (case, ga2_tmax, out['aco']['Tmax_h'], out['aco']['Tmin_h'], time.time() - t0))
    logf.close()
    return out


def main():
    cases = ['Case1', 'Case2', 'Case3', 'Case4']
    mp.set_start_method('spawn', force=True)
    with mp.Pool(len(cases)) as pool:
        outs = pool.map(solve_case, [(c, i + 1) for i, c in enumerate(cases)])

    print('\n===== 问题1 第二层 ACO 结果汇总 =====')
    print('%-8s %4s %12s %10s %12s' % ('算例', 'N', '簇编码GA', 'ACO', 'OR-Tools'))
    for o in outs:
        print('%-8s %4d %11.3f h %9.3f h %11.3f h'
              % (o['case'], o['Nmin'], o['cluster_ga']['Tmax_h'],
                 o['aco']['Tmax_h'], REF_TMAX_H[o['case']]))

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表2_ACO'
    ws.append(['测试算例', '无人机数量N', 'Tmax(h)', 'Tmin(h)'])
    for o in outs:
        ws.append([o['case'], o['Nmin'], round(o['aco']['Tmax_h'], 4),
                   round(o['aco']['Tmin_h'], 4)])
    ws2 = wb.create_sheet('三算法对比')
    ws2.append(['测试算例', '簇编码GA Tmax(h)', 'ACO Tmax(h)', 'OR-Tools参考 Tmax(h)',
                'ACO相对GA改善(h)'])
    for o in outs:
        ws2.append([o['case'], round(o['cluster_ga']['Tmax_h'], 4),
                    round(o['aco']['Tmax_h'], 4), REF_TMAX_H[o['case']],
                    round(o['aco']['improvement_h'], 4)])
    wb.save(os.path.join(OUT, 'ga_p1_aco.xlsx'))
    print('\n已导出: A 题/output/ga_p1_aco.xlsx 与各算例 ga_p1_aco_*.json / 路径图')


if __name__ == '__main__':
    main()
