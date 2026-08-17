# -*- coding: utf-8 -*-
"""多链 ILS 打磨:从(当前第二层解 + 第一层解)出发,多条不同种子的 ILS 链搜索,
取最优回写 ga_p1_layer2_<Case>.json(4 算例并行)。用于消除退火 ILS 的随机性回归。"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HORIZON_MIN, load_points
from ga_cluster import ils
from ga_minmax import build_D_ext, validate_schedule
from p1_nmin_ga import OUT, FIG, expand_tasks, schedule_times, plot_case

CHAINS_PER_START = 4
ILs_ITERS = 400


def solve_case(args):
    case, ci = args
    logf = open(os.path.join(OUT, 'ga_p1_polish_%s.log' % case), 'w', encoding='utf-8')

    def log(*a):
        print(*a, file=logf, flush=True)

    t0 = time.time()
    p = load_points(case)
    tasks = expand_tasks(p)
    M = len(tasks)
    D_ext, _ = build_D_ext(tasks, p['xy'])
    d2 = json.load(open(os.path.join(OUT, 'ga_p1_layer2_%s.json' % case), encoding='utf-8'))
    d1 = json.load(open(os.path.join(OUT, 'ga_p1_%s.json' % case), encoding='utf-8'))
    routes2 = [[t['task_id'] for t in r['tasks']] for r in d2['schedule']['routes']]
    routes1 = [[t['task_id'] for t in r['tasks']] for r in d1['schedule']['routes']]
    mk2 = d2['layer2']['Tmax_min']
    log('=== %s: 当前第二层 %.3f h, 第一层 %.3f h ==='
        % (case, mk2 / 60, d1['schedule']['Tmax_h']))

    best_routes, best_times, best_mk = routes2, None, mk2
    best_src = 'current'
    for si, sr in enumerate([routes2, routes1]):
        for k in range(CHAINS_PER_START):
            r, times, mk = ils(sr, D_ext, M, iters=ILs_ITERS, perturb_k=3,
                               seed=ci * 7777 + si * 100 + k)
            log('  start=%d chain=%d: mk=%.2f min (%.3f h)'
                % (si, k, mk, mk / 60))
            if mk < best_mk - 1e-9:
                best_mk = mk
                best_routes = [x[:] for x in r]
                best_times = times
                best_src = 'start%d-chain%d' % (si, k)
    log('  => 最优 %.3f h(来源 %s)' % (best_mk / 60, best_src))

    if best_mk < mk2 - 1e-9:
        rep = validate_schedule(M, best_routes, D_ext, horizon=HORIZON_MIN)
        assert rep['coverage_ok'] and rep['horizon_ok']
        det = schedule_times(best_routes, D_ext, M)
        d2['layer2'].update(dict(
            Tmax_min=rep['Tmax_min'], Tmax_h=rep['Tmax_min'] / 60,
            Tmin_min=rep['Tmin_min'], Tmin_h=rep['Tmin_min'] / 60,
            improvement_h=d2['layer1']['Tmax_h'] - rep['Tmax_min'] / 60,
            tsum_min=sum(rep['route_times']), polish_source=best_src))
        d2['schedule'] = dict(
            Tmax_min=rep['Tmax_min'], Tmin_min=rep['Tmin_min'],
            route_times=rep['route_times'],
            routes=[dict(drone=d['drone'], total_min=d['total_min'],
                         dist_u=d['dist_u'], n_tasks=d['n_tasks'],
                         tasks=[dict(task_id=t[0],
                                     point_id=int(p['ids'][tasks[t[0]][0]]),
                                     point_idx=int(tasks[t[0]][0]),
                                     visit=int(tasks[t[0]][1]),
                                     x=float(p['xy'][tasks[t[0]][0], 0]),
                                     y=float(p['xy'][tasks[t[0]][0], 1]),
                                     arr_min=t[1], dep_min=t[2])
                                for t in d['tasks']])
                    for d in det])
        with open(os.path.join(OUT, 'ga_p1_layer2_%s.json' % case), 'w',
                  encoding='utf-8') as f:
            json.dump(d2, f, ensure_ascii=False, indent=1)
        plot_case(case, p, tasks, best_routes, rep['Tmax_min'],
                  fname='ga_p1_layer2_%s_路径图.png' % case)
        log('=== %s 打磨完成: %.3f -> %.3f h ===' % (case, mk2 / 60, best_mk / 60))
    else:
        log('=== %s 无改善(保持 %.3f h)===' % (case, mk2 / 60))
    logf.close()
    return dict(case=case, old_h=mk2 / 60, new_h=best_mk / 60, source=best_src,
                wall=time.time() - t0)


def main():
    cases = ['Case1', 'Case2', 'Case3', 'Case4']
    mp.set_start_method('spawn', force=True)
    with mp.Pool(len(cases)) as pool:
        outs = pool.map(solve_case, [(c, i + 1) for i, c in enumerate(cases)])
    print('\n===== 多链 ILS 打磨结果 =====')
    for o in outs:
        print('%-8s %.3f h -> %.3f h (%s, %.0f s)'
              % (o['case'], o['old_h'], o['new_h'], o['source'], o['wall']))


if __name__ == '__main__':
    main()
