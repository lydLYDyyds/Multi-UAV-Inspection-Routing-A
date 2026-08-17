# -*- coding: utf-8 -*-
"""问题1 第二层(1.2):簇编码 GA 在 N=Nmin 下最小化全局完成时间(4 算例并行)。

流程:
1. 读第一层结果 ga_p1_<Case>.json 的 Nmin 与最终划分(作为初始种子);
2. 簇编码 GA(ClusterGA)以该划分为种子继续进化:簇间交换/搬移/合并拆分变异 +
   簇内贪心/2-opt/模拟退火,目标 min max T_k(副目标总时长破平);
3. 独立校验(validate_schedule + 按坐标重算)后导出。

输出:A 题/output/ga_p1_layer2_<Case>.json、ga_p1_layer2_<Case>.log、
     figs/ga_p1_layer2_<Case>_路径图.png、ga_p1_makespan.xlsx(表2 最终值 + 三层对比)。
"""
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import A_DIR, HORIZON_MIN, load_points
from ga_cluster import ClusterGA, ils
from ga_minmax import validate_schedule
from p1_nmin_ga import (OUT, FIG, expand_tasks, plot_case, schedule_times)

os.makedirs(FIG, exist_ok=True)

# 参数
POP_SIZE = 40
MAX_GENS = 500
PATIENCE = 200
RESTART_EVERY = 100
N_SEEDS = 4
ILS_ITERS = 400
HORIZON = HORIZON_MIN

# OR-Tools 参考解(docs/03 / result1.xlsx),仅用于对比展示
REF_TMAX_H = {'Case1': 7.883, 'Case2': 7.178, 'Case3': 8.373, 'Case4': 8.020}


def solve_case(args):
    case, ci = args
    logf = open(os.path.join(OUT, 'ga_p1_layer2_%s.log' % case), 'w', encoding='utf-8')

    def log(*a):
        print(*a, file=logf, flush=True)

    t0 = time.time()
    p = load_points(case)
    tasks = expand_tasks(p)
    M = len(tasks)
    d1 = json.load(open(os.path.join(OUT, 'ga_p1_%s.json' % case), encoding='utf-8'))
    nmin = d1['Nmin']
    seed_routes = [[t['task_id'] for t in r['tasks']] for r in d1['schedule']['routes']]
    l1 = d1['schedule']
    log('=== %s: M=%d Nmin=%d, 第一层 Tmax=%.3f h ===' % (case, M, nmin, l1['Tmax_h']))

    best = None
    for s in range(N_SEEDS):
        ga = ClusterGA(tasks, p['xy'], N=nmin, seed=ci * 7000 + s, pop_size=POP_SIZE,
                       seed_routes=seed_routes, sa_intra=True)
        r = ga.run(max_gens=MAX_GENS, patience=PATIENCE, restart_every=RESTART_EVERY)
        log('  seed=%d: mk=%.2f min (%.3f h) tsum=%.2f min gens=%d valid=%s'
            % (s, r['mk'], r['mk'] / 60, r['tsum'], r['gens'], ga.is_valid(r['routes'])))
        if best is None or r['mk'] < best[1]['mk'] - 1e-9:
            best = (ga, r)
    ga, r = best
    # ILS 末段:强制扰动 + 局部搜索循环,突破深局部最优(Case3 类平台)
    ils_routes, ils_times, ils_mk = ils(r['routes'], ga.D_ext, M, iters=ILS_ITERS,
                                        perturb_k=3, seed=ci * 9000 + 1)
    ils_improved = 0.0
    if ils_mk < r['mk'] - 1e-9:
        ils_improved = r['mk'] - ils_mk
        r = dict(mk=ils_mk, tsum=sum(ils_times), routes=ils_routes,
                 gens=r['gens'], seed=r['seed'])
        log('  ILS: mk -> %.2f min (%.3f h), 改善 %.2f min'
            % (ils_mk, ils_mk / 60, ils_improved))
    rep = validate_schedule(M, r['routes'], ga.D_ext, horizon=HORIZON)
    assert rep['coverage_ok'] and rep['horizon_ok'], '第二层解未通过校验'
    det = schedule_times(r['routes'], ga.D_ext, M)

    l2 = dict(Tmax_min=rep['Tmax_min'], Tmax_h=rep['Tmax_min'] / 60,
              Tmin_min=rep['Tmin_min'], Tmin_h=rep['Tmin_min'] / 60,
              improvement_h=l1['Tmax_h'] - rep['Tmax_min'] / 60,
              tsum_min=r['tsum'], seed=r['seed'], gens=r['gens'],
              ils_improved_min=ils_improved)
    out = dict(case=case, n=int(len(p['ids'])), M=M, Nmin=nmin,
               wall_s=round(time.time() - t0, 1),
               params=dict(pop_size=POP_SIZE, max_gens=MAX_GENS, patience=PATIENCE,
                           restart_every=RESTART_EVERY, n_seeds=N_SEEDS, sa_intra=True),
               layer1=dict(Tmax_h=l1['Tmax_h'], Tmin_h=l1['Tmin_h']),
               layer2=l2,
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
    with open(os.path.join(OUT, 'ga_p1_layer2_%s.json' % case), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    plot_case(case, p, tasks, r['routes'], rep['Tmax_min'],
              fname='ga_p1_layer2_%s_路径图.png' % case)
    log('=== %s 第二层完成: Tmax %.3f -> %.3f h (改善 %.3f h), Tmin=%.3f h, wall=%.1f s ==='
        % (case, l1['Tmax_h'], l2['Tmax_h'], l2['improvement_h'], l2['Tmin_h'],
           time.time() - t0))
    logf.close()
    return out


def main():
    cases = ['Case1', 'Case2', 'Case3', 'Case4']
    mp.set_start_method('spawn', force=True)
    with mp.Pool(len(cases)) as pool:
        outs = pool.map(solve_case, [(c, i + 1) for i, c in enumerate(cases)])

    print('\n===== 问题1 第二层(簇编码GA)结果汇总 =====')
    print('%-8s %4s %10s %10s %8s %10s' % ('算例', 'N', '第一层Tmax', '第二层Tmax', '改善', 'OR-Tools'))
    for o in outs:
        print('%-8s %4d %9.3f h %9.3f h %+7.3f h %9.3f h'
              % (o['case'], o['Nmin'], o['layer1']['Tmax_h'], o['layer2']['Tmax_h'],
                 o['layer2']['improvement_h'], REF_TMAX_H[o['case']]))

    # 导出表2(最终值)与三层对比
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表2'
    ws.append(['测试算例', '无人机数量N', '单架无人机最长工作时间Tmax(h)',
               '单架无人机最短工作时间Tmin(h)'])
    for o in outs:
        ws.append([o['case'], o['Nmin'], round(o['layer2']['Tmax_h'], 4),
                   round(o['layer2']['Tmin_h'], 4)])
    ws2 = wb.create_sheet('三层对比')
    ws2.append(['测试算例', '第一层GA Tmax(h)', '第二层簇编码GA Tmax(h)',
                'OR-Tools参考 Tmax(h)', '第二层改善(h)'])
    for o in outs:
        ws2.append([o['case'], round(o['layer1']['Tmax_h'], 4),
                    round(o['layer2']['Tmax_h'], 4), REF_TMAX_H[o['case']],
                    round(o['layer2']['improvement_h'], 4)])
    wb.save(os.path.join(OUT, 'ga_p1_makespan.xlsx'))
    print('\n已导出: A 题/output/ga_p1_makespan.xlsx 与各算例 ga_p1_layer2_*.json / 路径图')


if __name__ == '__main__':
    main()
