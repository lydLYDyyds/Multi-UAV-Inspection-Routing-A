# -*- coding: utf-8 -*-
"""问题1 结果独立校验 + 导出 result1.xlsx。

校验(与求解器输出交叉核对,不信任求解器自身):
1. 每个 (巡检点, 第k次) 任务恰被覆盖一次;
2. 每机首尾均为基地,顺序连续;
3. 时间递推:离开=到达+5min;下一段到达=离开+飞行时间(τ×距离,秒取整);
4. Tmax ≤ 540 min(问题1);Tmin = min(各机返航时刻);
5. 重新计算总飞行距离与求解器记录一致。

导出:A 题/result1.xlsx
  - sheet "表2_问题一结果": 表2 格式
  - 每算例一个 sheet "<Case>_调度方案": 逐任务明细 + 每机汇总行
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import openpyxl
from common import load_points, A_DIR, TAU_MIN, SERVICE_MIN, HORIZON_MIN, LEVEL_VISITS

CASES = ['Case1', 'Case2', 'Case3', 'Case4']
TAU_S = 0.1 / 55.0 * 3600.0
SERVICE_S = 300


def hhmm(minutes):
    return f'{int(8 + minutes // 60):02d}:{int(minutes % 60):02d}'


def validate(case, data):
    p = load_points(case)
    xy, visits = p['xy'], p['visits']
    M = int(visits.sum())
    seen = set()
    rep = []
    for r in data['routes']:
        nodes = [0] + [None] * len(r['seq']) + [0]
        cum = 0
        arr_rep, dep_rep, legs_rep = [], [], []
        prev = 0
        for k, s in enumerate(r['seq']):
            pid = s['point_id']
            key = (pid, s['visit_no'])
            assert key not in seen, f'{case}: 任务 {key} 重复'
            seen.add(key)
            pxy = xy[pid - 1]
            leg_u = float(np.hypot(*(pxy - (xy[prev - 1] if prev else [0, 0]))))
            cum += int(round(leg_u * TAU_S))            # 到达
            arr_rep.append(cum / 60.0)
            cum += SERVICE_S                             # 服务
            dep_rep.append(cum / 60.0)
            legs_rep.append(round(leg_u, 3))
            prev = pid
        leg_back = float(np.hypot(*(xy[prev - 1] if prev else [0, 0])))
        cum += int(round(leg_back * TAU_S))
        legs_rep.append(round(leg_back, 3))
        # 与求解器输出比对
        for k in range(len(r['seq'])):
            assert abs(arr_rep[k] - r['arr_min'][k]) < 0.02, f'{case} 机{r["veh"]} 到达时刻不符'
            assert abs(dep_rep[k] - r['dep_min'][k]) < 0.02, f'{case} 机{r["veh"]} 离开时刻不符'
            assert abs(legs_rep[k] - r['legs_u'][k]) < 0.005, f'{case} 机{r["veh"]} 段距离不符'
        assert abs(sum(legs_rep) - r['travel_units']) < 0.02, f'{case} 机{r["veh"]} 总距离不符'
        assert abs(cum - r['end_s']) <= 1 + 2 * len(r['seq']), f'{case} 机{r["veh"]} 返航时刻不符'
        rep.append(dict(arr=arr_rep, dep=dep_rep, legs=legs_rep, end_s=cum))
    assert len(seen) == M, f'{case}: 覆盖 {len(seen)}/{M} 任务'
    span = max(x['end_s'] for x in rep)
    assert span <= HORIZON_MIN * 60 + 1, f'{case}: Tmax={span/3600:.3f}h 超过 9h'
    assert abs(span - data['span_s']) <= 2, f'{case}: span 不一致'
    return rep


def export():
    summary_rows = []
    for case in CASES:
        with open(os.path.join(A_DIR, 'output', f'p1_{case}.json'), encoding='utf-8') as f:
            data = json.load(f)['best']
        rep = validate(case, data)
        s = dict(case=case, N=len(data['routes']),
                 tmax=round(data['span_h'], 3), tmin=round(min(r['busy_h'] for r in data['routes']), 3))
        summary_rows.append(s)
        print(f'{case} 校验通过: N={s["N"]} Tmax={s["tmax"]}h Tmin={s["tmin"]}h')

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表2_问题一结果'
    ws.append(['测试算例', '无人机数量N', '单架无人机最长工作时间Tmax(h)',
               '单架无人机最短工作时间Tmin(h)'])
    for s in summary_rows:
        ws.append([s['case'], s['N'], s['tmax'], s['tmin']])

    for case in CASES:
        with open(os.path.join(A_DIR, 'output', f'p1_{case}.json'), encoding='utf-8') as f:
            data = json.load(f)['best']
        p = load_points(case)
        ws = wb.create_sheet(f'{case}_调度方案')
        ws.append(['无人机编号', '顺序号', '巡检点ID', '巡检等级', '第几次巡检',
                   '到达时刻(8:00起)', '离开时刻', '本段飞行距离(单位)',
                   '本机累计飞行距离(单位)', '本机总工作时长(h)'])
        for r in data['routes']:
            for k, s in enumerate(r['seq']):
                pid = s['point_id']
                ws.append([r['veh'], k + 1, pid, p['level'][pid - 1], s['visit_no'],
                           hhmm(r['arr_min'][k]), hhmm(r['dep_min'][k]), r['legs_u'][k],
                           round(sum(r['legs_u'][:k + 1]), 2), r['busy_h']])
    out = os.path.join(A_DIR, 'result1.xlsx')
    wb.save(out)
    print('\n已写入:', out)
    print('表2:')
    for s in summary_rows:
        print(f"  {s['case']}: N={s['N']}  Tmax={s['tmax']}h  Tmin={s['tmin']}h")


if __name__ == '__main__':
    export()
