# -*- coding: utf-8 -*-
"""问题2 结果独立校验 + 导出 result2.xlsx(表3 格式 + 调度明细)。"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import openpyxl
from common import load_points, A_DIR, HORIZON_MIN

CASES = ['Case1', 'Case2', 'Case3', 'Case4']
TAU_S = 0.1 / 55.0 * 3600.0
SERVICE_S = 300


def hhmm(minutes):
    return f'{int(8 + minutes // 60):02d}:{int(minutes % 60):02d}'


def validate(case, routes):
    p = load_points(case)
    xy, visits = p['xy'], p['visits']
    M = int(visits.sum())
    seen = set()
    rep = []
    for r in routes:
        cum, prev, legs = 0, 0, []
        arrs, deps = [], []
        for s in r['seq']:
            key = (s['point_id'], s['visit_no'])
            assert key not in seen, f'{case}: 任务 {key} 重复'
            seen.add(key)
            pxy = xy[s['point_id'] - 1]
            leg = float(np.hypot(*(pxy - (xy[prev - 1] if prev else [0, 0]))))
            cum += int(round(leg * TAU_S)); arrs.append(cum / 60)
            cum += SERVICE_S; deps.append(cum / 60)
            legs.append(round(leg, 3)); prev = s['point_id']
        cum += int(round(float(np.hypot(*(xy[prev - 1] if prev else [0, 0]))) * TAU_S))
        legs.append(round(float(np.hypot(*(xy[prev - 1] if prev else [0, 0]))), 3))
        for k in range(len(r['seq'])):
            assert abs(arrs[k] - r['arr_min'][k]) < 0.02
            assert abs(deps[k] - r['dep_min'][k]) < 0.02
        rep.append((cum, arrs, deps, legs))
    assert len(seen) == M, f'{case}: 覆盖 {len(seen)}/{M}'
    ends = [x[0] for x in rep]
    return max(ends) / 3600.0, min(ends) / 3600.0


def export():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '表3_问题二结果'
    ws.append(['测试算例', '无人机数量N', '单架无人机最长工作时间Tmax(h)',
               '单架无人机最短工作时间Tmin(h)', 'δ=|Tmax−Tmin|(h)'])
    rows = []
    for case in CASES:
        with open(os.path.join(A_DIR, 'output', f'p2_{case}.json'), encoding='utf-8') as f:
            data = json.load(f)
        rec = next((g for g in data['grid'] if g['eps'] == 0.0), None) \
            or next((g for g in data['grid'] if g['eps'] == 0.01), None) \
            or data['grid'][0]
        tmax, tmin = validate(case, rec['routes'])
        delta = round(tmax - tmin, 4)
        rows.append(dict(case=case, N=data['N'], tmax=round(tmax, 3),
                         tmin=round(tmin, 3), delta=delta))
        ws.append([case, data['N'], round(tmax, 3), round(tmin, 3), delta])
        print(f'{case} 校验通过: N={data["N"]} Tmax={tmax:.3f}h Tmin={tmin:.3f}h δ={delta:.3f}h')

        ws2 = wb.create_sheet(f'{case}_调度方案')
        ws2.append(['无人机编号', '顺序号', '巡检点ID', '巡检等级', '第几次巡检',
                    '到达时刻(8:00起)', '离开时刻', '本段飞行距离(单位)',
                    '本机累计飞行距离(单位)', '本机总工作时长(h)'])
        p = load_points(case)
        for r in rec['routes']:
            for k, s in enumerate(r['seq']):
                pid = s['point_id']
                ws2.append([r['veh'], k + 1, pid, p['level'][pid - 1], s['visit_no'],
                            hhmm(r['arr_min'][k]), hhmm(r['dep_min'][k]), r['legs_u'][k],
                            round(sum(r['legs_u'][:k + 1]), 2), r['busy_h']])
    out = os.path.join(A_DIR, 'result2.xlsx')
    wb.save(out)
    print('\n已写入:', out)
    for r in rows:
        print(f"  {r['case']}: N={r['N']}  Tmax={r['tmax']}h  Tmin={r['tmin']}h  δ={r['delta']}h")


if __name__ == '__main__':
    export()
