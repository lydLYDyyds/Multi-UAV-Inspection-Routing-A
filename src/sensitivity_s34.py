# -*- coding: utf-8 -*-
"""S3/S4 敏感性:禁飞半径膨胀 + Case4 Z8 持续生效(基于最新 p3 json)。"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from common import load_zones, A_DIR
import p3_case
from p3_case import build_schedule, verify

CASES = ['Case1', 'Case2', 'Case3', 'Case4']
out = {}
for case in CASES:
    p3_case.load_case(case)
    with open(os.path.join(A_DIR, 'output', f'p3_{case}.json'), encoding='utf-8') as f:
        d = json.load(f)
    zones = load_zones(case)
    base_routes = d['routes']
    row = {}
    for eps in (1.0, 1.1, 1.2):
        zz = [dict(zone_id=z['zone_id'], c=z['c'], r=z['r'] * eps,
                   t0=z['t0'], t1=z['t1']) for z in zones]
        routes, span, delta = build_schedule(base_routes, zz)
        viol = verify(routes, zz)
        if viol:
            print('  违例详情:', viol[:5], flush=True)
        row[f'r{eps}'] = dict(Tmax=round(span, 3), delta=round(delta, 3), viol=len(viol))
    if case == 'Case4':
        zz = [dict(zone_id=z['zone_id'], c=z['c'], r=z['r'],
                   t0=(540.0 if z['zone_id'] == 'Z8' else z['t0']),
                   t1=(720.0 if z['zone_id'] == 'Z8' else z['t1'])) for z in zones]
        routes, span, delta = build_schedule(base_routes, zz)
        viol = verify(routes, zz)
        row['Z8_persist'] = dict(Tmax=round(span, 3), delta=round(delta, 3), viol=len(viol))
    out[case] = row
    print(case, row, flush=True)

# 合并进 sensitivity.json
sp = os.path.join(A_DIR, 'output', 'sensitivity.json')
if os.path.exists(sp):
    with open(sp, encoding='utf-8') as f:
        rep = json.load(f)
else:
    rep = {}
for case, row in out.items():
    rep[f'S3S4_{case}'] = row
with open(sp, 'w', encoding='utf-8') as f:
    json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
print('sensitivity.json 已合并 S3/S4', flush=True)
