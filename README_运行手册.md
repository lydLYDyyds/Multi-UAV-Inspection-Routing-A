# A 题求解管线:运行手册(可复现性说明)

## 环境

- Windows 11 + Python 3.13.5(Anaconda/E 盘安装)
- 依赖:`numpy 2.4.4`、`scipy 1.15.3`、`pandas 2.3.3`、`openpyxl 3.1.5`、`matplotlib 3.10.5`、`networkx 3.6.1`、`ortools 9.15.6755`(pip 安装)
- 数据(只读,不修改原件):`A 题/附件1.xlsx`、`A 题/附件2.xlsx`、`A 题/TJMML_A.pdf`

## 目录结构

```
TJMM/
  SKILL.md                 # 工作契约(用户指定)
  audit_dataset.py         # 数据审计 → A 题/docs/02_数据审计.md
  plot_map.py              # 坐标地图 → A 题/map_附件1_坐标地图.png
  src/
    common.py              # 常量/加载/单位换算
    p1_lb.py               # 问题1.1 下界(N_LB)
    p1_case.py <Case>      # 问题1: Nmin 搜索 + min-makespan → output/p1_<Case>.json
    p1_export.py           # 校验 + result1.xlsx
    ga_minmax.py           # 问题1 第一部分 GA 引擎(巨型路线+最优分割+局部搜索)
    p1_nmin_ga.py          # 问题1 第一部分: GA 求 Nmin(4 算例并行)→ output/ga_p1_*.json
    ga_cluster.py          # 问题1 第二层: 簇编码 GA(簇间交换/搬移/重组变异,簇内贪心+SA)+ ILS/喷射链
    p1_makespan_ga.py      # 问题1 第二层: Nmin 下最小化 makespan → output/ga_p1_layer2_*.json
    aco_minmax.py          # 问题1 第二层: 蚁群算法(巨型路线+最优分割+deep_ls,信息素热启动)
    p1_makespan_aco.py     # 问题1 第二层 ACO 驱动 → output/ga_p1_aco_*.json
    polish_ils.py          # 多链 ILS 打磨(从第二层/第一层解出发多链搜索,回写最优)
    p1_milp.py <Case> <N>  # 问题1 第一层: MTZ 精确模型(scipy/HiGHS),小规模验证 + 限时求解试点
    _test_ga.py            # GA 冒烟测试(分割DP vs 暴力DP 对照等)
    _test_sym.py           # 对称性破缺专项测试(副本重排无损性、规范化多重集)
    _verify_ga.py          # GA 结果独立复核(直读 JSON+附件1 重算,支持文件名模式参数)
    _fuzz_ls.py            # 局部搜索算子时间一致性 fuzz 回归(曾定位 2-opt* off-by-one)
    _fuzz_ils.py           # ejection/deep_ls/ils 任务多重集 + 时间一致性 fuzz 回归
    p2_case.py <Case>      # 问题2: L-网格均衡 → output/p2_<Case>.json
    p2_export.py           # 校验 + result2.xlsx
    p3_case.py <Case>      # 问题3: 修复+再优化 → output/p3_<Case>.json
    p3_export.py           # 校验 + result3.xlsx
    plot_routes.py pN <Case>  # 路径图/甘特图 → output/figs/
    sensitivity.py         # S1..S6 → output/sensitivity.json
  A 题/
    docs/ 01..04           # 台账/审计/建模文档
    output/                # 全部机器可读结果(JSON/figs/日志)
    result1/2/3.xlsx       # 交付表格
```

## 运行顺序(单一入口等效)

```powershell
python src\p1_lb.py
python src\p1_case.py Case1   # Case2..4 同理(可并行)
python src\p1_export.py
python src\_test_ga.py        # GA 单元测试(可选)
python src\p1_nmin_ga.py      # 问题1 第一部分 GA 方案(4 算例并行,约 1.5 min)
python src\p1_makespan_ga.py  # 问题1 第二层: 簇编码 GA + ILS(依赖第一层 JSON,约 10 min)
python src\p1_makespan_aco.py # 问题1 第二层: ACO(依赖第二层 JSON 作热启动,约 5 min)
python src\polish_ils.py      # 多链 ILS 打磨(依赖第二层 JSON,约 30 min)
python src\_verify_ga.py      # 第一层结果独立复核(第二层: python src\_verify_ga.py ga_p1_layer2_%s.json)
python src\p2_case.py Case1   # 依赖 p1_<Case>.json
python src\p2_export.py
python src\p3_case.py Case1   # 依赖 p2_<Case>.json
python src\p3_export.py
python src\plot_routes.py p1 Case1   # p1|p2|p3 × Case1..4
python src\sensitivity.py
```

## 随机性与种子

- OR-Tools 9.15 无 `random_seed` 顶层字段,改用 `sat_parameters.random_seed`(固定种子 1..7,脚本内写死)。
- 问题1:Nmin 搜索 3 种子×40 s;终解 6 种子×90 s 取最优。
- 问题1 第一部分 GA 方案:pop 80、≤1000 代、patience 250、重启 120;证据 N 3 种子、候选 N 6 种子、抛光 2 种子×1500 代;种子 = ci×1000+N×10+s。
- 问题1 第二层(簇编码 GA):pop 40、≤500 代、patience 200、4 种子 + ILS 300 迭代;种子 = ci×7000+s;以第一层划分为初始种子。
- 问题2:L 网格步长 30 min,每点 2 种子×30 s。
- 问题3:Pass A 确定性修复;Pass B 3 个 L×30 s×1 种子,≤2 轮。
- S6 扰动种子 31/32/33。
- 所有输出 JSON 均含求解参数与校验结果,可追溯。

## 单位与精度

- 1 坐标单位 = 100 m;速度 55 km/h;单次巡检 5 min。
- 时间以 8:00 为 0;求解器内部取整数秒(6.54545… s/单位取整);输出时刻为分钟(2 位小数)。
- 校验脚本按相同取整规则独立重算,容差:时刻 0.02 min、距离 0.005 u、返航 1 s/任务。

## 已知工程注意事项

1. OR-Tools 9.15 + Py3.13:transit 回调内调用 SWIG(`IndexToNode`)抛 OverflowError → 已用"主线程预计算矩阵 + 回调纯查表"规避(见 `src/_smoke_915.py` 最小复现)。
2. OR-Tools 9.15 API 改名:`Start/End → GetStartIndex/GetEndIndex`;`random_seed → sat_parameters.random_seed`。
3. 模块可安全 import(主流程均在 `main()` 中,`__main__` 守卫)。
4. 问题3 的禁飞区边界约定:开圆(相切/压线允许);Case4 Z8 零长度窗口按瞬时生效,另有敏感性对照。
5. 局部搜索曾有两个 off-by-one 缺陷,均已修复并由 fuzz 回归:①2-opt\* 的前缀和与切片内部距离差一条边界边(`_fuzz_ls.py` 定位);②Or-opt(机内/跨机块搬迁)的块内距离应为 pref[p+b]−pref[p+1](`_fuzz_ils.py` 定位,配合任务多重集检查)。两层与 ACO 结果均已按修复后代码重跑并经独立复核。
6. 对称性破缺:等级展开产生的同点副本按"出现顺序"规范重编号(排列编码/簇编码在评估时规范化、ACO 构造时按点选代表),严格无损,冗余度(∏ k_i!)压缩为 1;见 `_test_sym.py` 与 docs/11。

## 诚实性声明(不回避)

- 问题1 下界(LB1..LB5)未计入"覆盖全部点"的路径总长,对 N≥2 宽松;Nmin 以构造可行调度 + N−1 求解证据确立,证明缺口在论文中明示。
- OR-Tools 输出为高质量近似解(makespan 上界),非全局最优证明。
- 所有表格数字均由 `*_export.py` 独立重算校验后写入,不抄求解器原始输出。
