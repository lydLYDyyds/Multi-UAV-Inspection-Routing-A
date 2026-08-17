# 低空经济背景下的多无人机协同巡检路径优化(A 题)

天津市五校数学建模联赛 A 题求解仓库。所有无人机由同一基地 (0,0) 起飞,按等级
I/II/III 对巡检点执行 3/2/1 次巡检(每次 5 分钟),巡航速度 55 km/h,8:00 起飞,
完成本机任务后返航。三问分别为:

1. **问题1**:9 小时内完成全部巡检所需的最小无人机数 Nmin,以及 Nmin 架机下最短总完成时间;
2. **问题2**:在问题1 基础上平衡各机工作负载(先最短完成时间,再最小 δ=Tmax−Tmin);
3. **问题3**:附件2 的动态圆形禁飞区(空间圆 + 生效时间窗)约束下的路径规划,飞行、巡检服务与空中等待三类状态均须避开生效禁飞区。

## 最终结果

**表2 问题1**

| 算例 | Nmin | Tmax (h) | Tmin (h) |
|---|---|---|---|
| Case1 | 4 | 7.883 | 7.712 |
| Case2 | 2 | 7.178 | 7.176 |
| Case3 | 4 | 8.373 | 8.342 |
| Case4 | 4 | 8.020 | 7.970 |

**表3 问题2**

| 算例 | N | Tmax (h) | Tmin (h) | δ (h) |
|---|---|---|---|---|
| Case1 | 4 | 7.840 | 7.763 | 0.077 |
| Case2 | 2 | 7.125 | 7.123 | 0.002 |
| Case3 | 4 | 8.368 | 8.339 | 0.029 |
| Case4 | 4 | 7.804 | 7.774 | 0.030 |

**表4 问题3**

| 算例 | N | Tmax (h) | Tmin (h) | δ (h) |
|---|---|---|---|---|
| Case1 | 4 | 7.840 | 7.764 | 0.077 |
| Case2 | 2 | 10.969 | 10.931 | 0.038 |
| Case3 | 4 | 8.427 | 8.323 | 0.103 |
| Case4 | 4 | 8.085 | 8.046 | 0.039 |

全部结果经 `src/p1_export.py`、`src/p2_export.py`、`src/p3_export.py` 独立重算校验,
写入 `A 题/result1.xlsx`、`result2.xlsx`、`result3.xlsx`。

## 环境与安装

- Python 3.10+(测试于 3.13),依赖见 `requirements.txt`,一键安装:

```powershell
python -m pip install -r requirements.txt
```

- 说明:OR-Tools 9.15 与 Python 3.13 组合下,在 transit 回调内调用 SWIG 方法会抛
  OverflowError,代码已用"主线程预计算 transit 矩阵、回调内纯查表"规避(见
  `src/p1_case.py` 顶部注释)。

## 仓库结构

```
TJMM/
├── README.md / requirements.txt / .gitignore
├── A 题/
│   ├── 附件1.xlsx, 附件2.xlsx          # 题目数据(只读)
│   ├── TJMML_A.pdf                     # 原题
│   ├── result1-3.xlsx                  # 三问最终表格
│   ├── docs/                           # 建模与求解说明文档(见下方索引)
│   └── output/                         # 最终调度 json 与路径图/甘特图/禁飞区地图
├── src/                                # 全部求解与校验代码
├── 统一时空硬约束模型_XeLaTeX_中文修正版.pdf   # 问题3 模型全文
├── 第三问_禁飞区时空解析验证模型.tex           # 问题3 模型(早期版)
└── 问题一第一部分.pdf
```

## 运行方法

```powershell
# ---- 问题1:下界、Nmin 与最短完成时间 ----
python src\p1_lb.py                # 理论下界 N_LB
python src\p1_case.py Case1        # Case1..Case4,Nmin 搜索 + min-makespan
python src\p1_export.py            # 独立校验,写 result1.xlsx
python src\plot_routes.py p1 Case1 # 路径图 + 甘特图

# ---- 问题2:ε-约束 + 下界 L 二分 ----
python src\p2_bin.py Case1         # Case1..Case4(初解来自 output/p1_*.json)
python src\p2_export.py            # 独立校验,写 result2.xlsx
python src\plot_routes.py p2 Case1

# ---- 问题3:时空硬约束模型 + ALNS ----
python src\p3_tex.py Case1         # Pass A 修复 + Pass B 带窗再优化(初解来自 output/p2_*.json)
python src\alns3.py Case1          # ALNS 全局优化(初解来自 output/p3_*.json)
python src\p3_export.py            # 三态独立校验,写 result3.xlsx
python src\plot_routes.py p3 Case1
python src\plot_zones.py           # 禁飞区空间分布与生效时段总览图

# ---- 敏感性分析(问题1/2/3 相关扰动) ----
python src\sensitivity.py
python src\sensitivity_s34.py
```

问题1 的其他求解路线(GA / 蚁群 / MILP / ILS,探索性对照)见
`src/ga_minmax.py`、`src/ga_cluster.py`、`src/aco_minmax.py`、`src/p1_milp.py`、
`src/p1_nmin_ga.py`、`src/polish_ils.py` 及入口 `p1_makespan_ga.py`、
`p1_makespan_aco.py`。

## 方法概览

- **问题1**:总工时下界(LB1-LB5)与构造性调度共同确定 Nmin;min-makespan VRP 用
  OR-Tools 求解(PATH_CHEAPEST_ARC 初解 + 引导式局部搜索 GLS),多种子取优。
- **问题2**:词典序目标 (Tmax, δ),ε-约束分层(ε=0/1%/2%);由可行域单调性对
  各机工作时长下界 L 二分搜索(容差 2 min),每个探针解 min-span 子问题。
- **问题3**:将直线航段参数化并与禁飞圆解析求交,把"空间障碍+时间窗"约束投影为
  航段禁止出发区间 F_ij;动态航段代价取直飞/等待/绕行(切线-圆弧-切线,弧向外
  偏移)最短合法时间;等待位置非法时转移至圆外悬停点。全局优化用 ALNS
  (破坏-修复 + relocate/swap/2-opt/cross 局部搜索 + 模拟退火),全程以真实动态
  代价评估,不可行方案直接排除;最后由独立解析验证器对飞行、服务、等待三类
  状态逐段复核,违例数为 0。

## 文档索引(`A 题/docs/`)

| 文档 | 内容 |
|---|---|
| 01_任务合同与需求台账 | 任务理解、假设与约束清单 |
| 02_数据审计 | 附件1/2 数据质量审计 |
| 03_问题1_建模与求解 | 问题1 下界推导与 min-makespan 模型 |
| 04_问题2与3_建模 | 问题2 二分、问题3 时空硬约束模型 |
| 05_论文大纲 | 论文骨架 |
| 06_结果汇总与论文正文 | 三问最终结果与结论(本文 README 表格的来源) |
| 06/08/09/10_问题1_* | 问题1 的 GA(两层编码)/MILP/蚁群等对照路线 |
| 08_问题2_文献综述 | 问题2 相关文献整理与引用建议 |
| 09_问题2_二分法原理与计算过程 | 二分法推导与真实计算实录 |
| 12_问题3_论文写法建议 | 问题3 论文正文素材与口径 |

## 说明

- 求解器(GLS、ALNS)给出的是高质量可行解(目标值的上界),非全局最优;
  论文口径见 docs/06 与 docs/12。
- 禁飞区按"开圆+开时间区间"约定:仅边界瞬时接触不算进入;Case4 的 Z8
  (17:00-17:00 零长度窗口)按瞬时生效处理。
- 多圆同时绕行未求全局最短路径(单圆切线绕行 + 全验证),已知局限。
