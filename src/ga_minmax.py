# -*- coding: utf-8 -*-
"""问题1 第一层:遗传算法求解给定 N 台机的 min-makespan 多无人机巡检调度(min-max mTSP)。

模型(与"问题一第一部分.pdf"公式一致,全部以分钟计):
  路线 k 工作时间 T_k = τ·(相邻点距之和 + 往返基地距离) + 5·(该机巡检次数), τ = 6/55 min/u
  目标:min max_k T_k,且每个巡检任务恰被服务一次(所有机同时从基地出发、完成各自任务后返回)。

算法组件(巨型路线 + 最优分割,Prins 风格混合遗传算法):
- 染色体:M 个巡检任务的排列(巨型路线);
- 适应度:split_minmax() 把排列切成 N 个非空连续段、使最长段工作时间最小。
  这是针对该排列的精确 N 段划分 DP,复杂度 O(MN log M)
  (利用段代价 cost[i,j]=u[i]+v[j]+5 的可分结构 + 两个堆),与 O(M²N) 暴力 DP 在测试中等价;
- 初始化:最近邻巨型路线、按基地扫掠角排序的巨型路线 + 随机排列;
- 选择:三元锦标赛;交叉:OX(顺序交叉,概率 pc);变异:两点交换/区段反转/移位;
- 局部搜索(对当前最优解译码出的路线):2-opt(机内)、跨机 relocate、跨机 exchange、2-opt*;
- 精英保留;随机性由 numpy default_rng(seed) 控制,全程可复现。
"""
import heapq
import numpy as np

from common import TAU_MIN, SERVICE_MIN

__all__ = ['split_minmax', 'split_minmax_bf', 'decode_routes', 'route_time_full',
           'local_search', 'GA', 'validate_schedule', 'build_D_ext',
           'canonicalize_seq']


def canonicalize_seq(seq, pt_of_task, task_id_of):
    """对称性破缺:同一物理点的副本按其在序列中的出现顺序重编号(规范代表)。

    只改任务标签、不改变物理路线(同点副本坐标相同),故对分割 DP 的段成本
    严格无损,任务多重集保持不变;等价编码族(≈ Π k_i! 个)被压缩为一个代表。
    O(M)。
    """
    cnt = {}
    out = np.empty(len(seq), dtype=np.int64)
    for pos in range(len(seq)):
        t = int(seq[pos])
        p = int(pt_of_task[t])
        c = cnt.get(p, 0) + 1
        cnt[p] = c
        out[pos] = task_id_of[(p, c)]
    return out


def build_D_ext(tasks, xy):
    """任务级扩展距离矩阵(含基地节点,编号 M)与任务坐标。返回 (D_ext, X)。"""
    tp = np.array([t[0] for t in tasks], dtype=np.int64)
    X = xy[tp]
    M = len(tasks)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    d0 = np.hypot(X[:, 0], X[:, 1])
    D_ext = np.zeros((M + 1, M + 1))
    D_ext[:M, :M] = D
    D_ext[M, :M] = d0
    D_ext[:M, M] = d0
    return D_ext, X


# ----------------------------------------------------------------------
# 最优分割 DP
# ----------------------------------------------------------------------
def split_minmax(seq, N, D, d0):
    """把巨型路线 seq 切成 N 个非空连续段,最小化最长段的工作时间。

    段 [i..j] 时间 = TAU*(d0[seq[i]] + Σ_{t=i}^{j-1} D[seq[t],seq[t+1]] + d0[seq[j]]) + 5*(j-i+1)。
    DP: g_k[i] = min_{j∈[k..i]} max(g_{k-1}[j-1], cost[j,i]), 答案 g_{N-1}[M-1]。
    设 t_j = g_{k-1}[j-1]-5-u[j]:当 v[i] < t_j 时 max 项取 g_{k-1}[j-1](堆A),
    否则取 cost[j,i] = v[i]+5+u[j](堆B,取 min u[j]);v[i] 随 i 不减,每个 j 只从 A 移入 B 一次。

    返回 (makespan, cuts):cuts 为 N 段起点下标(长度 N);不可行返回 (inf, None)。
    """
    M = len(seq)
    if N > M:
        return float('inf'), None
    P = np.zeros(M)
    if M > 1:
        P[1:] = np.cumsum(D[seq[:-1], seq[1:]])
    u = TAU_MIN * (d0[seq] - P) - SERVICE_MIN * np.arange(M)
    v = TAU_MIN * (P + d0[seq]) + SERVICE_MIN * np.arange(M)

    gprev = u[0] + v + SERVICE_MIN                       # g_0[i] = cost[0,i]
    cut_rows = [np.zeros(M, dtype=np.int64)]
    for k in range(1, N):
        gcur = np.full(M, np.inf)
        cutk = np.zeros(M, dtype=np.int64)
        heapA, heapB = [], []                            # A:(gprev[j-1],j)  B:(u[j],j)
        for i in range(k, M):
            j = i
            t_j = gprev[j - 1] - SERVICE_MIN - u[j]
            if t_j > v[i]:
                heapq.heappush(heapA, (gprev[j - 1], j))
            else:
                heapq.heappush(heapB, (u[j], j))
            while heapA:                                 # 陈旧项 A -> B
                _, ja = heapA[0]
                if gprev[ja - 1] - SERVICE_MIN - u[ja] > v[i]:
                    break
                heapq.heappop(heapA)
                heapq.heappush(heapB, (u[ja], ja))
            valA = heapA[0][0] if heapA else np.inf
            valB = (v[i] + SERVICE_MIN + heapB[0][0]) if heapB else np.inf
            if valA <= valB:
                gcur[i], cutk[i] = valA, heapA[0][1]
            else:
                gcur[i], cutk[i] = valB, heapB[0][1]
        gprev = gcur
        cut_rows.append(cutk)

    if not np.isfinite(gprev[M - 1]):
        return float('inf'), None
    cuts = [0] * N
    i = M - 1
    for k in range(N - 1, 0, -1):
        j = int(cut_rows[k][i])
        cuts[k] = j
        i = j - 1
    return float(gprev[M - 1]), cuts


def split_minmax_bf(seq, N, D, d0):
    """O(M²N) 暴力 DP,仅用于测试对照。"""
    M = len(seq)
    P = np.zeros(M)
    if M > 1:
        P[1:] = np.cumsum(D[seq[:-1], seq[1:]])
    C = np.full((M, M), np.inf)
    for i in range(M):
        for j in range(i, M):
            C[i, j] = TAU_MIN * (d0[seq[i]] + (P[j] - P[i]) + d0[seq[j]]) + SERVICE_MIN * (j - i + 1)
    g = C[0].copy()
    for k in range(1, N):
        g2 = np.full(M, np.inf)
        for i in range(k, M):
            g2[i] = min(max(g[j - 1], C[j, i]) for j in range(k, i + 1))
        g = g2
    return float(g[M - 1])


def decode_routes(seq, cuts, N):
    """由巨型路线与段起点解码为 N 条路线(任务 id 列表)。"""
    M = len(seq)
    routes = []
    for k in range(N):
        s = cuts[k]
        e = (cuts[k + 1] - 1) if k + 1 < N else M - 1
        routes.append([int(x) for x in seq[s:e + 1]])
    return routes


# ----------------------------------------------------------------------
# 路线时间与局部搜索
# ----------------------------------------------------------------------
def route_time_full(route, D_ext, M):
    """一条路线(任务 id 列表)的总工作时间(分钟),基地节点编号 = M。"""
    if not route:
        return 0.0
    r = np.asarray(route, dtype=np.int64)
    L = len(r)
    total = D_ext[M, r[0]] + D_ext[r[-1], M]
    if L > 1:
        total += D_ext[r[:-1], r[1:]].sum()
    return TAU_MIN * float(total) + SERVICE_MIN * L


def two_opt_once(route, D_ext, M, t_cur):
    """机内 2-opt 首改进(环形,含基地)。返回 (新时间, 是否改进)。"""
    L = len(route)
    if L < 3:
        return t_cur, False
    cyc = [M] + route
    D = D_ext
    for a in range(L):
        ca, ca1 = cyc[a], cyc[a + 1]
        for b in range(a + 2, L + 1):
            cb = cyc[b]
            cb1 = cyc[b + 1] if b < L else cyc[0]
            delta = D[ca, ca1] + D[cb, cb1] - D[ca, cb] - D[ca1, cb1]
            if delta > 1e-9:
                route[a:b] = route[a:b][::-1]
                return t_cur - TAU_MIN * delta, True
    return t_cur, False


def two_opt_all(route, D_ext, M, t_cur):
    while True:
        t_new, ok = two_opt_once(route, D_ext, M, t_cur)
        if not ok:
            return t_cur
        t_cur = t_new


def _removal_delta(route, p, D_ext, M):
    """从路线 route 位置 p 移除一个任务带来的时间减少(分钟)。"""
    L = len(route)
    x = route[p]
    prev = route[p - 1] if p > 0 else M
    nxt = route[p + 1] if p < L - 1 else M
    return TAU_MIN * (D_ext[prev, x] + D_ext[x, nxt] - D_ext[prev, nxt]) + SERVICE_MIN


def _insertion_delta(route, q, x, D_ext, M):
    """把任务 x 插入路线 route 位置 q 带来的时间增加(分钟)。"""
    L = len(route)
    pred = route[q - 1] if q > 0 else M
    succ = route[q] if q < L else M
    return TAU_MIN * (D_ext[pred, x] + D_ext[x, succ] - D_ext[pred, succ]) + SERVICE_MIN


def _swap_delta(route, p, y, D_ext, M):
    """把路线 route 位置 p 的任务替换为 y 带来的时间变化(分钟)。"""
    L = len(route)
    x = route[p]
    prev = route[p - 1] if p > 0 else M
    nxt = route[p + 1] if p < L - 1 else M
    return TAU_MIN * (D_ext[prev, y] + D_ext[y, nxt] - D_ext[prev, x] - D_ext[x, nxt])


def _prefix(route, D_ext):
    """pref[i] = 前 i 个任务的内部相邻距离之和(pref[0]=pref[1]=0,pref[L]=总内部距离)。

    pref[i] = Σ_{t=0}^{i-2} D[route[t], route[t+1]]  (i≥2),即只含两端点都在前 i 个任务内的边。
    """
    L = len(route)
    pref = np.zeros(L + 1)
    if L > 1:
        r = np.asarray(route, dtype=np.int64)
        a = D_ext[r[:-1], r[1:]]          # a[t] = D[route[t], route[t+1]]
        pref[2:] = np.cumsum(a)
    return pref


def _concat_time(rA, p, rB, q, prA, prB, D_ext, M):
    """路线 rA[:p] + rB[q:] 的工作时间(分钟);空路线返回 0(调用方负责跳过)。"""
    LA, LB = len(rA), len(rB)
    n = p + (LB - q)
    if n == 0:
        return 0.0
    if p == 0:
        start, end = rB[q], rB[-1]
        internal = prB[LB] - prB[q + 1]            # rB[q:] 的内部距离
    elif q == LB:
        start, end = rA[0], rA[p - 1]
        internal = prA[p]
    else:
        start, end = rA[0], rB[-1]
        internal = prA[p] + D_ext[rA[p - 1], rB[q]] + (prB[LB] - prB[q + 1])
    return TAU_MIN * (D_ext[M, start] + internal + D_ext[end, M]) + SERVICE_MIN * n


def relocate_pass(routes, times, D_ext, M):
    """跨机 relocate:把最长机的一个任务搬到另一机的最佳位置(首改进)。"""
    N = len(routes)
    k = int(np.argmax(times))
    if len(routes[k]) <= 1:
        return False
    for p in range(len(routes[k])):
        x = routes[k][p]
        dk = _removal_delta(routes[k], p, D_ext, M)
        for r2 in range(N):
            if r2 == k:
                continue
            for q in range(len(routes[r2]) + 1):
                di = _insertion_delta(routes[r2], q, x, D_ext, M)
                nk, n2 = times[k] - dk, times[r2] + di
                if max(nk, n2) >= times[k] - 1e-9:
                    continue
                cand = list(times)
                cand[k], cand[r2] = nk, n2
                if max(cand) >= max(times) - 1e-9:
                    continue
                routes[k].pop(p)
                routes[r2].insert(q, x)
                times[k], times[r2] = nk, n2
                return True
    return False


def exchange_pass(routes, times, D_ext, M):
    """跨机 exchange:最长机与另一机交换一个任务(首改进)。"""
    N = len(routes)
    k = int(np.argmax(times))
    for p in range(len(routes[k])):
        for r2 in range(N):
            if r2 == k:
                continue
            for q in range(len(routes[r2])):
                dk = _swap_delta(routes[k], p, routes[r2][q], D_ext, M)
                d2 = _swap_delta(routes[r2], q, routes[k][p], D_ext, M)
                nk, n2 = times[k] + dk, times[r2] + d2
                if max(nk, n2) >= times[k] - 1e-9:
                    continue
                cand = list(times)
                cand[k], cand[r2] = nk, n2
                if max(cand) >= max(times) - 1e-9:
                    continue
                routes[k][p], routes[r2][q] = routes[r2][q], routes[k][p]
                times[k], times[r2] = nk, n2
                return True
    return False


def two_opt_star_pass(routes, times, D_ext, M):
    """跨机 2-opt*:切断最长机与另一机的两条边并交叉重连(首改进,不产生空路线)。"""
    N = len(routes)
    k = int(np.argmax(times))
    Lk = len(routes[k])
    prk = _prefix(routes[k], D_ext)
    for r2 in range(N):
        if r2 == k:
            continue
        L2 = len(routes[r2])
        pr2 = _prefix(routes[r2], D_ext)
        for p in range(Lk + 1):
            for q in range(L2 + 1):
                if (p == 0 and q == L2) or (p == Lk and q == 0):
                    continue                                # 会产生空路线
                t1 = _concat_time(routes[k], p, routes[r2], q, prk, pr2, D_ext, M)
                t2 = _concat_time(routes[r2], q, routes[k], p, pr2, prk, D_ext, M)
                if max(t1, t2) >= times[k] - 1e-9:
                    continue
                cand = list(times)
                cand[k], cand[r2] = t1, t2
                if max(cand) >= max(times) - 1e-9:
                    continue
                new_k = routes[k][:p] + routes[r2][q:]
                new_2 = routes[r2][:q] + routes[k][p:]
                routes[k], routes[r2] = new_k, new_2
                times[k], times[r2] = t1, t2
                return True
    return False


def intra_or_opt(route, D_ext, M, t_cur):
    """机内 Or-opt:把长度 2~3 的任务块搬到同机另一位置(首改进,循环至收敛)。"""
    L = len(route)
    improved = True
    while improved:
        improved = False
        if L < 4:
            return t_cur
        pref = _prefix(route, D_ext)
        for b in (2, 3):
            if L <= b:
                continue
            for p in range(L - b + 1):
                x, y = route[p], route[p + b - 1]
                internal = pref[p + b] - pref[p + 1]        # 块 [p..p+b-1] 的内部距离
                prev = route[p - 1] if p > 0 else M
                nxt = route[p + b] if p + b < L else M
                d_rem = TAU_MIN * (D_ext[prev, x] + internal + D_ext[y, nxt]
                                   - D_ext[prev, nxt]) + SERVICE_MIN * b
                rest = route[:p] + route[p + b:]
                Lr = len(rest)
                for q in range(Lr + 1):
                    pred = rest[q - 1] if q > 0 else M
                    succ = rest[q] if q < Lr else M
                    d_ins = TAU_MIN * (D_ext[pred, x] + internal + D_ext[y, succ]
                                       - D_ext[pred, succ]) + SERVICE_MIN * b
                    tn = t_cur - d_rem + d_ins
                    if tn < t_cur - 1e-9:
                        route[:] = rest[:q] + route[p:p + b] + rest[q:]
                        t_cur = tn
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return t_cur


def or_opt_pass(routes, times, D_ext, M):
    """跨机 Or-opt:把最长机长度 2~3 的任务块整体搬到另一机(首改进)。"""
    N = len(routes)
    k = int(np.argmax(times))
    Lk = len(routes[k])
    prk = _prefix(routes[k], D_ext)
    for b in (2, 3):
        if Lk <= b:
            continue
        for p in range(Lk - b + 1):
            x, y = routes[k][p], routes[k][p + b - 1]
            internal = prk[p + b] - prk[p + 1]      # 块 [p..p+b-1] 的内部距离
            prev = routes[k][p - 1] if p > 0 else M
            nxt = routes[k][p + b] if p + b < Lk else M
            dk = TAU_MIN * (D_ext[prev, x] + internal + D_ext[y, nxt] - D_ext[prev, nxt]) + SERVICE_MIN * b
            for r2 in range(N):
                if r2 == k:
                    continue
                r2r = routes[r2]
                L2 = len(r2r)
                pr2 = _prefix(r2r, D_ext)
                for q in range(L2 + 1):
                    pred = r2r[q - 1] if q > 0 else M
                    succ = r2r[q] if q < L2 else M
                    di = TAU_MIN * (D_ext[pred, x] + internal + D_ext[y, succ] - D_ext[pred, succ]) + SERVICE_MIN * b
                    nk, n2 = times[k] - dk, times[r2] + di
                    if max(nk, n2) >= times[k] - 1e-9:
                        continue
                    cand = list(times)
                    cand[k], cand[r2] = nk, n2
                    if max(cand) >= max(times) - 1e-9:
                        continue
                    block = routes[k][p:p + b]
                    del routes[k][p:p + b]
                    routes[r2][q:q] = block
                    times[k], times[r2] = nk, n2
                    return True
    return False


def local_search(routes, D_ext, M, max_passes=60):
    """对译码路线做局部搜索(2-opt / Or-opt / relocate / exchange / 2-opt*)。返回 (routes, times, makespan)。"""
    routes = [list(r) for r in routes]
    times = [route_time_full(r, D_ext, M) for r in routes]
    imp = True
    for _ in range(max_passes):
        if not imp:
            break
        imp = False
        for ri in range(len(routes)):
            t2 = two_opt_all(routes[ri], D_ext, M, times[ri])
            if t2 < times[ri] - 1e-9:
                times[ri] = t2
                imp = True
            t3 = intra_or_opt(routes[ri], D_ext, M, times[ri])
            if t3 < times[ri] - 1e-9:
                times[ri] = t3
                imp = True
        if relocate_pass(routes, times, D_ext, M):
            imp = True
        if exchange_pass(routes, times, D_ext, M):
            imp = True
        if or_opt_pass(routes, times, D_ext, M):
            imp = True
        if two_opt_star_pass(routes, times, D_ext, M):
            imp = True
    return routes, times, max(times)


# ----------------------------------------------------------------------
# 遗传算法
# ----------------------------------------------------------------------
class GA:
    """巨型路线 + 最优分割的混合遗传算法,求解给定 N 台机的 min-makespan 调度。"""

    def __init__(self, tasks, xy, N, seed=0, pop_size=80, elite=2, pc=0.9,
                 pm=(0.25, 0.12, 0.08), ls_every=5):
        # tasks: [(点下标, 第几次巡检), ...];xy: 点坐标数组
        self.M = len(tasks)
        D_ext, X = build_D_ext(tasks, xy)
        self.xy_t = X
        self.D_ext = D_ext
        self.D = D_ext[:self.M, :self.M]
        self.d0 = D_ext[self.M, :self.M]
        self.N = N
        self.pop_size = pop_size
        self.elite = elite
        self.pc = pc
        self.pm = pm
        self.ls_every = ls_every
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # 对称性破缺(副本重编号)所需的映射
        self.pt_of_task = np.array([t[0] for t in tasks], dtype=np.int64)
        self.task_id_of = {(int(t[0]), int(t[1])): i for i, t in enumerate(tasks)}

    # ---- 初始化 ----
    def _nn_tour(self):
        """从基地出发的最近邻巨型路线。"""
        remain = np.ones(self.M, dtype=bool)
        seq = np.empty(self.M, dtype=np.int64)
        cur = None
        for pos in range(self.M):
            if cur is None:
                best = int(np.argmin(self.d0))
            else:
                Drow = self.D[cur].copy()
                Drow[~remain] = np.inf
                best = int(np.argmin(Drow))
            seq[pos] = best
            remain[best] = False
            cur = best
        return seq

    def _sweep_tour(self):
        ang = np.arctan2(self.xy_t[:, 1], self.xy_t[:, 0])
        return np.argsort(ang, kind='stable').astype(np.int64)

    def init_pop(self):
        pop = [self._nn_tour(), self._sweep_tour()]
        for _ in range(self.pop_size - len(pop)):
            pop.append(self.rng.permutation(self.M).astype(np.int64))
        return pop

    # ---- 遗传算子 ----
    def eval(self, seq):
        # 对称性破缺:先把副本标签规范化(严格无损),再求最优分割
        seq[:] = canonicalize_seq(seq, self.pt_of_task, self.task_id_of)
        mk, cuts = split_minmax(seq, self.N, self.D, self.d0)
        return mk, cuts

    def ox(self, p1, p2):
        i, j = sorted(self.rng.choice(self.M, size=2, replace=False))
        child = np.full(self.M, -1, dtype=np.int64)
        seg = p1[i:j + 1]
        child[i:j + 1] = seg
        used = np.zeros(self.M, dtype=bool)
        used[seg] = True
        pos = (j + 1) % self.M
        for x in np.roll(p2, -(j + 1)):
            if not used[x]:
                child[pos] = x
                used[x] = True
                pos = (pos + 1) % self.M
        return child

    def mutate(self, c):
        r = self.rng
        if r.random() < self.pm[0]:
            a, b = r.choice(self.M, size=2, replace=False)
            c[a], c[b] = c[b], c[a]
        if r.random() < self.pm[1]:
            a, b = sorted(r.choice(self.M, size=2, replace=False))
            c[a:b + 1] = c[a:b + 1][::-1]
        if r.random() < self.pm[2]:
            a = int(r.integers(self.M))
            b = int(r.integers(self.M - 1))
            if b >= a:
                b += 1
            x = int(c[a])
            c2 = np.insert(np.delete(c, a), b, x)
            c[:] = c2

    def tournament(self, fits):
        idx = self.rng.choice(self.pop_size, size=3, replace=False)
        return int(min(idx, key=lambda i: fits[i][0]))

    def inject(self, pop, fits, routes):
        """把局部搜索改进后的路线(按长度降序拼接为巨型路线)注入种群,替换最差个体。"""
        concat = np.concatenate([np.asarray(r, dtype=np.int64)
                                 for r in sorted(routes, key=len, reverse=True)])
        j = int(np.argmax([f[0] for f in fits]))
        pop[j] = concat
        fits[j] = self.eval(concat)

    # ---- 主循环 ----
    def run(self, max_gens=1000, patience=250, restart_every=120, stop_if_le=None, log=None):
        log = log or (lambda *a, **k: None)
        pop = self.init_pop()
        fits = [self.eval(s) for s in pop]

        def best_of(pop, fits):
            i = int(np.argmin([f[0] for f in fits]))
            return i, fits[i][0], decode_routes(pop[i], fits[i][1], self.N)

        bi, best_mk, best_routes = best_of(pop, fits)
        sched_mk, sched_routes = best_mk, [r[:] for r in best_routes]

        # 对初代最优做局部搜索
        routes, times, mk = local_search([r[:] for r in sched_routes], self.D_ext, self.M)
        if mk < sched_mk - 1e-9:
            sched_mk, sched_routes = mk, [r[:] for r in routes]
            self.inject(pop, fits, sched_routes)

        no_imp = 0
        gen = 0
        while gen < max_gens:
            offspring = []
            for _ in range(self.pop_size - self.elite):
                a = self.tournament(fits)
                b = self.tournament(fits)
                c = self.ox(pop[a], pop[b]) if self.rng.random() < self.pc else pop[a].copy()
                self.mutate(c)
                offspring.append(c)
            ofits = [self.eval(c) for c in offspring]
            elite_idx = np.argsort([f[0] for f in fits])[:self.elite]
            pop = [pop[i] for i in elite_idx] + offspring
            fits = [fits[i] for i in elite_idx] + ofits

            bi, best_mk, best_routes = best_of(pop, fits)
            improved = False
            if best_mk < sched_mk - 1e-9:
                sched_mk, sched_routes = best_mk, [r[:] for r in best_routes]
                improved = True
            if best_mk < sched_mk - 1e-9 or gen % self.ls_every == 0:
                routes, times, mk = local_search([r[:] for r in best_routes], self.D_ext, self.M)
                if mk < sched_mk - 1e-9:
                    sched_mk, sched_routes = mk, [r[:] for r in routes]
                    self.inject(pop, fits, sched_routes)
                    improved = True
            no_imp = 0 if improved else no_imp + 1
            gen += 1
            if stop_if_le is not None and sched_mk <= stop_if_le:
                break
            if no_imp >= patience:
                break
            if restart_every and no_imp > 0 and no_imp % restart_every == 0:
                # 灾难式部分重随机化:保留最优 15%,其余重新随机,避免平台期
                keep = max(self.elite, self.pop_size * 15 // 100)
                order = np.argsort([f[0] for f in fits])
                kept = [pop[i] for i in order[:keep]]
                kfits = [fits[i] for i in order[:keep]]
                fresh = [self.rng.permutation(self.M).astype(np.int64)
                         for _ in range(self.pop_size - keep)]
                ffits = [self.eval(c) for c in fresh]
                pop = kept + fresh
                fits = kfits + ffits
        return dict(mk=sched_mk, routes=sched_routes, best_fit=best_mk,
                    gens=gen, seed=self.seed)


# ----------------------------------------------------------------------
# 独立校验
# ----------------------------------------------------------------------
def validate_schedule(M, routes, D_ext, horizon=None):
    """独立复核调度:任务覆盖恰一次、逐机时间由坐标重算、Tmax/Tmin、时限。
    routes 为任务 id 列表;基地节点编号 = M。返回校验报告 dict。"""
    used = np.concatenate([np.asarray(r, dtype=np.int64) for r in routes]) if routes \
        else np.array([], dtype=np.int64)
    ok_cov = (sorted(used.tolist()) == list(range(M)))
    tms = [route_time_full(r, D_ext, M) for r in routes]
    tmax, tmin = max(tms), min(tms)
    ok_h = (tmax <= horizon + 1e-6) if horizon is not None else None
    return dict(coverage_ok=ok_cov, M=M, used=len(used), route_times=tms,
                Tmax_min=tmax, Tmin_min=tmin, horizon_ok=ok_h)
