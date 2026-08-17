# -*- coding: utf-8 -*-
"""问题1 第二层:蚁群算法(ACO)在固定 N=Nmin 下最小化全局完成时间。

设计(巨型路线 + 最优分割的 memetic ACO):
- 蚂蚁构造覆盖 M 个任务的排列(巨型路线);转移用伪随机比例规则
  (概率 q0 贪心取 τ^α·η^β 最大者,否则轮盘赌),候选表 = 当前点近邻表;
- 适应度 = split_minmax(排列, N):对该排列**精确**的最优 N 段划分,
  即 min-makespan(平衡问题交给 DP,蚂蚁专注排序——min-max mTSP 用 ACO 的关键设计);
- 信息素:蒸发率 ρ;迭代最优 + 全局最优双重沉积;初值 τ0 = 1/C_nn
  (最近邻巨型路线的单机总时长标度);支持以已知解(第二层 GA 解)铺设初始信息素;
- 拉马克式局部搜索:每 ls_every 代对最优解译码路线做 deep_ls
  (2-opt/Or-opt/relocate/exchange/2-opt*/喷射链),改进后回写并加强其信息素;
- 固定 N,直接最小化 makespan(同值以总时长破平)。随机种子固定,可复现。
"""
import numpy as np

from common import TAU_MIN, SERVICE_MIN
from ga_minmax import (build_D_ext, canonicalize_seq, decode_routes, split_minmax)
from ga_cluster import deep_ls

__all__ = ['ACO']


class ACO:
    def __init__(self, tasks, xy, N, seed=0, n_ants=30, alpha=1.0, beta=2.0,
                 rho=0.1, q0=0.9, nn_size=25, ls_every=10, seed_routes=None):
        self.M = len(tasks)
        self.D_ext, self.xy_t = build_D_ext(tasks, xy)
        self.D = self.D_ext[:self.M, :self.M]
        self.d0 = self.D_ext[self.M, :self.M]
        self.N = N
        self.n_ants = n_ants
        self.alpha, self.beta, self.rho, self.q0 = alpha, beta, rho, q0
        self.nn_size = nn_size
        self.ls_every = ls_every
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        # 对称性破缺(副本按点选代表)所需的映射
        self.pt_of_task = np.array([t[0] for t in tasks], dtype=np.int64)
        self.task_id_of = {(int(t[0]), int(t[1])): i for i, t in enumerate(tasks)}
        self.n_points = int(self.pt_of_task.max()) + 1
        self.k_of_point = np.zeros(self.n_points, dtype=int)
        for p in self.pt_of_task:
            self.k_of_point[p] += 1

        # 启发信息(距离 + eps,避免同坐标副本除零)
        eps = 0.5
        self.eta = 1.0 / (self.D + eps)
        np.fill_diagonal(self.eta, 0.0)

        # 信息素(任务-任务)
        nn_tour = canonicalize_seq(self._nn_tour(), self.pt_of_task, self.task_id_of)
        c_nn = self._tour_cost(nn_tour)
        self.tau0 = 1.0 / c_nn
        self.tau = np.full((self.M, self.M), self.tau0)
        np.fill_diagonal(self.tau, 0.0)

        # 信息素热启动:用已知解(第二层 GA 解,先规范化)的连续弧铺设 2×τ0
        if seed_routes:
            canon = []
            cnt = {}
            for r in seed_routes:
                rr = []
                for t in r:
                    p = int(self.pt_of_task[t])
                    c = cnt.get(p, 0) + 1
                    cnt[p] = c
                    rr.append(self.task_id_of[(p, c)])
                canon.append(rr)
            seq = np.concatenate([np.asarray(r, dtype=np.int64) for r in canon])
            for a, b in zip(seq[:-1], seq[1:]):
                self.tau[a, b] += 2.0 * self.tau0

    # ---- 构造(对称性破缺:按物理点选代表,每点取剩余最小副本号)----
    def _nn_tour(self):
        remain = np.ones(self.M, bool)
        seq = np.empty(self.M, np.int64)
        cur = None
        for pos in range(self.M):
            cand = np.where(remain)[0]
            if cur is None:
                j = cand[np.argmin(self.d0[cand])]
            else:
                j = cand[np.argmin(self.D[cur][cand])]
            seq[pos] = j
            remain[j] = False
            cur = j
        return seq

    def _tour_cost(self, seq):
        """巨型路线的单机总时长(信息素标度用)。"""
        L = self.d0[seq[0]] + self.d0[seq[-1]]
        if self.M > 1:
            L += self.D[seq[:-1], seq[1:]].sum()
        return TAU_MIN * L + SERVICE_MIN * self.M

    def construct(self):
        """按点构造规范序列:候选 = 各物理点的剩余最小副本(一个点一个代表),
        选中的代表即该点的下一个副本号 —— 天然满足副本按出现顺序编号。"""
        seq = np.empty(self.M, np.int64)
        minv = np.ones(self.n_points, dtype=int)      # 每点下一个可用副本号(1 起)
        cand_p = np.where(minv <= self.k_of_point)[0]
        p0 = int(self.rng.choice(cand_p))
        cur = self.task_id_of[(p0, 1)]
        seq[0] = cur
        minv[p0] += 1
        for pos in range(1, self.M):
            cand_p = np.where(minv <= self.k_of_point)[0]
            reps = [self.task_id_of[(int(p), int(minv[p]))] for p in cand_p]
            if len(reps) == 1:
                nxt = reps[0]
            else:
                t = self.tau[cur][reps] ** self.alpha * self.eta[cur][reps] ** self.beta
                if self.rng.random() < self.q0:
                    nxt = reps[int(np.argmax(t))]
                else:
                    pr = t / t.sum()
                    nxt = reps[int(self.rng.choice(len(reps), p=pr))]
            seq[pos] = nxt
            minv[self.pt_of_task[nxt]] += 1
            cur = nxt
        return seq

    def _deposit(self, seq, weight):
        if self.M > 1:
            a, b = seq[:-1], seq[1:]
            np.add.at(self.tau, (a, b), weight)

    # ---- 主循环 ----
    def run(self, iters=400, log=None):
        log = log or (lambda *a, **k: None)
        best_seq, best_split_mk = None, np.inf
        best_routes, best_mk, best_tsum = None, np.inf, np.inf
        for it in range(iters):
            it_best_seq, it_best_mk = None, np.inf
            for _ in range(self.n_ants):
                seq = self.construct()
                mk, cuts = split_minmax(seq, self.N, self.D, self.d0)
                if mk < it_best_mk:
                    it_best_mk, it_best_seq = mk, seq
            if it_best_seq is not None:
                self._deposit(it_best_seq, 1.0 / it_best_mk)
                if it_best_mk < best_split_mk - 1e-9:
                    best_split_mk = it_best_mk
                    best_seq = it_best_seq.copy()
                    self._deposit(best_seq, 1.0 / best_split_mk)
            self.tau *= (1.0 - self.rho)          # 蒸发
            # 拉马克局部搜索
            if best_seq is not None and (it % self.ls_every == 0 or it == iters - 1):
                mk, cuts = split_minmax(best_seq, self.N, self.D, self.d0)
                routes = decode_routes(best_seq, cuts, self.N)
                routes, times, mk2 = deep_ls(routes, self.D_ext, self.M,
                                             rng=self.rng, seed=self.seed + it)
                if mk2 < best_mk - 1e-9 or \
                   (abs(mk2 - best_mk) < 1e-9 and sum(times) < best_tsum - 1e-9):
                    best_routes = [r[:] for r in routes]
                    best_mk, best_tsum = mk2, sum(times)
                    seq2 = np.concatenate([np.asarray(r, np.int64) for r in
                                           sorted(routes, key=len, reverse=True)])
                    seq2 = canonicalize_seq(seq2, self.pt_of_task, self.task_id_of)
                    best_seq = seq2
                    self._deposit(seq2, 2.0 / max(best_mk, 1e-9))
        if best_routes is None:
            mk, cuts = split_minmax(best_seq, self.N, self.D, self.d0)
            routes = decode_routes(best_seq, cuts, self.N)
            routes, times, best_mk = deep_ls(routes, self.D_ext, self.M,
                                             rng=self.rng, seed=self.seed + 1)
            best_routes, best_tsum = [r[:] for r in routes], sum(times)
        return dict(mk=best_mk, tsum=best_tsum, routes=best_routes, gens=iters,
                    seed=self.seed)
