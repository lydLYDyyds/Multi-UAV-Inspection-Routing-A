# -*- coding: utf-8 -*-
"""问题1 第二层(1.2):簇编码遗传算法 —— 在 N=Nmin 下最小化全局完成时间。

染色体 = N 条有序路线(簇)的列表:划分(任务→簇归属)与簇内访问顺序同时显式编码,
通过簇间与簇内的变异/交换协同进化。

- 簇间算子(划分层,本层重点):
    交叉   路线继承:继承父代1最长路线 + 父代2不冲突路线,冲突任务剔除后最便宜插入修复;
    变异a  跨簇交换:两簇各取一任务互换(仅接受 makespan 不上升);
    变异b  跨簇搬移:最长簇一个任务搬到另一簇最佳位置(仅接受 makespan 不上升);
    变异c  簇级重组:最小簇并入其他簇 + 最大簇按最远两点种子拆二(保持 N 个非空簇);
- 簇内算子(布线层):
    变异   簇内交换 / 区段反转;
    引擎   最近邻贪心构造(nn_order)+ 2-opt / Or-opt 确定收敛(two_opt_all, intra_or_opt);
    精修   可选簇内模拟退火(intra_sa,交换/反转移动 + Metropolis 准则);
- 拉马克式:对最优解复用 ga_minmax.local_search
  (机内 2-opt/Or-opt、跨簇 relocate/exchange/块搬迁/2-opt*),只接受 makespan 严格下降;
- 精英保留、三元锦标赛(主目标 makespan,副目标总时长破平)、耐心早停、灾难式重启。

输入 seed_routes 时,以第一层(ga_p1_*.json)的最终划分为种子初始化,实现两层接力。
"""
import numpy as np

from ga_minmax import (build_D_ext, route_time_full, local_search, two_opt_all,
                       intra_or_opt, _insertion_delta, _removal_delta, _swap_delta)

__all__ = ['ClusterGA', 'intra_sa', 'nn_order', 'ils', 'deep_ls', 'double_bridge',
           'ejection_pass']


def double_bridge(route, rng):
    """double-bridge 扰动:四段重连 A B C D -> A C B D(不反转,经典 TSP 逃逸)。"""
    L = len(route)
    if L < 8:
        return
    i, j, k = sorted(rng.choice(range(1, L), size=3, replace=False))
    route[:] = route[:i] + route[j:k] + route[i:j] + route[k:]


def ejection_pass(routes, times, D_ext, M, max_depth=5, n_start=24, seed=0):
    """深度受限喷射链:从最长簇喷出一个任务,链式搬移腾挪,取链上 makespan 最优状态。

    链:移除 x(最长簇)→ 把 x 插入另一簇最优位置 → 从该簇喷出 y → 插入第三簇 ...,
    最后把链尾任务插回最长簇;每个中间状态任务多重集不变,仅最终 makespan 下降才接受。
    返回是否改进(routes/times 原地更新)。
    """
    rng = np.random.default_rng(seed)
    N = len(routes)
    k = int(np.argmax(times))
    if len(routes[k]) <= 1:
        return False
    base_mk = max(times)
    best, best_mk = None, base_mk
    starts = rng.choice(len(routes[k]), size=min(n_start, len(routes[k])), replace=False)
    for p0 in starts:
        cur = [r[:] for r in routes]
        t = list(times)
        x = cur[k][p0]
        d_rem = _removal_delta(cur[k], p0, D_ext, M)
        cur[k].pop(p0)
        t[k] -= d_rem
        from_r, cur_x = k, x
        for _ in range(max_depth):
            best_r2, best_q, best_d = None, None, np.inf
            best_mk_cand = np.inf
            mk_now = max(t)
            for r2 in range(N):
                if r2 == from_r:
                    continue
                for q in range(len(cur[r2]) + 1):
                    d = _insertion_delta(cur[r2], q, cur_x, D_ext, M)
                    mk_cand = max(t[r2] + d, mk_now)
                    if mk_cand < best_mk_cand - 1e-9 or \
                       (abs(mk_cand - best_mk_cand) < 1e-9 and d < best_d - 1e-9):
                        best_mk_cand, best_r2, best_q, best_d = mk_cand, r2, q, d
            if best_r2 is None:
                break
            cur[best_r2].insert(best_q, cur_x)
            t[best_r2] += best_d
            if len(cur[best_r2]) > 1:
                best_p, best_d_rem = None, np.inf
                for p in range(len(cur[best_r2])):
                    d = _removal_delta(cur[best_r2], p, D_ext, M)
                    if d < best_d_rem - 1e-9:
                        best_d_rem, best_p = d, p
                cur_x = cur[best_r2][best_p]
                cur[best_r2].pop(best_p)
                t[best_r2] -= best_d_rem
                from_r = best_r2
            else:
                cur_x = None
                break
        if cur_x is not None:
            best_q, best_d = None, np.inf
            for q in range(len(cur[k]) + 1):
                d = _insertion_delta(cur[k], q, cur_x, D_ext, M)
                if d < best_d - 1e-9:
                    best_d, best_q = d, q
            cur[k].insert(best_q, cur_x)
            t[k] += best_d
        mk = max(t)
        if mk < best_mk - 1e-9:
            best_mk, best = mk, (cur, t)
    if best is not None:
        routes[:] = [r[:] for r in best[0]]
        times[:] = best[1]
        return True
    return False


def deep_ls(routes, D_ext, M, max_rounds=6, rng=None, seed=0):
    """深度局部搜索:local_search 收敛 + 喷射链,循环至无改进。"""
    rng = rng or np.random.default_rng(seed)
    cur = [r[:] for r in routes]
    cur, times, mk = local_search(cur, D_ext, M)
    for _ in range(max_rounds):
        if not ejection_pass(cur, times, D_ext, M, max_depth=5, n_start=24,
                             seed=int(rng.integers(1_000_000))):
            break
        cur, times, mk = local_search(cur, D_ext, M)
    return cur, times, mk


def ils(routes, D_ext, M, iters=400, perturb_k=3, seed=0, anneal=True):
    """迭代局部搜索:扰动(double-bridge + 跨簇交换/搬移)+ deep_ls 收敛,
    模拟退火式接受劣解作为游走点(记录的最优不受影响)。返回 (routes, times, makespan)。"""
    rng = np.random.default_rng(seed)
    walk = [r[:] for r in routes]
    walk, wtimes, wmk = deep_ls(walk, D_ext, M, rng=rng, seed=seed)
    best_routes, best_times, best_mk = [r[:] for r in walk], wtimes, wmk
    T0 = 6.0
    for it in range(iters):
        cand = [r[:] for r in walk]
        for _ in range(perturb_k):
            if rng.random() < 0.6 and len(cand) > 1:
                a, b = rng.choice(len(cand), size=2, replace=False)
                if cand[a] and cand[b]:
                    x = int(rng.choice(cand[a]))
                    y = int(rng.choice(cand[b]))
                    cand[a][cand[a].index(x)] = y
                    cand[b][cand[b].index(y)] = x
            else:
                a, b = rng.choice(len(cand), size=2, replace=False)
                if cand[a] and len(cand[a]) > 1:
                    x = int(rng.choice(cand[a]))
                    cand[a].remove(x)
                    cand[b].insert(int(rng.integers(len(cand[b]) + 1)), x)
        kk = int(np.argmax([route_time_full(r, D_ext, M) for r in cand]))
        double_bridge(cand[kk], rng)
        cand, times, mk = deep_ls(cand, D_ext, M, rng=rng, seed=seed + it + 1)
        better = (mk < best_mk - 1e-9) or \
                 (abs(mk - best_mk) < 1e-9 and sum(times) < sum(best_times) - 1e-9)
        if better:
            best_routes, best_times, best_mk = [r[:] for r in cand], times, mk
            walk = [r[:] for r in cand]
        elif anneal and mk > best_mk + 1e-9:
            T = T0 * (1.0 - it / iters) + 0.2
            if rng.random() < np.exp(-(mk - best_mk) / T):
                walk = [r[:] for r in cand]
        else:
            walk = [r[:] for r in cand]
    return best_routes, best_times, best_mk


# ----------------------------------------------------------------------
# 簇内引擎:贪心构造与模拟退火
# ----------------------------------------------------------------------
def nn_order(subset, D_ext, M):
    """对任务子集构造最近邻路线(自基地出发的贪心)。subset:任务 id 列表。"""
    if not subset:
        return []
    rem = list(subset)
    route = []
    cur = M
    while rem:
        i = int(np.argmin([D_ext[cur, x] for x in rem]))
        route.append(rem.pop(i))
        cur = route[-1]
    return route


def intra_sa(route, D_ext, M, iters=1500, T0=10.0, rng=None):
    """簇内模拟退火(交换/反转两种移动,Metropolis 准则)。返回 (最终时间, 路线)。"""
    rng = rng or np.random.default_rng(0)
    L = len(route)
    if L < 4:
        return route_time_full(route, D_ext, M), route[:]
    cur = route[:]
    t = route_time_full(cur, D_ext, M)
    for it in range(iters):
        T = T0 * (1.0 - it / iters) + 1e-3
        if rng.random() < 0.5:
            a, b = rng.choice(L, size=2, replace=False)
            cand = cur[:]
            cand[a], cand[b] = cand[b], cand[a]
        else:
            a, b = sorted(rng.choice(L, size=2, replace=False))
            cand = cur[:a] + cur[a:b + 1][::-1] + cur[b + 1:]
        tn = route_time_full(cand, D_ext, M)
        if tn < t - 1e-12 or rng.random() < np.exp(-(tn - t) / max(T, 1e-9)):
            cur, t = cand, tn
    return t, cur


# ----------------------------------------------------------------------
# 簇编码遗传算法
# ----------------------------------------------------------------------
class ClusterGA:
    """N 条路线的显式簇编码 GA,最小化 makespan(副目标:总时长)。"""

    def __init__(self, tasks, xy, N, seed=0, pop_size=40, elite=2, pc=0.85,
                 ls_every=5, seed_routes=None, sa_intra=True):
        self.M = len(tasks)
        self.D_ext, self.xy_t = build_D_ext(tasks, xy)
        self.N = N
        self.pop_size = pop_size
        self.elite = elite
        self.pc = pc
        self.ls_every = ls_every
        self.sa_intra = sa_intra
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.seed_routes = seed_routes
        # 对称性破缺(副本重编号)所需的映射
        self.pt_of_task = np.array([t[0] for t in tasks], dtype=np.int64)
        self.task_id_of = {(int(t[0]), int(t[1])): i for i, t in enumerate(tasks)}

    # ---- 个体工具 ----
    def times_of(self, routes):
        return [route_time_full(r, self.D_ext, self.M) for r in routes]

    def _canonicalize(self, routes):
        """按路线顺序全局重编号同点副本(严格无损:只改标签,不改变任何物理路线)。"""
        cnt = {}
        for r in routes:
            for i, t in enumerate(r):
                p = int(self.pt_of_task[t])
                c = cnt.get(p, 0) + 1
                cnt[p] = c
                r[i] = self.task_id_of[(p, c)]
        return routes

    def fitness(self, routes):
        self._canonicalize(routes)
        ts = self.times_of(routes)
        return (max(ts), sum(ts))

    def is_valid(self, routes):
        flat = sorted(x for r in routes for x in r)
        return all(len(r) > 0 for r in routes) and flat == list(range(self.M))

    # ---- 初始化 ----
    def _random_assign(self):
        perm = self.rng.permutation(self.M).tolist()
        cuts = sorted(self.rng.choice(range(1, self.M), size=self.N - 1, replace=False))
        bounds = [0] + cuts + [self.M]
        groups = [perm[bounds[i]:bounds[i + 1]] for i in range(self.N)]
        return [nn_order(g, self.D_ext, self.M) for g in groups]

    def _perturb(self, routes, k=3):
        routes = [r[:] for r in routes]
        for _ in range(k):
            if self.rng.random() < 0.5 and len(routes) > 1:
                a, b = self.rng.choice(len(routes), size=2, replace=False)
                if routes[a] and routes[b]:
                    x = int(self.rng.choice(routes[a]))
                    y = int(self.rng.choice(routes[b]))
                    routes[a][routes[a].index(x)] = y
                    routes[b][routes[b].index(y)] = x
            else:
                r = int(self.rng.choice(len(routes)))
                if len(routes[r]) > 1:
                    a, b = sorted(self.rng.choice(len(routes[r]), size=2, replace=False))
                    routes[r][a:b + 1] = routes[r][a:b + 1][::-1]
        return routes

    def init_pop(self):
        pop = []
        if self.seed_routes:
            pop.append([r[:] for r in self.seed_routes])
            for _ in range(self.pop_size * 40 // 100):
                pop.append(self._perturb(self.seed_routes, k=int(self.rng.integers(1, 5))))
        while len(pop) < self.pop_size:
            pop.append(self._random_assign())
        return pop[:self.pop_size]

    # ---- 交叉:路线继承 ----
    def crossover(self, p1, p2):
        ts1 = self.times_of(p1)
        k = int(np.argmax(ts1))
        child = [p1[k][:]]
        used = set(child[0])
        for r in p2:
            if len(child) >= self.N:
                break
            if used.isdisjoint(r):
                child.append(r[:])
                used |= set(r)
        while len(child) < self.N:
            child.append([])
        remaining = [t for t in range(self.M) if t not in used]
        # 空簇先各补一个距基地最近的任务,保证 N 个非空簇
        for r in child:
            if not r and remaining:
                i = int(np.argmin([self.D_ext[self.M, t] for t in remaining]))
                r.append(remaining.pop(i))
        # 其余任务最便宜插入
        for x in remaining:
            best_r, best_q, best_d = None, None, np.inf
            for ri, r in enumerate(child):
                L = len(r)
                for q in range(L + 1):
                    d = _insertion_delta(r, q, x, self.D_ext, self.M)
                    if d < best_d:
                        best_r, best_q, best_d = ri, q, d
            child[best_r].insert(best_q, x)
        return child

    # ---- 变异(簇间 + 簇内)----
    def mutate(self, routes):
        rng = self.rng
        # 1) 簇内:交换 / 反转(无条件接受,提供多样性)
        if rng.random() < 0.25:
            r = int(rng.choice(len(routes)))
            if len(routes[r]) > 1:
                a, b = rng.choice(len(routes[r]), size=2, replace=False)
                routes[r][a], routes[r][b] = routes[r][b], routes[r][a]
        if rng.random() < 0.12:
            r = int(rng.choice(len(routes)))
            if len(routes[r]) > 1:
                a, b = sorted(rng.choice(len(routes[r]), size=2, replace=False))
                routes[r][a:b + 1] = routes[r][a:b + 1][::-1]
        # 2) 簇间:交换(仅接受 makespan 不上升)
        if rng.random() < 0.25 and len(routes) > 1:
            ts = self.times_of(routes)
            a, b = rng.choice(len(routes), size=2, replace=False)
            if routes[a] and routes[b]:
                x = int(rng.choice(routes[a]))
                y = int(rng.choice(routes[b]))
                da = _swap_delta(routes[a], routes[a].index(x), y, self.D_ext, self.M)
                db = _swap_delta(routes[b], routes[b].index(y), x, self.D_ext, self.M)
                cand = list(ts)
                cand[a], cand[b] = ts[a] + da, ts[b] + db
                if max(cand) <= max(ts) + 1e-9:
                    routes[a][routes[a].index(x)] = y
                    routes[b][routes[b].index(y)] = x
        # 3) 簇间:最长簇搬移一个任务到另一簇最佳位置(仅接受 makespan 不上升)
        if rng.random() < 0.15 and len(routes) > 1:
            ts = self.times_of(routes)
            k = int(np.argmax(ts))
            if len(routes[k]) > 1:
                p = int(rng.integers(len(routes[k])))
                x = routes[k][p]
                dk = _removal_delta(routes[k], p, self.D_ext, self.M)
                others = [r for r in range(len(routes)) if r != k]
                r2 = int(rng.choice(others))
                best_q, best_d = None, np.inf
                for q in range(len(routes[r2]) + 1):
                    d = _insertion_delta(routes[r2], q, x, self.D_ext, self.M)
                    if d < best_d:
                        best_q, best_d = q, d
                cand = list(ts)
                cand[k], cand[r2] = ts[k] - dk, ts[r2] + best_d
                if max(cand) <= max(ts) + 1e-9:
                    routes[k].pop(p)
                    routes[r2].insert(best_q, x)
        # 4) 簇级重组:最小簇并入其他簇 + 最大簇按最远两点种子拆二(保持 N 个非空簇)
        if rng.random() < 0.06 and len(routes) > 1:
            ts = self.times_of(routes)
            i_min = int(np.argmin(ts))
            i_max = int(np.argmax(ts))
            if i_min != i_max and len(routes[i_max]) > 1:
                moved = routes[i_min]
                for x in moved:
                    best_r, best_q, best_d = None, None, np.inf
                    for ri, r in enumerate(routes):
                        if ri == i_min:
                            continue
                        for q in range(len(r) + 1):
                            d = _insertion_delta(r, q, x, self.D_ext, self.M)
                            if d < best_d:
                                best_r, best_q, best_d = ri, q, d
                    routes[best_r].insert(best_q, x)
                routes[i_min] = []
                big = routes[i_max]
                i1, i2, best_dd = 0, 0, -1.0
                for a in range(len(big)):
                    for b in range(a + 1, len(big)):
                        dd = self.D_ext[big[a], big[b]]
                        if dd > best_dd:
                            best_dd, i1, i2 = dd, a, b
                g1 = nn_order([big[i1]], self.D_ext, self.M)
                g2 = nn_order([big[i2]], self.D_ext, self.M)
                for t in big:
                    if t == big[i1] or t == big[i2]:
                        continue
                    d1 = _insertion_delta(g1, len(g1), t, self.D_ext, self.M)
                    d2 = _insertion_delta(g2, len(g2), t, self.D_ext, self.M)
                    (g1 if d1 <= d2 else g2).append(t)
                routes[i_max] = g1
                routes[i_min] = g2

    def tournament(self, fits):
        idx = self.rng.choice(self.pop_size, size=3, replace=False)
        return int(min(idx, key=lambda i: fits[i]))

    # ---- 主循环 ----
    def run(self, max_gens=500, patience=200, restart_every=100, log=None):
        log = log or (lambda *a, **k: None)
        pop = self.init_pop()
        fits = [self.fitness(r) for r in pop]

        def better(a, b):
            return a[0] < b[0] - 1e-9 or (abs(a[0] - b[0]) < 1e-9 and a[1] < b[1] - 1e-9)

        bi = min(range(len(fits)), key=lambda i: fits[i])
        best = [r[:] for r in pop[bi]]
        best_fit = fits[bi]
        routes, times, mk = local_search([r[:] for r in best], self.D_ext, self.M)
        if better((mk, sum(times)), best_fit):
            best, best_fit = [r[:] for r in routes], (mk, sum(times))

        no_imp = 0
        gen = 0
        while gen < max_gens:
            offspring = []
            for _ in range(self.pop_size - self.elite):
                a = self.tournament(fits)
                b = self.tournament(fits)
                c = self.crossover(pop[a], pop[b]) if self.rng.random() < self.pc \
                    else [r[:] for r in pop[a]]
                self.mutate(c)
                offspring.append(c)
            ofits = [self.fitness(c) for c in offspring]
            elite_idx = sorted(range(len(fits)), key=lambda i: fits[i])[:self.elite]
            pop = [pop[i] for i in elite_idx] + offspring
            fits = [fits[i] for i in elite_idx] + ofits

            bi = min(range(len(fits)), key=lambda i: fits[i])
            improved = False
            if better(fits[bi], best_fit):
                best = [r[:] for r in pop[bi]]
                best_fit = fits[bi]
                improved = True
            if improved or gen % self.ls_every == 0:
                routes, times, mk = local_search([r[:] for r in pop[bi]], self.D_ext, self.M)
                if better((mk, sum(times)), best_fit):
                    best = [r[:] for r in routes]
                    best_fit = (mk, sum(times))
                    improved = True
            no_imp = 0 if improved else no_imp + 1
            gen += 1
            if no_imp >= patience:
                break
            if restart_every and no_imp > 0 and no_imp % restart_every == 0:
                keep = max(self.elite, self.pop_size * 15 // 100)
                order = sorted(range(len(fits)), key=lambda i: fits[i])
                kept = [pop[i] for i in order[:keep]]
                kfits = [fits[i] for i in order[:keep]]
                fresh = [self._random_assign() for _ in range(self.pop_size - keep)]
                ffits = [self.fitness(c) for c in fresh]
                pop = kept + fresh
                fits = kfits + ffits

        # 末段精修:局部搜索 + 簇内模拟退火(仅接受改善,保证单调不劣)
        routes, times, mk = local_search([r[:] for r in best], self.D_ext, self.M)
        if better((mk, sum(times)), best_fit):
            best, best_fit = [r[:] for r in routes], (mk, sum(times))
        if self.sa_intra:
            cand = [r[:] for r in best]
            for r in cand:
                _, r2 = intra_sa(r, self.D_ext, self.M, rng=self.rng)
                r[:] = r2
            routes, times, mk = local_search(cand, self.D_ext, self.M)
            if better((mk, sum(times)), best_fit):
                best, best_fit = [r[:] for r in routes], (mk, sum(times))
        return dict(mk=best_fit[0], tsum=best_fit[1], routes=best, gens=gen, seed=self.seed)
