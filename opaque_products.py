import numpy as np
import itertools

from utils import unimodal_maximize


class ChoiceModel:
    def __init__(self, v_func, r):
        if np.isscalar(r):
            r = np.array([r], dtype=float)
        r = np.array(r, dtype=float)
        assert np.all(r >= 0)
        self.r = np.nan_to_num(np.array(r))
        self.v_func = v_func  # valuation of each product relative to outside option
        self.n = len(r)

    def set_r(self, r):
        if np.isscalar(r):
            r = np.array([r], dtype=float)
        r = np.array(r, dtype=float)
        assert len(r) == self.n
        assert np.all(r >= 0)
        self.r = np.nan_to_num(r)

    @property
    def r0(self):
        return np.r_[self.r, 0]

    def get_valuations(self):
        return self.v_func()

    def simulate(self, rho, iters):
        revenue_trad = 0
        revenue_averse = 0
        revenue_neutral = 0
        sales_trad = np.zeros_like(self.r0)
        sales_averse = np.zeros_like(self.r0)
        sales_neutral = np.zeros_like(self.r0)
        for _ in range(iters):
            vals = self.get_valuations()
            utilities = vals - self.r
            utility_neutral = np.mean(vals) - rho  # Risk Neutral
            utility_averse = np.min(vals) - rho  # Risk Averse

            u_max = utilities.max()
            u0_max = max(u_max, 0)
            i_max = utilities.argmax() if u_max > 0 else -1
            r_max = self.r0[i_max]
            revenue_trad += r_max
            sales_trad[i_max] += 1
            if u0_max > utility_averse:
                revenue_averse += r_max
                sales_averse[i_max] += 1
                if u0_max > utility_neutral:
                    revenue_neutral += r_max
                    sales_neutral[i_max] += 1
                else:
                    revenue_neutral += rho
            else:
                revenue_averse += rho
                revenue_neutral += rho
        return {
            "trad": {
                "prob": sales_trad / iters,
                "rev": revenue_trad / iters
            },
            "averse": {
                "prob": sales_averse / iters,
                "rev": revenue_averse / iters
            },
            "neutral": {
                "prob": sales_neutral / iters,
                "rev": revenue_neutral / iters
            }
        }


class MNL(ChoiceModel):
    def __init__(self, v, r):
        if np.isscalar(v):
            v = np.array([v], dtype=float)
        if np.isscalar(r):
            r = np.array([r], dtype=float)
        assert len(v) == len(r)
        v = np.array(v, dtype=float)
        assert np.all(v >= 0)
        self.v = v
        self.n = len(v)
        super(MNL, self).__init__(
                v_func=lambda: np.random.gumbel(loc=self.v, scale=1) - np.random.gumbel(loc=0, scale=1), r=r)

    @property
    def a(self):
        return np.exp(self.v - self.r)

    @property
    def v0(self):
        return np.r_[self.v, 0]

    @property
    def a0(self):
        return np.exp(self.v0 - self.r0)

    def prob_trad(self, assortment=None, r=None):
        r_old = None
        if r is not None:
            r_old = np.copy(self.r)
            self.set_r(r)
        _a0 = self.a0
        if assortment is None:
            assortment = np.ones(self.n)
        assortment = np.r_[np.array(assortment), 1.]
        if r is not None:
            self.set_r(r_old)
        return _a0 * assortment / np.sum(_a0 * assortment)

    def prob_opq(self, rho, assortment=None, r=None):
        r_old = None
        if r is not None:
            r_old = np.copy(self.r)
            self.set_r(r)
        if assortment is None:
            assortment = np.ones(self.n)
        asst_bool = np.array(assortment, dtype=bool)
        rho = np.nan_to_num(rho)
        # print(rho, asst_bool, self.r, np.min(self.r[asst_bool]))
        assert rho <= np.min(self.r[asst_bool])
        asst_size = int(np.sum(assortment))
        index_sets = itertools.product([0, 1], repeat=asst_size)
        probs = np.zeros(self.n + 2)
        for index_set in index_sets:
            if not np.any(index_set):
                continue
            index_arr = np.zeros(self.n)
            index_arr[asst_bool] = np.array(index_set)
            set_length = np.sum(index_arr)
            _r = self.r * (1 - index_arr) + rho * index_arr
            probs_ = self.prob_trad(assortment, r=_r)
            index_arr = np.r_[index_arr, 0]
            probs[1:] += (-1) ** (set_length % 2 + 1) * probs_ * (1 - index_arr)
            probs[0] += (-1) ** (set_length % 2 + 1) * np.sum(probs_ * index_arr)
        if r is not None:
            self.set_r(r_old)
        return probs

    def revenue_trad(self, assortment=None, r=None):
        probs = self.prob_trad(assortment, r)
        if r is None:
            r = self.r
        return np.sum(r * probs[:-1])

    def revenue_opq(self, rho, assortment=None, r=None):
        probs = self.prob_opq(rho, assortment, r)
        if r is None:
            r = self.r
        return np.sum(np.r_[rho, r] * probs[:-1])

    def trad_single_price_opt(self, i, r=None, lb=0, ub=np.inf):
        """
        For traditional MNL assortments, r is the (unconstrained) optimal price for a product if the revenue of the
        assortment is r-1. Further, if the optimal point is not in the interior it will occur at one of the end points.
        """
        r_old = None
        if r is not None:
            r_old = np.copy(self.r)
            self.set_r(r)
        _r = np.copy(self.r)
        if r is not None:
            self.set_r(r_old)
        r_min = 0.0
        r_max = 10.0
        _r[i] = r_max
        while r_max - 1 < self.revenue_trad(r=_r):
            r_min = r_max
            r_max *= 2
            _r[i] = r_max

        r_mid = -1
        while r_max - r_min > 1e-5:
            r_mid = (r_max + r_min) / 2
            _r[i] = r_mid
            if r_mid - 1 < self.revenue_trad(r=_r):
                r_min = r_mid
            else:
                r_max = r_mid
        if lb <= r_mid <= ub:
            return r_mid
        else:
            _r[i] = lb
            rev_down = self.revenue_trad(r=_r)
            _r[i] = ub
            rev_up = self.revenue_trad(r=_r)
            if rev_up > rev_down:
                return ub
            else:
                return lb

    def opt_price(self, lb=0, ub=np.inf):
        v_max = np.max(self.v)
        v_new = v_max + np.log(np.sum(np.exp(self.v - v_max)))
        new_model = MNL(v_new, 0.0)
        return new_model.trad_single_price_opt(0, lb=lb, ub=ub)

    def find_optimal_opaque_price(self, assortment=None, r=None):
        _r = r if r is not None else self.r
        if assortment is None:
            ub = np.min(_r)
        else:
            ub = np.min(_r[np.array(assortment, dtype=bool)])
        return unimodal_maximize(lambda x: self.revenue_opq(x, assortment, r), lb=0, ub=ub)

    def nested_by_rev_n_val(self):
        best_asst = None
        best_opq_price = 0
        best_revenue = 0
        # Add random perturbations
        r_pert = self.r + np.random.uniform(-1e-4, 1e-4, size=self.n)
        for v_thresh in np.unique(self.v):
            for r_thresh in r_pert:
                asst = np.array((self.v >= v_thresh) & (r_pert >= r_thresh), dtype=float)
                if np.sum(asst) == 0:
                    continue
                rho = self.find_optimal_opaque_price(asst)
                rev = self.revenue_opq(rho, asst)
                if rev > best_revenue:
                    best_revenue = rev
                    best_asst = np.copy(asst)
                    best_opq_price = rho
        return {
            "rev": best_revenue,
            "asst": best_asst,
            "rho": best_opq_price
        }

    def nested_by_rev_trad(self):
        best_asst = None
        best_revenue = 0
        for r_thresh in self.r:
            asst = np.array(self.r >= r_thresh, dtype=float)
            rev = self.revenue_trad(asst)
            if rev > best_revenue:
                best_revenue = rev
                best_asst = np.copy(asst)
        rho = self.find_optimal_opaque_price(best_asst)
        rev_opq = self.revenue_opq(rho, best_asst)
        return {
            "rev": rev_opq,
            "asst": best_asst,
            "rho": rho
        }

    def best_singleton(self):
        best_revenue = 0
        best_i = None
        best_opq_price = 0
        for i, (ri, vi) in enumerate(zip(self.r, self.v)):
            singleton_model = MNL(vi, ri)
            r_star = singleton_model.opt_price()
            if r_star < ri:
                rev = singleton_model.revenue_trad(r=r_star)
                rho = r_star
            else:
                rev = singleton_model.revenue_trad()
                rho = ri
            if rev > best_revenue:
                best_revenue = rev
                best_i = i
                best_opq_price = rho
        best_asst = np.zeros(self.n)
        best_asst[best_i] = 1
        return {
            "rev": best_revenue,
            "asst": best_asst,
            "rho": best_opq_price
        }

    def optimal_trad_or_singleton(self):
        trad_asst = self.nested_by_rev_trad()
        single_asst = self.best_singleton()
        if trad_asst["rev"] > single_asst["trad"]:
            return trad_asst
        else:
            return single_asst

    def brute_force(self):
        best_asst = None
        best_opq_price = 0
        best_revenue = 0
        for asst in itertools.product([0, 1], repeat=self.n):
            if not any(asst):
                continue
            rho = self.find_optimal_opaque_price(asst)
            rev = self.revenue_opq(rho, asst)
            if rev > best_revenue:
                best_revenue = rev
                best_asst = np.copy(asst)
                best_opq_price = rho
        return {
            "rev": best_revenue,
            "asst": best_asst,
            "rho": best_opq_price
        }
