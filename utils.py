import numpy as np


def unimodal_find_largest_equal(f, x_0, ub=np.inf):
    target = f(x_0)
    f_ub = f(ub)
    if f_ub > target:
        return np.inf
    x_min = x_0
    x_max = x_min * 2
    while f(x_max) >= target:
        x_min = x_max
        x_max *= 2
    while x_max - x_min > 1e-5:
        x_mid = (x_max + x_min) / 2
        f_mid = f(x_mid)
        if f_mid > target:
            x_min = x_mid
        elif f_mid < target:
            x_max = x_mid
        else:
            return x_mid
    return x_min


def unimodal_maximize(f, lb=0, ub=np.inf):
    x_max = ub
    x_min = lb

    f_down = f(x_min)
    f_up = f(x_max)

    if x_max > 20:
        x_max = 2
        f_prev = f(1)
        f_up = f(2)
        while f_up > f_prev:
            f_prev = f_up
            x_max *= 2
            f_up = f(x_max)

    # find point above
    # both points
    x_mid = np.nan_to_num((x_max + x_min) / 2)
    f_mid = f(x_mid)
    while (f_down > f_mid or f_up > f_mid) and x_max - x_min > 1e-8:
        if f_down > f_mid:
            x_max = x_mid
            f_up = f_mid

            x_mid = (x_min + x_mid) / 2
            f_mid = f(x_mid)

        elif f_up > f_mid:
            x_min = x_mid
            f_down = f_mid

            x_mid = (x_max + x_mid) / 2
            f_mid = f(x_mid)

    while x_max - x_min > 1e-8:
        x_low = (x_min + x_mid) / 2
        f_low = f(x_low)

        x_high = (x_max + x_mid) / 2
        f_high = f(x_high)
        try:
            assert not(f_low > f_mid + 1e-5 and f_high > f_mid + 1e-5)
        except AssertionError:
            print(x_low, f_low)
            print(x_mid, f_mid)
            print(x_high, f_high)
            raise AssertionError
        if f_low > f_mid:
            x_max = x_mid

            x_mid = x_low
            f_mid = f_low

        elif f_high > f_mid:
            x_min = x_mid

            x_mid = x_high
            f_mid = f_high

        else:
            x_min = x_low

            x_max = x_high

    return x_mid
