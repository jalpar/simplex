#!/usr/bin/env python3
# -*- indent-tabs-mode: nil; -*-

import copy
import random
import json
from fractions import Fraction as Frac


def defrac_num(v):
    return [v.numerator, v.denominator] if isinstance(v, Frac) else v


def refrac_num(v):
    return Frac(v[0], v[1]) if isinstance(v, list) else v


def defrac_list(vec):
    return [defrac_num(v) for v in vec]


def refrac_list(vec):
    return [refrac_num(v) for v in vec]


def defrac_dict(dct):
    return {k: defrac_num(v) for k, v in dct.items()}


def refrac_dict(dct):
    return {k: refrac_num(v) for k, v in dct.items()}


def print_m(M):
    M = [[str(v) for v in r] for r in M]
    M = [[v if v and v[0] == '-' else ' '+v for v in r] for r in M]
    n = max(len(r) for r in M)
    lens = [0]*n
    for r in M:
        for j, v in enumerate(r):
            lens[j] = max(lens[j], len(v))
    for r in M:
        for v, p in zip(r, lens):
            print(f"{v}", ' '*(p-len(v)), end='')
        print()


class SimplexTableau:
    def update_vars(self):
        self.vars = {v: (0, i) for i, v in enumerate(self.nb_vars)}
        self.vars.update((v, (1, i)) for i, v in enumerate(self.b_vars))

    def __init__(self, m, n=0):
        self.debug_level = 2
        if not isinstance(m, str):
            self.m = m
            self.n = n
            self.A = [[0]*n for i in range(m)]
            self.b = [0]*m
            self.c = [0]*n
            self.nb_vars = list(f"x{i}" for i in range(1, n+1))
            self.b_vars = list(f"s{i}" for i in range(1, m+1))
            self.update_vars()
            self.costs = {}
            self.epsilon = 0.0000001
        else:
            inp = json.load(open(m))
            self.m = inp["m"]
            self.n = inp["n"]
            self.A = [refrac_list(r) for r in inp["A"]]
            self.b = refrac_list(inp["b"])
            self.c = refrac_list(inp["c"])
            self.nb_vars = inp["nb_vars"]
            self.b_vars = inp["b_vars"]
            self.update_vars()
            self.costs = refrac_dict(inp["costs"])
            self.epsilon = inp["epsilon"]

    def save(self, fname):
        out = {
            "m": self.m,
            "n": self.n,
            "A": [defrac_list(r) for r in self.A],
            "b": defrac_list(self.b),
            "c": defrac_list(self.c),
            "nb_vars": self.nb_vars,
            "b_vars": self.b_vars,
            "costs": defrac_dict(self.costs),
            "epsilon": self.epsilon
        }
        json.dump(out, open(fname, "w"), indent=2)

    def clone(self):
        s = SimplexTableau(self.m, self.n)
        s.A = copy.deepcopy(self.A)
        s.b = copy.deepcopy(self.b)
        s.c = copy.deepcopy(self.c)
        s.nb_vars = copy.deepcopy(self.nb_vars)
        s.b_vars = copy.deepcopy(self.b_vars)
        s.vars = copy.deepcopy(self.vars)
        s.costs = copy.deepcopy(self.costs)
        s.epsilon = self.epsilon
        return s

    def _make_frac_num(self, v, max_denominator=None):
        if max_denominator:
            return Frac(v).limit_denominator(max_denominator)
        else:
            return Frac(v)

    def _make_frac_vec(self, vec, max_denominator=None):
        return [self._make_frac_num(v, max_denominator=max_denominator)
                for v in vec]

    def make_frac(self, max_denominator=1000000):
        self.A = [self._make_frac_vec(row, max_denominator=max_denominator)
                  for row in self.A]
        self.b = self._make_frac_vec(self.b, max_denominator=max_denominator)
        self.c = self._make_frac_vec(self.c, max_denominator=max_denominator)
        self.epsilon = 0

    def is_slack(self, v):
        return v[0] == 's'

    def set_costs(self, **costs):
        self.costs = copy.copy(costs)

    def init_zero_reduced_costs(self):
        for j in range(self.n):
            self.c = [self.Z] * self.n

    def init_first_phase_costs(self):
        for j in range(self.n):
            self.c[j] = sum(self.A[i][j] for i in range(self.m))

    def init_reduced_costs(self):
        self.c = [self.costs[v] if v in self.costs else self.Z
                  for v in self.nb_vars]
        for v, r in zip(self.b_vars, self.A):
            if v in self.costs:
                c = self.costs[v]
                for j in range(self.n):
                    self.c[j] -= c*r[j]

    def calc_obj(self):
        return sum((self.costs[v] if v in self.costs else self.Z)*b
                   for v, b in zip(self.b_vars, self.b))

    def calc_first_phase_obj(self):
        return sum(b for v, b in zip(self.b_vars, self.b) if self.is_slack(v))

    def dump(self):
        M = [[''] + self.nb_vars] + \
            [[bv]+r+['|', rh] for bv, r, rh in zip(self.b_vars,
                                                   self.A,
                                                   self.b)] + \
            [['c'] + self.c]
        print_m(M)
        print()
        print("Primal solution:",
              ' '.join(f"{vr}={vl}"
                       for vr, vl in sorted(zip(self.b_vars, self.b))))
        print("Dual solution:",
              ' '.join(f"{vr}={-vl}"
                       for vr, vl in sorted(zip(self.nb_vars, self.c))
                       if self.is_slack(vr)))
        print("Objective value:", self.calc_obj())
        print()

    def pivot(self, i, j):
        if self.debug_level > 0:
            print(f"\n*** Pivot {self.b_vars[i]} -> "
                  f"{self.nb_vars[j]}  ({i},{j})")
        piv = self.A[i][j]
        delta = self.c[j]/piv
        if self.debug_level > 1:
            print()
            print(f"Delta: primal={self.b[i]/self.A[i][j]} dual={delta}")
        # modify c
        for l in range(self.n):
            self.c[l] -= delta*self.A[i][l]
        self.c[j] = -delta
        # modify row i
        for l in range(self.n):
            self.A[i][l] = 1/piv if l == j else self.A[i][l]/piv  # noqa
        self.b[i] /= piv
        # modify other rows
        for k in range(self.m):
            if k != i:
                f = self.A[k][j]
                for l in range(self.n):
                    self.A[k][l] = (-f*1/piv if l == j  # noqa
                                    else self.A[k][l]-f*self.A[i][l])
                self.b[k] -= f*self.b[i]

        self.nb_vars[j], self.b_vars[i] = self.b_vars[i], self.nb_vars[j]
        self.vars[self.b_vars[i]] = (1, i)
        self.vars[self.nb_vars[j]] = (0, j)
        if self.debug_level > 1:
            print()
            self.dump()

    def primal_step(self, skip_slacks=True):
        try:
            v, j = min((v, j)
                       for j, v in enumerate(self.nb_vars)
                       if self.c[j] > self.epsilon and
                       not (skip_slacks and self.is_slack(v)))
        except ValueError:
            return 'optimal'
        try:
            delta, v, i = min((self.b[i]/self.A[i][j], self.b_vars[i], i)
                              for i in range(self.m)
                              if self.A[i][j] > self.epsilon)
        except ValueError:
            return 'unbounded'
        self.pivot(i, j)
        return 'go_on'

    def primal(self, skip_slacks=True):
        if self.debug_level > 0:
            print("Start Primal Simplex")
        if self.debug_level > 1:
            self.dump()
        while True:
            ret = self.primal_step(skip_slacks=skip_slacks)
            if ret in ['optimal', 'unbounded']:
                return ret

    def dual_step(self):
        try:
            v, i = min((self.b_vars[i], i)
                       for i in range(self.m)
                       if self.b[i] < -self.epsilon)
        except ValueError:
            return 'optimal'
        try:
            delta, v, j = min((self.c[j]/self.A[i][j], v, j)
                              for j, v in enumerate(self.nb_vars)
                              if self.A[i][j] < -self.epsilon and
                              not self.is_slack(v))
        except ValueError:
            return 'unbounded'
        self.pivot(i, j)

    def dual(self):
        if self.debug_level > 0:
            print("Start Dual Simplex")
        if self.debug_level > 1:
            self.dump()
        while True:
            ret = self.dual_step()
            if ret in ['optimal', 'unbounded']:
                return ret

    def pivot_out_slacks(self, remove_dependent_rows=False):
        for i, sv in enumerate(self.b_vars):
            if self.is_slack(sv):
                if self.debug_level > 0:
                    print(f"Pivoting out {sv}.")
                for j, nv in enumerate(self.nb_vars):
                    if not self.is_slack(nv) and \
                       abs(self.A[i][j]) > self.epsilon:
                        self.pivot(i, j)
                        break
                else:
                    if self.debug_level > 0:
                        print("Warning: Singular Matrix")
        if remove_dependent_rows:
            for i, sv in reversed(list(enumerate(self.b_vars))):
                if self.is_slack(sv):
                    if self.debug_level > 0:
                        print(f"Singular matrix, remove row {i} ({sv}).")
                    del self.A[i]
                    del self.b_vars[i]
            self.update_vars()

    def remove_slacks(self):
        for j, sv in reversed(list(enumerate(self.nb_vars))):
            if self.is_slack(sv):
                for a in self.A:
                    del a[j]
                del self.nb_vars[j]
        self.update_vars()

    def two_phase_simplex_pp(self):
        # TODO: check if b>=0
        if self.debug_level > 1:
            self.dump()
        if self.debug_level > 0:
            print("*********************\n"
                  "* Start First Phase *\n"
                  "*********************")

        self.init_first_phase_costs()
        self.primal(skip_slacks=False)
        if self.calc_first_phase_obj() > self.epsilon:
            return 'infeasible'

        if self.debug_level > 0:
            print("********************\n"
                  "* Pivot Out Slacks *\n"
                  "********************")

        self.pivot_out_slacks()

        if self.debug_level > 0:
            print("**********************\n"
                  "* Start Second Phase *\n"
                  "**********************")

        self.init_reduced_costs()
        return self.primal(skip_slacks=True)

    def two_phase_simplex_dp(self):
        if self.debug_level > 1:
            self.dump()
        if self.debug_level > 0:
            print("********************\n"
                  "* Pivot Out Slacks *\n"
                  "********************")

        self.pivot_out_slacks()

        if self.debug_level > 0:
            print("*********************\n"
                  "* Start First Phase *\n"
                  "*********************")

        self.init_zero_reduced_costs()
        if self.dual() == 'unbounded':
            return 'infeasible'

        if self.debug_level > 0:
            print("**********************\n"
                  "* Start Second Phase *\n"
                  "**********************")

        self.init_reduced_costs()
        return self.primal(skip_slacks=True)

    def two_phase_simplex_pd(self):
        b_save = copy.copy(self.b)
        self.b = [self.Z]*self.m

        self.init_reduced_costs()

        if self.debug_level > 1:
            self.dump()
        if self.debug_level > 0:
            print("********************\n"
                  "* Pivot Out Slacks *\n"
                  "********************")

        self.pivot_out_slacks()

        if self.debug_level > 0:
            print("*********************\n"
                  "* Start First Phase *\n"
                  "*********************")

        if self.primal(skip_slacks=True) == 'unbounded':
            return 'infeasible_or_unbounded'

        self.b = [
            sum((row[self.vars[f"s{j+1}"][1]]
                 if self.vars[f"s{j+1}"][0] == 0 else 0)
                * b_save[j] for j in range(self.m)) +
            (b_save[i] if self.b_vars[i] == f"s{i+1}" else 0)
            for i, row in enumerate(self.A)]

        if self.debug_level > 0:
            print("**********************\n"
                  "* Start Second Phase *\n"
                  "**********************")

        ret = self.dual()
        if ret == 'optimal':
            return 'optimal'
        elif ret == 'unbounded':
            return 'infeasible'


def make_example():
    s = SimplexTableau(3, 5)
    s.A[0] = [1, 1, 1, 0, 1]
    s.b[0] = 7

    s.A[1] = [1, 0, 0, 2, 2]
    s.b[1] = 9

    s.A[2] = [-1, 2, -3, 5, 1]
    s.b[2] = 14

    s.set_costs(x1=1, x2=2, x3=3, x4=4, x5=5)
    s.make_frac()
    return s


def make_example2():
    s = SimplexTableau(3, 5)
    s.A[0] = [1, 1, 1, 0, 1]
    s.b[0] = 2

    s.A[1] = [1, 0, 0, 2, 2]
    s.b[1] = 4

    s.A[2] = [-1, 2, -3, 5, 1]
    s.b[2] = 3

    s.set_costs(x1=1, x2=2, x3=3, x4=4, x5=5)
    s.make_frac()
    return s


def make_example3():
    s = SimplexTableau(3, 5)
    s.A[0] = [1, 1, 1, 0, 1]
    s.b[0] = 3

    s.A[1] = [1, 0, 0, 2, 2]
    s.b[1] = 2

    s.A[2] = [-1, 2, -3, 5, 1]
    s.b[2] = 0

    s.set_costs(x1=1, x2=2, x3=3, x4=4, x5=5)
    s.make_frac()
    return s


def make_example_rand(m, n, r):
    s = SimplexTableau(m, n)
    s.A = [[random.randint(0, r) for j in range(n)] for i in range(m)]
    s.b = [random.randint(0, r) for i in range(m)]

    s.set_costs(**{f"x{j}": random.randint(-r, r) for j in range(1, n+1)})
    s.make_frac()
    return s


def make_example_assingment(n, r):
    s = SimplexTableau(2*n, n*n)
    s.b = [1 for i in range(2*n)]
    for i in range(n):
        for j in range(n):
            s.A[i][i*n+j] = 1
            s.A[i+n][i+j*n] = 1
    s.nb_vars = [f"x{i}-{j}" for i in range(1, n+1) for j in range(1, n+1)]

    s.set_costs(**{f"x{i}-{j}": random.randint(0, r)
                   for i in range(1, n+1) for j in range(1, n+1)})
    # s.set_costs(**{f"x{i}-{j}": 1
    #                for i in range(1, n+1) for j in range(1, n+1)})
    # s.make_frac()
    return s


def test_ass(s, r):
    import datetime
    now = datetime.datetime.now

    ao = make_example_assingment(s, r)

    a = ao.clone()
    a.debug_level = 0
    n = now()
    a.two_phase_simplex()
    d = now() - n
    print("Obj:", a.calc_obj())
    print(d.seconds+d.microseconds/1000000)

    a = ao.clone()
    a.debug_level = 0
    n = now()
    a.two_phase_simplex_2()
    d = now() - n
    print("Obj:", a.calc_obj())
    print(d.seconds+d.microseconds/1000000)

    a = ao.clone()
    a.debug_level = 0
    n = now()
    a.two_phase_simplex_3()
    d = now() - n
    print("Obj:", a.calc_obj())
    print(d.seconds+d.microseconds/1000000)
