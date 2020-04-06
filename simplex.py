#!/usr/bin/env python3
# -*- indent-tabs-mode: nil; -*-

import copy
from fractions import Fraction as Frac


def sign(n):
    return 1 if n >= 0 else 0


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

    def __init__(self, m, n):
        Z = 0
        self.debug_level = 2
        self.epsilon = 0
        self.m = m
        self.n = n
        self.A = [[Z]*n for i in range(m)]
        self.b = [Z]*m
        self.c = [Z]*n
        self.nb_vars = list(f"x{i}" for i in range(1, n+1))
        self.b_vars = list(f"s{i}" for i in range(n+1, n+m+1))
        self.update_vars()
        self.costs = {}

    def dump(self):
        M = [[''] + self.nb_vars] + \
            [[bv]+r+['|', rh] for bv, r, rh in zip(self.b_vars,
                                                   self.A,
                                                   self.b)] + \
            [['c'] + self.c]
        print_m(M)
        print()
        print("Primal solution: ", end="")
        for i in range(self.m):
            print(f"{self.b_vars[i]}:{self.b[i]} ", end='')
        print()

    def is_slack(v):
        return v[1] == 's'

    def pivot(self, i, j):
        if self.debug_level > 0:
            print(f"Pivot {self.b_vars[i]} -> {self.nb_vars[j]}  ({i},{j})")
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

    def calc_cost(self):
        return sum(v*b for v, b in zip(self.b_vars, self.b))

    def primal_step(self, skip_slacks=True):
        try:
            v, j = min((v, j)
                       for j, v in enumerate(self.nb_vars)
                       if self.c[j] > self.epsilon and not self.is_slack(v))
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

    def primal(self):
        while True:
            ret = self.bland_primal_step()
            if ret in ['optimal', 'unbounded']:
                return ret

    def first_phase_cost(self):
        for j in range(self.n):
            self.c[j] = sum(self.A[i][j] for i in range(self.m))

    def calc_first_phase_cost(self):
        return sum(b for v, b in zip(self.b_vars, self.b) if self.is_slack(v))

    def setup_costs(self):
        self.c = [self.costs[v] if v in self.costs else self.Z
                  for v in self.nb_vars]
        for v, r in zip(self.b_vars, self.A):
            if v in self.costs:
                c = self.costs[v]
                for j in range(self.n):
                    c[j] -= c*r[j]

    def pivot_out_slacks(self, remove_dependent_rows=False):
        for i, sv in enumerate(self.b_vars):
            if self.is_slack(sv):
                if self.debug_level > 0:
                    print(f"Pivoting out {sv}.")
                for j, nv in enumerate(self.nb_vars):
                    if not self.is_slack(sv) and \
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

    def two_phase_simplex(self):
        self.first_phase_cost()
        self.primal(skip_slacks=False)
        if self.calc_first_phase_cost() > self.epsilon:
            return 'infeasible'
        self.pivot_out_slacks()
        self.setup_costs()
        return self.primal(skip_slacks=True)

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
        while True:
            ret = self.dual_step()
            if ret in ['optimal', 'unbounded']:
                return ret


def make_example():
    s = SimplexTableau(3, 5)
    s.A[0][0] = Frac(1)
    s.A[0][1] = Frac(1)
    s.A[0][2] = Frac(1)
    s.A[0][4] = Frac(1)
    s.b[0] = Frac(7)

    s.A[1][0] = Frac(1)
    s.A[1][3] = Frac(2)
    s.A[1][4] = Frac(2)
    s.b[1] = Frac(9)

    s.A[2][0] = Frac(-1)
    s.A[2][1] = Frac(2)
    s.A[2][2] = Frac(-3)
    s.A[2][3] = Frac(5)
    s.A[2][4] = Frac(1)
    s.b[2] = Frac(14)

    s.first_phase_cost()
    return s


def make_example2():
    s = SimplexTableau(3, 5)
    s.A[0][0] = Frac(1)
    s.A[0][1] = Frac(1)
    s.A[0][2] = Frac(1)
    s.A[0][4] = Frac(1)
    s.b[0] = Frac(2)

    s.A[1][0] = Frac(1)
    s.A[1][3] = Frac(2)
    s.A[1][4] = Frac(2)
    s.b[1] = Frac(4)

    s.A[2][0] = Frac(-1)
    s.A[2][1] = Frac(2)
    s.A[2][2] = Frac(-3)
    s.A[2][3] = Frac(5)
    s.A[2][4] = Frac(1)
    s.b[2] = Frac(3)

    s.first_phase_cost()
    return s


def make_example3():
    s = SimplexTableau(3, 5)
    s.A[0][0] = Frac(1)
    s.A[0][1] = Frac(1)
    s.A[0][2] = Frac(1)
    s.A[0][4] = Frac(1)
    s.b[0] = Frac(3)

    s.A[1][0] = Frac(1)
    s.A[1][3] = Frac(2)
    s.A[1][4] = Frac(2)
    s.b[1] = Frac(2)

    s.A[2][0] = Frac(-1)
    s.A[2][1] = Frac(2)
    s.A[2][2] = Frac(-3)
    s.A[2][3] = Frac(5)
    s.A[2][4] = Frac(1)
    s.b[2] = Frac(0)

    s.first_phase_cost()
    return s
