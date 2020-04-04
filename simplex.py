#!/usr/bin/env python3
# -*- indent-tabs-mode: nil; -*-

import json
import sys
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
        self.nb_vars = list(range(n))
        self.b_vars = list(range(n, n+m))
        self.update_vars()

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

    def bland_primal_step(self):
        # for l in range(self.n):
        #     if self.c[l] > self.epsilon:
        #         j = l
        #         break
        # else:
        #     return 'optimal'
        try:
            v, j = min((self.nb_vars[j], j)
                       for j in range(self.n)
                       if self.c[j] > self.epsilon)
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

    def bland_primal(self):
        while True:
            ret = self.bland_primal_step()
            if ret in ['optimal', 'unbounded']:
                return ret

    def bland_dual(self):
        while True:
            for k in range(self.n):
                if self.b[k] < -self.epsilon:
                    i = k
                    break
            else:
                return 'optimal'
            try:
                delta, j = min((self.c[j]/self.A[i][j], j)
                               for j in range(self.n)
                               if self.A[i][j] < -self.epsilon)
            except ValueError:
                return 'unbounded'
            self.pivot(i, j)

    def first_phase_cost(self):
        for j in range(self.n):
            self.c[j] = sum(self.A[i][j] for i in range(self.m))

    def pivot_out_slacks(self):
        while True:
            for i, sv in enumerate(self.b_vars):
                if self.debug_level > 0:
                    print("Pivoting out {sv}.")
                if sv[0] == 's':
                    for j, nv in enumerate(self.nb_vars):
                        if nv[0] != 's' and abs(self.A[i][j]) > self.epsilon:
                            break
                    else:
                        print("Singular matrix!!!!")
                        return
                    self.pivot(i, j)
                    break
            else:
                break

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

    s.nb_vars = ["x1", "x2", "x3", "x4", "x5"]
    s.b_vars = ["s1", "s2", "s3"]
    s.update_vars()
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

    s.nb_vars = ["x1", "x2", "x3", "x4", "x5"]
    s.b_vars = ["s1", "s2", "s3"]
    s.update_vars()
    s.first_phase_cost()
    return s

def make_example2():
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

    s.nb_vars = ["x1", "x2", "x3", "x4", "x5"]
    s.b_vars = ["s1", "s2", "s3"]
    s.update_vars()
    s.first_phase_cost()
    return s

# class Simplex:
#     def __init__(self, n, m):
#         Z = 0
#         self.epsilon = 0
#         self.m = m
#         self.n = n
#         self.A = [[Z]*n for i in range(m)]
#         self.Inv = [[Z]*m for i in range(m)]
#         self.b = [Z]*m
#         self.c = [Z]*n
#         self.vstat = [(0, i) for i in range(n)] + [(1, i) for i in range(n, m)]
#         self.vars = (list(range(0)), list(range(m)))

#     def gauss(self, i, j):
#         for l in range(n):
#             self.A[i][l] /= scale
#         for l in range(m):
#             self.Inv[i][l] /= scale
#         for k in range(m):
#             if k != i:
#                 scale = self.A[k][j]
#                 for l in range(n):
#                     self.A[k][l] -= scale*self.A[i][l]
#                 for l in range(m):
#                     self.Inv[k][l] -= scale*self.Inv[i][l]

#     def gauss_base(self, rows, cols):
#         base = []
#         for j in cols:
#             for i in rows:
#                 if self.A[i][j] > self.epsilon:
#                     base.append((i, j))
#                     self.gauss[i][j]
#                     break
#             if len(base) == m:
#                 return base
#         print('ERROR: Not full rank')
#         exit(1)

#     def two_phase():
#         for i in range(m):
#             if b[i] < 0:
#                 for j in range(n):
#                     self.A[i][j] *= -1
#                 for j in range(m):
#                     self.Inv[i][j] *= -1
#                 b[i] *= -1
#         sim = SimplexTableau(m, n)
#         sim.A = copy.copy(self.A)
#         sim.b = copy.copy(self.b)
#         sim.c = [-sum(A[i][j] for j in range(m)) for i in range(n)]
#         sim.bland_primal()
#         base = sim.vars[1]
#         for i in range(m):
#             if base[i] >= n and sim.b[i] > self.epsilon:
#                 return 'infeasible'
