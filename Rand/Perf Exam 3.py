import numpy as np
import sympy as sp

q = sp.var('q')
k = sp.var('i')
w = sp.var('w')
s = sp.var('s')
t = sp.var('t')
c = sp.var('c')
n = sp.var('n')

expr = sp.sqrt((q / (k * (w / s))) * ((t / w) - ((q * c) / (w / s)))) - n
sp.init_printing()
for_t = expr.subs(n, 3).subs(w, 9500).subs(s, 170).subs(c, .015).subs(k, 0.0943).subs(q, 677.1467553)
