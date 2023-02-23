import sympy
import sympy as sp

m = sp.var('m')
a = sp.var('a')

expr = ((1 / a) * sp.atan(a * sp.sqrt((m ** 2) - 1))) - sp.atan((m ** 2) - 1)

expr_d1 = sp.diff(expr, m)
expr_d2 = sp.diff(expr_d1, m)
sympy.init_printing(use_unicode=True)
expr2 = (m / (sp.sqrt((m ** 2) - 1) * (a ** 2 * (m ** 2 - 1) + 1))) - sp.atan(sp.sqrt(m ** 2 - 1))

print(expr.subs(a, 2).subs(m, 5))
print(expr2.subs(a, 2).subs(m, 5))
