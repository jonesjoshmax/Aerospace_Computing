from numpy import *
import sympy as sp
import matplotlib.pyplot as plt

a1 = arange(1.5, 4.75, .25)
n = a1.size
y = 1.4
m_initial = 2.8
p_initial = 130
a_total = 5

m = sp.var('m', real=True)
p = sp.var('p', real=True)
a_r = sp.var('a_r')
p_r = sp.var('p_r')
p_o = sp.var('p_o')
p_po = sp.var('p_po')

v_17 = sp.Eq((1 / m) * sp.Pow(((1 + ((y - 1) / 2) * sp.Pow(m, 2)) / ((y + 1) / 2)), ((y + 1) / (2 * (y - 1)))) - a_r, 0)
c_13 = sp.Eq(sp.Pow(((((y + 1) / 2) * sp.Pow(m, 2)) / (1 + ((y - 1) / 2) * sp.Pow(m, 2))), y / (y - 1)) *
             sp.Pow((((2 * y) / (y + 1)) * sp.Pow(m, 2) - ((y - 1) / (y + 1))), (1 / (1 - y))), p_r)
v_7 = sp.Eq(p * sp.Pow((1 + (((y - 1) / 2) * sp.Pow(m, 2))), (y / (y - 1))), p_o)
v_7_ratio = sp.Eq(1 / sp.Pow((1 + (((y - 1) / 2) * sp.Pow(m, 2))), (y / (y - 1))), p_po)

astar = sp.solve(v_17.subs(m, m_initial), a_r)[0]
a1_astar = a1 * astar

m_roots = zeros(n)
ai1_a2e = zeros(n)
ae_a2e = zeros(n)
m_e = zeros(n)
p_stag = p_initial / sp.solve(v_7_ratio.subs(m, m_initial), p_po)[0]
p_stag_e = zeros(n)
p_e = zeros(n)
for i in range(n):
    m_roots[i] = sp.solve(v_17.subs(a_r, a1_astar[i]), m)[1]
    ai1_a2e[i] = sp.solve(c_13.subs(m, m_roots[i]), p_r)[0]
    ae_a2e[i] = a_total * astar * ai1_a2e[i]
    m_e[i] = sp.solve(v_17.subs(a_r, ae_a2e[i]), m)[0]
    p_stag_e[i] = p_stag * ai1_a2e[i]
    p_e[i] = p_stag_e[i] * sp.solve(v_7_ratio.subs(m, m_e[i]), p_po)[0]

plt.plot(a1, p_e, color='red')
plt.title('Back Pressure vs Area Ratio Point')
plt.xlabel('Area Ratio')
plt.ylabel('Exit Pressure (kPa)')
plt.grid()
plt.tight_layout()
plt.show()

plt.plot(a1, m_e, color='red')
plt.title('Exit Mach vs Area Ratio Point')
plt.xlabel('Area Ratio')
plt.ylabel('Mach')
plt.grid()
plt.tight_layout()
plt.show()
