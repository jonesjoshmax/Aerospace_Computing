import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

m = sp.var('m')
y = 1.4

expr = (1 + (((y - 1) / 2) * (m ** 2))) ** (y / (y - 1))
x = np.linspace(0, 2, 11)
p_r = np.zeros(np.size(x))
for i in range(np.size(x)):
    p_r[i] = expr.subs(m, x[i])

# Creating a dataFrame from pandas to create table
data = {'Mach Number': x,
        'Pressure Ratio': p_r}
# Tabled values with relevant labels
df = pd.DataFrame(data)
blankIndex = [''] * len(df)
df.index = blankIndex
print(df)

