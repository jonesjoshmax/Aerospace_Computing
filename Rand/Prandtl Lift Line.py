import numpy as np
import sympy as sp
naca = ''
while len(naca) != 4:
    naca = str(input('NACA Number: '))

c = 1
m = float(naca[0]) / 100
p = float(naca[1]) / 10
thickness = (10 * float(naca[2])) + float(naca[3])
