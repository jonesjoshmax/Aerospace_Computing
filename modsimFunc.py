import numpy as np


# K for hollow
# G Modulus, D O Diameter, d I Diameter, L length
def k_hollowShaft(G, D, d, L):
    return np.pi * G * (D ** 4 - d ** 4) / (32 * L)


# K for coil
# G Modulus, d Diameter, n Number of Coils, R Coil Radius
def k_coil(G, d, n, R):
    return G * (d ** 4) / (64 * n * (R ** 3))


# Axial Deformation
# L Length, E Young's Mod, A Cross Sectional Area, f Axial Force
def axial_deformation(L, E, A, f):
    return L * f / (E * A)

