from numpy import *

w = .02
h = .09
t = .003
m = 2552
v = 3254
i = (w * pow(h, 3) / 12) - ((w - (2 * t)) * pow((h - (2 * t)), 3)) / 12
j = h * w * (pow(h, 2) + pow(w, 2)) / 12
siga = - m / i * h / 2
sigb = - m / i * - h / 2
tau = v * (h - t) / (4 * i) * w + (v / i) * ((((h - t) * ((h - t) / 2)) / 2) - (pow(((h - t) / 2), 2) / 2))
sig1 = (siga / 2) - sqrt(pow((siga / 2), 2) + pow(tau, 2))
sig2 = (siga / 2) + sqrt(pow((siga / 2), 2) + pow(tau, 2))
sig3 = 0
sig4 = sqrt(((sig1 - sig2) ** 2 + (sig2 - sig3) ** 2 + (sig3 - sig1) ** 2) / 2)
