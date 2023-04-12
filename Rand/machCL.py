from numpy import *

cl = .15
m = .8
y = 1.4

cl_pg = cl / sqrt(1 - pow(m, 2))
cl_kt = cl / (sqrt(1 - pow(m, 2)) + cl / 2 * (pow(m, 2) / (1 + sqrt(1 - pow(m, 2)))))
cl_l = cl / (sqrt(1 - pow(m, 2)) + cl * (pow(m, 2) * (1 + ((y - 1) / 2) * pow(m, 2))) / (2 * sqrt(1 - pow(m, 2))))

print(cl_pg, cl_kt, cl_l)
