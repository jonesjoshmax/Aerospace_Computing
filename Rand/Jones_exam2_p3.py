from numpy import *
import matplotlib.pyplot as plt

mfr = 3
p_stag = 1400000
t_stag = 200 + 273.15
p_exit = 100
r = 287
y = 1.4
c_p = r * y / (y - 1)
p_array = flip(arange(100000, 1401000, 50000)).astype(float)
p_array[0] = p_array[0] - .001
t = t_stag * pow(p_array / p_stag, (y - 1) / y)
v = sqrt(2 * c_p * (t_stag - t))
m = v / sqrt(y * r * t)
a = mfr / (p_stag / sqrt(r * t_stag) * sqrt(y) * m / pow((1 + ((y - 1) / 2) * pow(m, 2)), (y + 1) / (2 * (y - 1))))
r = sqrt(a / pi)
p_array = p_array / 1000

plt.title('Area vs Pressure')
plt.xlabel('Pressure (kPa)')
plt.ylabel('Area (m^2)')
plt.plot(p_array[1:], a[1:], color='red')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.grid()
plt.show()

plt.title('Scale Nozzle')
plt.xlabel('Pressure (kPa)')
plt.ylabel('(m)')
plt.plot(p_array[1:], r[1:], color='red')
plt.plot(p_array[1:], -r[1:], color='red')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.grid()
plt.show()

plt.title('Mach vs Point in Nozzle')
plt.xlabel('Nozzle Location')
plt.ylabel('Mach')
plt.plot(arange(m.size), m, color='red')
plt.tight_layout()
plt.grid()
plt.show()
