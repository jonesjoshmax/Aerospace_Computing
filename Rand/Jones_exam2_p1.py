from numpy import *
import matplotlib.pyplot as plt

p_scale = 2116.220402

y = 1.4
p = 1.15 * p_scale
t1 = 50 + 459.67
r = 1717
vg = arange(900, 1455, 5)
vs = (y + 1) / 4 * vg + (1 / 2) * sqrt(pow((y + 1) / 2, 2) * pow(vg, 2) + (4 * y * r * t1))
mach_s = vs / sqrt(y * r * t1)
mach_2 = sqrt((pow(mach_s, 2) + (2 / (y - 1))) / (2 * y / (y - 1) * pow(mach_s, 2) - 1))
t2 = t1 * (1 + ((y - 1) / 2) * pow(mach_s, 2)) / (1 + ((y - 1) / 2) * pow(mach_2, 2)) - 459.67

plt.plot(vg, vs, color='red')
plt.title('Shock Velocity vs Projectile Velocity')
plt.xlabel('Projectile Velocity (ft/s)')
plt.ylabel('Shock Velocity (ft/s)')
plt.grid()
plt.tight_layout()
plt.show()

plt.plot(vg, t2, color='red')
plt.title('Static Temperature vs Projectile Velocity')
plt.xlabel('Projectile Velocity (ft/s)')
plt.ylabel('Static Temperature (F)')
plt.grid()
plt.tight_layout()
plt.show()
