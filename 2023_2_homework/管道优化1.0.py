import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calc_optimal_diameter(Q, L, f):
    """
    计算最优管道内径d
    Q: 流量，单位m3/s
    L: 管道长度，单位m
    f: 摩擦系数
    返回最优管道内径d，单位m
    """
    g = 9.81  # 重力加速度，单位m/s^2
    rho = 1000  # 流体密度，单位kg/m^3
    epsilon = 0.0000015  # 管壁绝对粗糙度，单位m
    k = 1 / (1.74 * np.log10(2 * L / epsilon) - 2)  # 管道粗糙度修正系数
    d = (4 * Q**2 * k / (np.pi**2 * g * f * L * rho))**(1/5)  # 最优管道内径公式
    return d

# 示例：计算流量为0.2m3/s，管道长度为200m，摩擦系数为0.02时的最优管道内径
Q = 0.2
L = 200
f = 0.02
d = calc_optimal_diameter(Q, L, f)
print("最优管道内径为：{:.4f}m".format(d))

# 3D可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Q_vals = np.linspace(0.1, 1, 20)
L_vals = np.linspace(100, 500, 20)
Q, L = np.meshgrid(Q_vals, L_vals)
d_vals = calc_optimal_diameter(Q, L, f)
ax.plot_surface(Q, L, d_vals, cmap='coolwarm')
ax.set_xlabel('流量 Q (m^3/s)')
ax.set_ylabel('管道长度 L (m)')
ax.set_zlabel('最优管道内径 d (m)')



plt.show()
