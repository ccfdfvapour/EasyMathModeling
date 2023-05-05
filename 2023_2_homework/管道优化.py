import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
def pressure_drop(D):
    rho = 1000 # 流体密度
    mu = 0.001 # 流体粘度
    L = 1000 # 管道长度
    P1 = 100000 # 管道入口压力
    P2 = 98000 # 管道出口压力
    v = (P1 - P2) * L / (8 * mu) # 流体速度
    delta_p = 64 * mu / (np.pi * D**2) * (P1 - P2)**2 / L**2 # 压力损失
    return delta_p

res = minimize(pressure_drop, x0=0.1, bounds=[(0.01, 1.0)]) # 以0.1为初始值优化管道直径
D_opt = res.x[0] # 得到优化的管道直径
print("Optimized diameter: ", D_opt)
print("Minimum pressure drop: ", pressure_drop(D_opt))

import numpy as np
import matplotlib.pyplot as plt

# 定义管道优化模型
def pressure_drop(D, L, rho=1000, mu=0.001, P1=100000, P2=98000):
    v = (P1 - P2) * L / (8 * mu) # 流体速度
    delta_p = 64 * mu / (np.pi * D**2) * (P1 - P2)**2 / L**2 # 压力损失
    return delta_p, v

# 生成数据
D_range = np.linspace(0.01, 0.5, 100) # 管道内径范围
L_range = np.linspace(100, 5000, 100) # 管道长度范围
D, L = np.meshgrid(D_range, L_range)
delta_p, v = pressure_drop(D, L)

# 绘制压力损失和流体速度的3D图
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(D, L, delta_p, cmap='viridis')
ax1.set_xlabel('Pipe diameter (m)')
ax1.set_ylabel('Pipe length (m)')
ax1.set_zlabel('Pressure drop (Pa)')
ax1.set_title('Pressure drop vs. pipe diameter and length')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(D, L, v, cmap='viridis')
ax2.set_xlabel('Pipe diameter (m)')
ax2.set_ylabel('Pipe length (m)')
ax2.set_zlabel('Flow velocity (m/s)')
ax2.set_title('Flow velocity vs. pipe diameter and length')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def delta_P(D, L, P1, P2, mu):
    """
    计算管道压力损失
    :param D: 管道内径
    :param L: 管道长度
    :param P1: 管道入口压力
    :param P2: 管道出口压力
    :param mu: 流体粘度
    :return: 压力损失
    """
    v = (P1 - P2) * L / (8 * mu)
    deltaP = 64 * mu / (np.pi * D ** 2) * (P1 - P2) ** 2 / L ** 2
    return deltaP, v

# 参数设置
D = np.linspace(0.1, 0.5, 20)
L = np.linspace(10, 100, 20)
P1 = 1e5
P2 = 0
mu = 1e-3

# 计算压力损失和流体速度
deltaP, v = np.meshgrid(np.zeros(len(D)), np.zeros(len(L)))
for i in range(len(D)):
    for j in range(len(L)):
        deltaP[i, j], v[i, j] = delta_P(D[i], L[j], P1, P2, mu)

# 绘制图像
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('管道内径和长度对压力损失和流体速度的影响', fontsize=16)

# 压力损失-管道内径和长度的关系
im1 = ax[0].contourf(D, L, deltaP, levels=20)
ax[0].set_xlabel('管道内径(m)')
ax[0].set_ylabel('管道长度(m)')
ax[0].set_title('压力损失')
cbar1 = fig.colorbar(im1, ax=ax[0])

# 流体速度-管道内径和长度的关系
im2 = ax[1].contourf(D, L, v, levels=20)
ax[1].set_xlabel('管道内径(m)')
ax[1].set_ylabel('管道长度(m)')
ax[1].set_title('流体速度')
cbar2 = fig.colorbar(im2, ax=ax[1])

plt.show()
