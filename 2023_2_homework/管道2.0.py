import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 定义常数
L = 100 # 管道长度
Q = 10 # 流量
f = 0.02 # 摩擦系数
rho = 1000 # 流体密度
g = 9.8 # 重力加速度
epsilon = 1e-6 # 管道内壁面粗糙度

# 计算最优的管道内径
d_star = (20*f*L*Q**2/(np.pi**2*g))**(1/5)

# 计算最小的摩擦损失
delta_h_star = 5*f*L*Q**2/(2*np.pi**(7/5)*g**(2/5)*(20*f)**(3/5))

print("最优的管道内径为：{:.2f}米".format(d_star))
print("最小的摩擦损失为：{:.2f}米".format(delta_h_star))

# 生成3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 定义管道内径和摩擦系数的范围
d_range = np.arange(d_star/2, 2*d_star, d_star/100)
f_range = np.arange(f/2, 2*f, f/100)

# 将d_range和f_range转换为网格矩阵
D, F = np.meshgrid(d_range, f_range)

# 计算对应的摩擦损失
delta_h = 4*F*L*Q**2/(np.pi**2*g*D**5)

# 绘制3D图像
ax.plot_surface(D, F, delta_h, cmap='viridis')
ax.set_xlabel('管道内径(m)')
ax.set_ylabel('摩擦系数')
ax.set_zlabel('摩擦损失(m)')

plt.show()
