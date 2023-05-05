import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# 参数设定
# m = 5  # 工厂数量
# n = 4  # 客户数量
# d = np.array([240, 250, 195, 100, 135])  # 工厂容量
# c = np.array([8, 9, 6, 5])  # 客户需求量
# q = np.array([[2, 3, 1, 4],
#               [2, 1, 4, 3],
#               [3, 1, 2, 4],
#               [1, 2, 3, 4],
#               [1, 3, 2, 4]])  # 工厂-客户配送费用矩阵
np.random.seed(1)
m = 10  # 工厂数量
n = 20  # 客户数量
d = np.random.randint(low=1000, high=3000, size=m)  # 工厂容量，随机生成
c = np.random.randint(low=50, high=300, size=n)  # 客户需求量，随机生成
q = np.random.randint(low=1, high=20, size=(m, n))  # 工厂-客户配送费用矩阵，随机生成

# 惯性权重决定了粒子的移动方向，个体加速常数和全局加速常数分别表示粒子个体和全局的影响程度，粒子速度上限和粒子位置下限和粒子位置上限是对粒子移动的约束。
# 粒子群算法参数设定
num_particles = 2000  # 粒子数量
num_iterations = 30  # 迭代次数
w = 0.7  # 惯性权重
c1 = 1.5  # 个体加速常数
c2 = 1.5  # 全局加速常数
v_max = 2  # 粒子速度上限
x_min = 0  # 粒子位置下限
x_max = 1  # 粒子位置上限

# 初始化粒子位置和速度
particles = np.random.uniform(x_min, x_max, size=(num_particles, m * n))
velocities = np.zeros((num_particles, m * n))

# 定义适应度函数和约束函数

# 总费用作为适应度，越小越好
def objective_function(x):
    total_cost = 0
    for i in range(m):
        for j in range(n):
            total_cost += x[i * n + j] * q[i][j] * c[j]
    return total_cost

# 检查每个客户的配送量是否超过了其需求量和每个工厂的配送量是否超过了其容量，来限制可行解的搜索空间。
def constraint_function(x):
    excess_demand = np.maximum(0, np.sum(x.reshape(m, n), axis=0) - c)
    excess_capacity = np.maximum(0, d - np.sum(x.reshape(m, n), axis=1))
    return np.sum(excess_demand) + np.sum(excess_capacity)


# 初始化全局最优解
global_best = particles[0]
fitness_history = np.zeros(num_iterations)

# 可视化粒子在搜索空间内的移动趋势和速度变化
fig, ax = plt.subplots(nrows=num_iterations, ncols=2, figsize=(10, 100))
fig.subplots_adjust(hspace=1.0)
# 开始迭代
for t in range(num_iterations):
    for i in range(num_particles):
        # 更新粒子速度
        r1 = np.random.uniform(size=(m * n,))
        r2 = np.random.uniform(size=(m * n,))
        velocities[i] = w * velocities[i] + c1 * r1 * (particles[i] - particles[i]) + \
                        c2 * r2 * (global_best - particles[i])
        velocities[i] = np.minimum(v_max, np.maximum(-v_max, velocities[i]))

        # 更新粒子位置
        particles[i] = np.minimum(x_max, np.maximum(x_min, particles[i] + velocities[i]))

        # 更新个体最优解
        if objective_function(particles[i]) + 450 * constraint_function(particles[i]) < \
                objective_function(global_best) + 450 * constraint_function(global_best):
            global_best = particles[i].copy()

    # 打印每次迭代后的全局最优解适应度值
    print(f"Iteration {t + 1}: {objective_function(global_best)}")
    fitness_history[t] = objective_function(global_best)


#     # 可视化当前迭代中所有粒子的位置分布
#     for i, alpha in enumerate(np.linspace(0.3, 1, num_iterations)):
#         ax[t][0].scatter(particles[:, 0], particles[:, 1], alpha=alpha, c=range(num_particles), cmap='viridis')
#     ax[t][0].set_title(f"Iteration {t + 1}: Particle Position Distribution", fontsize=8)
#
#     # 可视化当前迭代中所有粒子的速度分布
#     for i, alpha in enumerate(np.linspace(0.3, 1, num_iterations)):
#         ax[t][1].scatter(velocities[:, 0], velocities[:, 1], alpha=alpha, c=range(num_particles), cmap='viridis')
#     ax[t][1].set_title(f"Iteration {t + 1}: Particle Velocity Distribution", fontsize=8)
#
# plt.show()

# 可视化工厂容量和客户需求量的分布情况
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].hist(d, bins=10, color='skyblue')
axs[0].set_xlabel('Factory Capacity')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Distribution of Factory Capacity')

axs[1].hist(c, bins=10, color='lightcoral')
axs[1].set_xlabel('Customer Demand')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Distribution of Customer Demand')

plt.tight_layout()
plt.show()
# 可视化工厂-客户配送费用矩阵的热力图
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(q, cmap='Blues')

# 添加颜色条
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Delivery Cost", rotation=-90, va="bottom")

# 添加文本标签
for i in range(m):
    for j in range(n):
        text = ax.text(j, i, q[i][j],
                       ha="center", va="center", color="w")

# 添加坐标轴
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(m))
ax.set_xticklabels(np.arange(1, n+1))
ax.set_yticklabels(np.arange(1, m+1))

# 添加网格线
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, m, 1), minor=True)
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

ax.set_title("Delivery Cost Matrix")
fig.tight_layout()
plt.show()


# 可视化粒子群算法迭代结果图
plt.plot(fitness_history)
plt.title('Fitness History')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.show()


# 可视化方案
solution = global_best.reshape((m, n))
print(solution)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(solution, cmap='Blues')
for (i, j), z in np.ndenumerate(solution):
    ax.text(j, i, int(z), ha='center', va='center', fontsize=8)
ax.set_xticks(np.arange(-0.5, n, 1))
ax.set_yticks(np.arange(-0.5, m, 1))
ax.grid()
ax.set_title('Best solution found by PSO')
plt.show()
