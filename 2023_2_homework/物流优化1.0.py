import numpy as np
import matplotlib.pyplot as plt

# 定义问题参数
m = 6
n = 8
d = 4
C = np.random.randint(10, 100, size=(m, n))
Q = np.random.randint(5, 10, size=d)
Q_min = np.min(Q)
Q_max = np.max(Q)
W = np.random.randint(10, 20, size=d)
W_min = np.min(W)
W_max = np.max(W)

# 定义粒子群算法参数
num_particles = 50
num_iterations = 100
w = 0.5
c1 = 1
c2 = 1
v_min = -1
v_max = 1


# 定义目标函数和约束函数
def objective_function(x):
    return np.sum(C * x.reshape(m, n))


def constraint_function(x):
    violations = 0
    for i in range(d):
        Q_i = np.sum(x * W[i])
        if Q_i < Q_min or Q_i > Q_max:
            violations += 1
    return violations


# 定义PSO算法
class Particle:
    def __init__(self, num_dimensions):
        self.position = np.random.randint(0, 2, size=num_dimensions)
        self.velocity = np.zeros(num_dimensions)
        self.best_position = np.copy(self.position)
        self.best_fitness = np.inf


class PSO:
    def __init__(self, num_particles, num_dimensions, w, c1, c2, v_min, v_max):
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_min = v_min
        self.v_max = v_max
        self.particles = [Particle(num_dimensions) for i in range(num_particles)]
        self.global_best_position = np.zeros(num_dimensions)
        self.global_best_fitness = np.inf

    def update(self):
        for particle in self.particles:
            # 更新速度
            r1 = np.random.random(self.num_dimensions)
            r2 = np.random.random(self.num_dimensions)
            particle.velocity = self.w * particle.velocity + self.c1 * r1 * (
                        particle.best_position - particle.position) + self.c2 * r2 * (
                                            self.global_best_position - particle.position)
            particle.velocity = np.clip(particle.velocity, self.v_min, self.v_max)

            # 更新位置
            particle.position = particle.position + particle.velocity
            particle.position = np.clip(particle.position, 0, 1)

            # 更新个体最优解
            fitness = objective_function(particle.position) + 100 * constraint_function(particle.position)
            if fitness < particle.best_fitness:
                particle.best_position = np.copy(particle.position)
                particle.best_fitness = fitness

            # 更新全局最优解
            if fitness < self.global_best_fitness:
                self.global_best_position = np.copy(particle.position)
                self.global_best_fitness = fitness


# 初始化PSO算法
pso = PSO(num_particles, m * n, w, c1, c2, v_min, v_max)

# 运行PSO算法
fitness_history = np.zeros(num_iterations)
for t in range(num_iterations):
    pso.update()
    fitness_history[t] = pso.global_best_fitness


# 可视化粒子群算法迭代结果图
plt.plot(fitness_history)
plt.title('Fitness History')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.show()

# 将最终结果转化为二进制矩阵
solution = np.around(pso.global_best_position).astype(int).reshape(m, n)

# 输出结果
print('最优解：')
print(solution)
print('最优解的目标函数值：')
print(pso.global_best_fitness)

# 可视化最优解方案图
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(C, cmap='cool')
for i in range(m):
    for j in range(n):
        if solution[i, j] == 1:
            circle = plt.Circle((j+0.5, i+0.5), 0.2, color='red')
            ax.add_patch(circle)
plt.title('Optimal Solution')
plt.axis('off')
plt.show()

