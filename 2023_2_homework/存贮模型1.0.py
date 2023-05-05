import numpy as np
import matplotlib.pyplot as plt

# 参数设置
T = 12  # 时间总数
I_0 = 20  # 初始存储量
I_max = 60  # 最大存储量
D_t = [50, 40, 60, 70, 80, 60, 70, 90, 80, 60, 50, 40]  # 每月的需求量
c = 2  # 订购成本
h = 0.5  # 存储成本

# 初始化DP数组
DP = np.zeros((I_max+1, T+1))

# 动态规划求解
for t in range(1, T+1):
    for i in range(I_max+1):
        min_cost = float('inf')
        for u in range(i+1):
            cost = h*(i-u) + c*u + DP[min(i-u+D_t[t-1], I_max), t-1]
            if cost < min_cost:
                min_cost = cost
        DP[i, t] = min_cost

# 计算最优订购策略
order = np.zeros(T)
I_t = I_0
for t in range(T-1, -1, -1):
    for u in range(I_t+1):
        cost = h*(I_t-u) + c*u + DP[min(I_t-u+D_t[t], I_max), t]
        if cost == DP[I_t, t+1]:
            order[t] = u
            I_t = min(I_t-u+D_t[t], I_max)
            break

# 敏感性分析
costs = []
I_max_list = range(20, 101, 5)
for i_max in I_max_list:
    DP = np.zeros((i_max+1, T+1))
    for t in range(1, T+1):
        for i in range(i_max+1):
            min_cost = float('inf')
            for u in range(i+1):
                cost = h*(i-u) + c*u + DP[min(i-u+D_t[t-1], i_max), t-1]
                if cost < min_cost:
                    min_cost = cost
            DP[i, t] = min_cost
    costs.append(DP[I_0, T])

# 可视化
plt.figure(figsize=(8, 6))
plt.plot(I_max_list, costs)
plt.xlabel('Maximum Inventory Level')
plt.ylabel('Total Cost')
plt.title('Sensitivity Analysis')
plt.show()

# 可视化最优的订购策略
plt.figure(figsize=(8, 6))
plt.plot(range(1, T+1), order)
plt.xlabel('Time')
plt.ylabel('Order Quantity')
plt.title('Optimal Ordering Strategy')
plt.show()
