import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 定义模型参数
p = 100  # 售出价
c = np.linspace(0, 100, 1000)  # 订购费
h = 5  # 贮存费
f = lambda x: 1/100*(x+20) if 0 <= x <= 80 else 0  # 需求密度函数
D_mean = quad(lambda x: x*f(x), 0, np.inf)[0]
D_var =  quad(lambda x: (x-D_mean)**2*f(x), 0, np.inf)[0]  # 需求的均值和方差

# 定义目标函数
def objective(Q, c):
    integral1 = quad(lambda r: r*f(r), 0, Q)[0]
    integral2 = quad(lambda r: f(r), Q, np.inf)[0]
    return p*(integral1 + Q*integral2) - c*Q - h*Q

# 定义平均利润随订购费变化的函数
def profit(c):
    # 求解最优订购量
    objective_c = lambda Q: -objective(Q, c)
    res = minimize_scalar(objective_c, bounds=(0, D_mean), method='bounded')
    Q_star = res.x
    # 计算期望贮存量
    integral1 = quad(lambda r: r*f(r), 0, Q_star)[0]
    integral2 = quad(lambda r: f(r), Q_star, np.inf)[0]
    E_Q = integral1 + integral2
    # 计算平均利润
    L_star = p*E_Q - h*E_Q - c*Q_star
    return L_star

# 计算平均利润随订购费变化的曲线
L = np.array([profit(c_i) for c_i in c])

# 求解最优订购量和最优订购费
objective_star = lambda x: -objective(x[0], x[1])
res = minimize(objective_star, [D_mean/2, c[0]], method='BFGS')
Q_star, c_star = res.x

# 绘制订购费曲线
plt.plot(c, L)
plt.xlabel('订购费')
plt.ylabel('平均利润')
plt.title('平均利润随订购费变化')
plt.axvline(c_star, color='r', linestyle='--', label=f'最优订购费={c_star:.2f}')
plt.legend()
plt.show()
