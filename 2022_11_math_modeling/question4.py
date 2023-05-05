import pandas as pd
import numpy as np
import statsmodels.api as sm #统计运算
import scipy.stats as scs #科学计算
import scipy.optimize as sco
import matplotlib.pyplot as plt #绘图
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

temp=pd.read_excel('资产指数.xlsx')

# 计算每年指数回报率
index_returns = temp[temp.columns[1:]].pct_change()

#给不同资产随机分配初始权重,所有的权重系数均在0-1之间
weights = np.random.random(4)
weights /= np.sum(weights)

# 投资组合优化使得夏普率最大
def stats(weights):
    weights = np.array(weights)
    port_returns = np.sum(index_returns.mean()*weights)
    port_variance = np.sqrt(np.dot(weights.T, np.dot(index_returns.cov(),weights)))
    return np.array([port_returns, port_variance, port_returns/port_variance])

#最小化夏普指数的负值
def min_sharpe(weights):
    return -stats(weights)[2]
#给定初始权重
x0 = 4*[1./4]
#权重（某股票持仓比例）限制在0和1之间。
bnds = tuple((0,1) for x in range(4))
#权重（股票持仓比例）的总和为1。
cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})
opts = sco.minimize(min_sharpe,
                    x0,
                    method = 'SLSQP',
                    bounds = bnds,
                    constraints = cons)

#最优投资组合权重向量
print('最优投资组合权重')
print(opts['x'].round(5))
