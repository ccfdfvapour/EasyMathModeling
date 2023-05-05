# modelCovid5_v1.py
# Demo01 of mathematical modeling for Covid2019
# ESEIR model for epidemic diseases (改进的 SEIR 模型)
# Copyright 2021 Youcans, XUPT
# Crated：2021-06-16
# Python小白的数学建模课 @ Youcans

# 1. SEIR2 模型，考虑潜伏期具有传染性
from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np  # 导入 numpy包
import matplotlib.pyplot as plt  # 导入 matplotlib包


def dySIS(y, t, lamda, mu):  # SI/SIS 模型，导数函数
    dy_dt = lamda * y * (1 - y) - mu * y  # di/dt = lamda*i*(1-i)-mu*i
    return dy_dt


def dySIR(y, t, lamda, mu):  # SIR 模型，导数函数
    s, i = y  # youcans
    ds_dt = -lamda * s * i  # ds/dt = -lamda*s*i
    di_dt = lamda * s * i - mu * i  # di/dt = lamda*s*i-mu*i
    return np.array([ds_dt, di_dt])


def dySEIR(y, t, lamda, delta, mu):  # SEIR 模型，导数函数
    s, e, i = y
    ds_dt = - lamda * s * i  # ds/dt = -lamda*s*i
    de_dt = lamda * s * i - delta * e  # de/dt = lamda*s*i - delta*e
    di_dt = delta * e - mu * i  # di/dt = delta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


def dyESEIR(y, t, lamda, lamE, delta, mu):  # 改进后的ESEIR 模型，导数函数
    s, e, i = y
    ds_dt = - lamda * s * i - lamE * s * e  # ds/dt = -lamda*s*i - lamE*s*e
    de_dt = lamda * s * i + lamE * s * e - delta * e  # de/dt = lamda*s*i - delta*e
    di_dt = delta * e - mu * i  # di/dt = delta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


# 设置模型参数
number = 1e5  # 总人数
lamda = 1.0  # 日接触率, 患病者每天有效接触的易感者的平均人数
lamE = 0.25  # 日接触率E, 潜伏者每天有效接触的易感者的平均人数
delta = 0.05  # 日发病率，每天发病成为患病者的潜伏者占潜伏者总数的比例
mu = 0.05  # 日治愈率, 每天治愈的患病者人数占患病者总数的比例
sigma = lamda / mu  # 传染期接触数
fsig = 1 - 1 / sigma
tEnd = 200  # 预测日期长度
t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
i0 = 1e-3  # 患病者比例的初值
e0 = 0.0001  # 潜伏者比例的初值
s0 = 1 - i0  # 易感者比例的初值
Y0 = (s0, e0, i0)  # 微分方程组的初值
colormap = ['r', 'c', 'y', 'g', 'b', 'm']
# odeint 数值解，求解微分方程初值问题
ySI = odeint(dySIS, i0, t, args=(lamda, 0))  # SI 模型
ySIS = odeint(dySIS, i0, t, args=(lamda, mu))  # SIS 模型
ySIR = odeint(dySIR, (s0, i0), t, args=(lamda, mu))  # SIR 模型
ySEIR = odeint(dySEIR, Y0, t, args=(lamda, delta, mu))  # SEIR 模型
yESEIR = odeint(dyESEIR, Y0, t, args=(lamda, lamE, delta, mu))  # 改进后的ESEIR 模型

# # 分析日接触率λ(lambda)的影响
# lamdalist = [0.125, 0.25, 0.5, 1.0, 2.0]
# plt.title("Impact of $\lambda$ in ESEIR model")
# plt.xlabel('Days')
# plt.axis([0, tEnd, -0.1, 1.1])
# for lb in range(len(lamdalist)):
#     yESEIR_i0 = odeint(dyESEIR, Y0, t, args=(lamdalist[lb], lamE, delta, mu))  # SEIR2 模型
#     plt.plot(t, yESEIR_i0[:, 2], '-' + colormap[lb], label='i(t)-' + '$\lambda=$' + str(lamdalist[lb]))
#     plt.plot(t, yESEIR_i0[:, 0], '--' + colormap[lb], label='s(t)-' + '$\lambda=$' + str(lamdalist[lb]))
# plt.legend(loc='center right')  # youcans
# plt.show()

# # 分析日接触率λ(lambda)的影响
# lamdaElist = [0.01, 0.05, 0.125, 0.25, 0.5]
# plt.title("Impact of $\lambda_E$ in ESEIR model")
# plt.xlabel('Days')
# plt.axis([0, tEnd, -0.1, 1.1])
# for lb in range(len(lamdaElist)):
#     yESEIR_i0 = odeint(dyESEIR, Y0, t, args=(lamda, lamdaElist[lb], delta, mu))  # SEIR2 模型
#     plt.plot(t, yESEIR_i0[:, 2], '-' + colormap[lb], label='i(t)-' + '$\lambda_E=$' + str(lamdaElist[lb]))
#     plt.plot(t, yESEIR_i0[:, 0], '--' + colormap[lb], label='s(t)-' + '$\lambda_E=$' + str(lamdaElist[lb]))
# plt.legend(loc='center right')  # youcans
# plt.show()

# # 分析日发病率δ(delta)的影响
# deltalist = [0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
# plt.title("Impact of $\delta$ in ESEIR model")
# plt.xlabel('Days')
# plt.axis([0, tEnd, -0.1, 1.1])
# for lb in range(len(deltalist)):
#     yESEIR_i0 = odeint(dyESEIR, Y0, t, args=(lamda, lamE, deltalist[lb], mu))  # SEIR2 模型
#     plt.plot(t, yESEIR_i0[:, 2], '-' + colormap[lb], label='i(t)-' + '$\delta=$' + str(deltalist[lb]))
#     plt.plot(t, yESEIR_i0[:, 0], '--' + colormap[lb], label='s(t)-' + '$\delta=$' + str(deltalist[lb]))
# plt.legend(loc='center right')  # youcans
# plt.show()

# # 分析日治愈率μ(mu)的影响
# mulist = [0.025, 0.05, 0.1, 0.2, 0.4]
# plt.title("Impact of $\mu$ in ESEIR model")
# plt.xlabel('Days')
# plt.axis([0, tEnd, -0.1, 1.1])
# for lb in range(len(mulist)):
#     yESEIR_i0 = odeint(dyESEIR, Y0, t, args=(lamda, lamE, delta, mulist[lb]))  # SEIR2 模型
#     plt.plot(t, yESEIR_i0[:, 2], '-' + colormap[lb], label='i(t)-' + '$\mu=$' + str(mulist[lb]))
#     plt.plot(t, yESEIR_i0[:, 0], '--' + colormap[lb], label='s(t)-' + '$\mu=$' + str(mulist[lb]))
# plt.legend(loc='center right')  # youcans
# plt.show()

# # 分析初值条件i0、e0、s0初始条件的影响
# i_0 = [0.0001, 0.0005, 0.001, 0.005]
# plt.title("Impact of $i_0$ in ESEIR model")
# plt.xlabel('Days')
# plt.axis([0, tEnd, -0.1, 1.1])
# for lb in range(len(i_0)):
#     yESEIR_i0 = odeint(dyESEIR, (s0, e0, i_0[lb]), t, args=(lamda, lamE, delta, mu))  # SEIR2 模型
#     plt.plot(t, yESEIR_i0[:, 2], '-' + colormap[lb], label='i(t)-' + '$i_0=$' + str(i_0[lb]))
#     plt.plot(t, yESEIR_i0[:, 0], '--' + colormap[lb], label='s(t)-' + '$i_0=$' + str(i_0[lb]))
# plt.legend(loc='center right')  # youcans
# plt.show()


# # SEIR 和改进后的SEIR模型对比
# print("lamda={}\tmu={}\tsigma={}\t(1-1/sig)={}".format(lamda, mu, sigma, fsig))
# plt.title("Comparison between SEIR and ESEIR model")
# plt.xlabel('Days')
# plt.axis([0, tEnd, -0.1, 1.1])
#
# # plt.plot(t, ySI, 'cadetblue', label='i(t)-SI')
# # plt.plot(t, ySIS, 'steelblue', label='i(t)-SIS')
# # plt.plot(t, ySIR[:, 1], 'cornflowerblue', label='i(t)-SIR')
#
# plt.plot(t, yESEIR[:, 0], '-' + colormap[0], label='s(t)-ESEIR')  # 易感者比例
# plt.plot(t, yESEIR[:, 1], '-' + colormap[1], label='e(t)-ESEIR')  # 潜伏者比例
# plt.plot(t, yESEIR[:, 2], '-' + colormap[2], label='i(t)-ESEIR')  # 患病者比例
# # plt.plot(t, 1-yESEIR[:,0]-yESEIR[:,1]-yESEIR[:,2], '-b', label='r(t)-ESEIR')
# plt.plot(t, ySEIR[:, 0], '--' + colormap[0], label='s(t)-SEIR')
# plt.plot(t, ySEIR[:, 1], '--' + colormap[1], label='e(t)-SEIR')
# plt.plot(t, ySEIR[:, 2], '--' + colormap[2], label='i(t)-SEIR')
# # plt.plot(t, 1-ySEIR[:,0]-ySEIR[:,1]-ySEIR[:,2], '--m', label='r(t)-SEIR')
# plt.legend(loc='upper right')  # youcans
# plt.show()


# # 改进SEIR 模型的相轨迹分析
# e0List = np.arange(0.01,0.6,0.1)  # (start,stop,step)
# for e0 in e0List:
#     # odeint 数值解，求解微分方程初值问题
#     i0 = 0  # 潜伏者比例的初值
#     s0 = 1 - i0 - e0  # 易感者比例的初值
#     ySEIR = odeint(dySEIR, (s0,e0,i0), t, args=(lamda,delta,mu))  # SEIR 模型
#     plt.plot(ySEIR[:,1], ySEIR[:,2])  # (e(t),i(t))
#     print("lamda={}\tdelta={}\mu={}\tsigma={}\ti0={}\te0={}".format(lamda,delta,mu,lamda/mu,i0,e0))
#
# plt.title("Phase trajectory of ESEIR models: e(t)~i(t)")
# plt.axis([0, 0.6, 0, 0.6])
# plt.plot([0,0.6],[0,0.35],'y--')  #[x1,x2][y1,y2]
# plt.plot([0,0.6],[0,0.18],'y--')  #[x1,x2][y1,y2]
# plt.text(0.02,0.36,r"$\lambda=0.3, \delta=0.1, \mu=0.1$",color='black')
# plt.xlabel('e(t)')
# plt.ylabel('i(t)')
# plt.show()

# 分析不同隔离方案对模型的影响
lamdalist = [1.0, 0.3, 0.3]
lamElist = [0.25, 0.25, 0.125]
policylist = ['No Isolation','Only isolate the sick','Isolate sick and lurkers']
plt.title("Impact of Isolation policy in SEIR and ESEIR model")
plt.xlabel('Days')
plt.axis([0, tEnd, -0.1, 1.1])
for lb in range(len(lamdalist)):
    yESEIR_i0 = odeint(dyESEIR, Y0, t, args=(lamdalist[lb], lamElist[lb], delta, mu))  # SEIR2 模型
    plt.plot(t, yESEIR_i0[:, 2], '-' + colormap[lb],
             label='i(t)-' + policylist[lb])
    plt.plot(t, yESEIR_i0[:, 0], '--' + colormap[lb],
             label='s(t)-' + policylist[lb])
plt.legend(loc='center right')
plt.show()
# '$\lambda=$' + str(lamdalist[lb]) + '$,\lambda_E=$' + str(lamElist[lb])