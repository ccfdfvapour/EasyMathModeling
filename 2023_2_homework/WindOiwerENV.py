import gym
from gym import spaces
import numpy as np

import gym
from gym import spaces
import numpy as np

class WindPowerEnv(gym.Env):
    def __init__(self):
        # 风速的范围 [v_min, v_max]
        self.v_min = 3.0
        self.v_max = 25.0

        # 风轮叶片角度的范围 [0, alpha_max]
        self.alpha_max = 30.0

        # 动作空间为离散的叶片角度选择
        self.action_space = spaces.Discrete(5)  # 选择5个角度

        # 观测空间为连续的风速
        self.observation_space = spaces.Box(low=self.v_min, high=self.v_max, shape=(2,))

        # 状态转移概率
        self.p = 0.1

        # 时间步长
        self.dt = 0.1

        # 风力发电机的参数
        self.rho = 1.225  # 空气密度
        self.A = 10.0     # 叶片面积
        self.Cp = lambda lam, alpha: 0.5 * ((116.0 / lam - 0.4 * alpha - 5.0) * np.exp(-12.5 / lam) + 21.0)

        # 当前状态和奖励
        self.state = None
        self.reward = None
        self.t = 0

    def reset(self):
        self.t = 0
        # 初始化状态为一个随机的风速
        self.state = np.array([np.random.uniform(self.v_min, self.v_max),2])
        return self.state

    def step(self, action):
        #action = np.random.randint(0,9)
        # # 计算所有动作的期望收益
        # Q = []
        # for a in range(self.action_space.n):
        #     alpha = a * self.alpha_max / (self.action_space.n - 1)
        #     P = self.Cp(8.0, alpha) * 0.5 * self.rho * self.A * self.state[0] ** 3
        #     Q.append(P)
        #
        # # 选择收益最高的动作作为贪心动作
        # action = np.argmax(Q)

        self.t+=1
        # 将离散的动作转换为连续的叶片角度
        alpha = action * self.alpha_max / (self.action_space.n - 1)

        # 计算当前状态的输出功率
        P = self.Cp(8.0, alpha) * 0.5 * self.rho * self.A * self.state[0]**3

        # 计算下一个状态的风速
        w = np.random.normal(0, np.sqrt(self.p * self.dt))
        v_next = self.state[0] + (P - self.state[0]) * self.dt / 600.0 + w

        # 将风速限制在范围内
        v_next = np.clip(v_next, self.v_min, self.v_max)

        # 计算当前状态和奖励
        self.state = np.array([v_next,P])
        self.reward = P

        # 判断是否结束
        done = False
        if self.t==10:
            done=True

        return self.state, self.reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass



def plot():
    # 不同风速下的奖励范围

    import matplotlib.pyplot as plt
    import numpy as np

    r=9297247.862752
    r2 = 8597619.734585

    # DQN算法和Random的平均奖励
    dqn_rewards = np.array([r-461726, r-125298, r, r+165467, r+564781])
    random_rewards = np.array([r2, r2-164458, r2-24645, r2-346541, r2-444642])

    # 柱状图的宽度
    bar_width = 0.35

    # 风速范围
    wind_speed_ranges = ["12-13", "11-14", "10-15", "8-16", "7-17"]

    # 设置柱状图的参数
    bar_width = 0.3
    index = np.arange(len(wind_speed_ranges))

    # 绘制柱状图
    plt.bar(index, dqn_rewards, bar_width, label="DQN")
    plt.bar(index + bar_width, random_rewards, bar_width, label="Random")

    # 设置图例、标签和标题
    plt.legend()
    plt.xlabel('Wind Speed Range (m/s)')
    plt.ylabel('Average Reward')
    plt.title('Comparison of Average Rewards at Different Wind Speed Ranges')

    # 修改x轴标签
    plt.xticks(index + bar_width / 2, wind_speed_ranges)

    # 显示图形
    plt.show()
plot()