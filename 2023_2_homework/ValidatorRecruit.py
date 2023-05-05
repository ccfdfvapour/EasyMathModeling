import gym
from gym import spaces
import numpy as np


class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 定义观测空间
        self.observation_space = spaces.Box(low=0, high=100, shape=(5, 3), dtype=np.float32)

        # 定义动作空间
        self.action_space = spaces.Discrete(5)

        # 初始化环境参数
        self.R_O = [81, 72, 76, 77, 82, 85, 82, 78, 84, 84, 81, 83, 82, 75, 84, 77, 77, 81, 78, 79]  # 设置validators的计算资源
        self.L = [89, 97, 99, 85, 102, 92, 99, 103, 91, 97, 104, 82, 110, 124, 110, 97]  # 设置16个validators的有效链路
        self.Q_O = [84, 87, 79, 82, 78, 82, 83, 76, 82, 84, 81, 92, 76, 75, 68, 86, 86, 79, 86, 81]  # 设置validators的信誉值
        self.a_w = [0.5, 0.4, 0.6, 0.53, 0.2]
        self.b_w = [0.5, 0.6, 0.4, 0.47, 0.8]
        self.m_1 = 0.5
        self.m_2 = 0.5
        self.k_1 = 0.5
        self.k_2 = 0.5
        self.T_max = 500
        self.b_size = 100
        self.l = 100
        self.V = [1, 2, 3, 4, 5, 6]

        self.T = np.zeros((20, 5, 20))  # 创建一个20行5列20页的矩阵存放每个metaverse对validator的收益评价
        self.R = self.R_O
        self.Q = self.Q_O
        self.A_w = self.a_w
        self.B_w = self.b_w
        self.Sum = 0

        self.T = np.zeros((20, 5, 20))  # 创建一个20行5列20页的矩阵存放每个metaverse对validator的收益评价
        self.C = np.zeros((5, 21, 2))

    def step(self, action):
        your_action = action  # 你的动作

        # 执行动作后改变某些变量导致state发生改变

        self.state = np.array([self.T, self.C])  # 你想要的观测

        reward = 111  # 你想要的奖励（一个数）

        done = False  # 判断一轮环境是否结束
        self.t += 1  # 环境时间加1
        if self.t == 20 - 1:  # 都执行完了
            done = True

        return self.state, reward, done

    def reset(self):
        # 一轮环境结束后重置状态，包括任意需要重置的变量以及state
        self.t = 0
        self.T = np.zeros((20, 5, 20))  # 创建一个20行5列20页的矩阵存放每个metaverse对validator的收益评价
        self.C = np.zeros((5, 21, 2))
        self.state = np.array([self.T, self.C])  #
        return self.state


def random_test():
    # 初始化环境
    env = MyEnv()
    env.reset()
    # 随机动作
    action = env.action_space.sample()
    done = False
    while not done:
        state, reward, done = env.step(action)
        print("########状态########")
        print(state)
        print("########奖励########")
        print(reward)
        print("########是否结束########")
        print(done)


random_test()  # 测试环境
