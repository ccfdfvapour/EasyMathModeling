import gym
from gym import spaces
import numpy as np

np.random.seed(1)  # 固定随机种子


class MultiKnapsackEnv(gym.Env):
    """
    自定义多背包问题的强化学习环境
    """

    def __init__(self, num_knapsacks=3, capacity_ratio=0.5):
        """
        初始化环境

        :param num_knapsacks: 背包数量，默认为3
        :param capacity_ratio: 每个背包的容量比例，即每个背包可以放置的物品重量总和与所有物品重量总和的比例，默认为0.5
        """

        super().__init__()

        self.num_knapsacks = num_knapsacks
        self.capacity_ratio = capacity_ratio

        # 状态空间为一个数组，其中每个元素表示一个物品的重量
        self.observation_space = spaces.Box(low=0, high=10, shape=(10,))

        # 动作空间为一个离散的空间，表示将一个物品放入哪个背包
        self.action_space = spaces.Discrete(num_knapsacks)

        # 背包容量初始化，每个背包的容量为总重量的capacity_ratio倍
        self.capacity = np.zeros(num_knapsacks)
        self.total_capacity = 0
        self.items = None

        # 重置环境
        self.reset()

    def reset(self):
        """
        重置环境
        """

        # 初始化物品，每个物品的重量在[1, 10]之间
        self.items = np.random.randint(low=1, high=10, size=10)

        # 初始化背包容量
        self.total_capacity = np.sum(self.items) * self.capacity_ratio
        self.capacity = np.full(shape=self.num_knapsacks, fill_value=self.total_capacity)

        # 当前步数和总奖励
        self.current_step = 0
        self.total_reward = 0

        # 返回初始状态
        return self.capacity

    def step(self, action):
        """
        执行一步动作

        :param action: 动作，表示将哪个物品放入哪个背包
        :return: 状态、奖励、是否结束、额外信息
        """

        # 将物品放入背包
        item_weight = self.items[self.current_step]
        self.capacity[action] -= item_weight

        # 判断是否结束
        self.current_step += 1

        # 计算奖励
        reward = 0
        info = [self.current_step,action]
        if self.capacity[action] >= 0:
            reward = item_weight
            done = False
        else:
            reward = -item_weight
            done = True
            info = [self.current_step, action, item_weight]

        if self.current_step == len(self.items):
            done = True

        # 更新总奖励
        self.total_reward += reward

        # 返回状态、奖励、是否结束、额外信息
        return self.capacity, self.total_reward, done, info

    def render(self, mode='human'):
        """
        可选的渲染环节
        """
        pass


def random_test():
    # 初始化环境
    env = MultiKnapsackEnv()
    state = env.reset()
    print("########初始状态########")
    print(state)
    done = False
    while not done:
        # 随机动作
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print("########状态########")
        print('每个背包的剩余容量:', state, '奖励:', reward, 'done:', done, '步数/分配背包/超量价值:', info)


random_test()  # 测试环境
