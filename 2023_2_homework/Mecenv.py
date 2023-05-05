import gym
import numpy as np

class MECEnv(gym.Env):
    def __init__(self, nodes = [
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.1},
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.2},
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.3},
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.4},
    ],
    cloud = {'cpu': 5, 'delay': 0.5}):
        self.nodes = nodes
        self.cloud = cloud
        self.action_space = gym.spaces.Discrete(len(nodes) + 1)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(len(nodes),), dtype=np.float32)
        self.reset()
        self.target_time = 10
        self.energy_limit = 30

    def reset(self):
        self.state = [node['cpu'] for node in self.nodes]
        self.time = 0
        self.reward = 0
        return self.state

    def step(self, action):
        if action < len(self.nodes):
            node = self.nodes[action]
            if node['cpu'] > 0:
                node['cpu'] -= 1
                self.time += node['delay']
                self.reward -= node['delay']
        else:
            self.cloud['cpu'] -= 1
            self.time += self.cloud['delay']
            self.reward -= self.cloud['delay']
        self.state = [node['cpu'] for node in self.nodes]
        return self.state, self.reward, self.time >= self.target_time, {}

    def render(self):
        print('Time: %.1f' % self.time)
        print('State: ', self.state)

    def set_target(self, target_time, energy_limit):
        self.target_time = target_time
        self.energy_limit = energy_limit

def main():
    nodes = [
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.1},
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.2},
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.3},
        {'cpu': 5, 'max_cpu': 5, 'delay': 0.4},
    ]
    cloud = {'cpu': 5, 'delay': 0.5}
    env = MECEnv(nodes, cloud)
    env.set_target(10, 30)
    state = env.reset()
    done = False
    while not done:
        action = np.random.randint(len(nodes) + 1)
        state, reward, done, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
