import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import random
import imageio
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

np.random.seed(1)


def draw_path(path, coords,color=' '):
    xs, ys = coords[path].T
    xs = np.concatenate([xs, [xs[0]]])
    ys = np.concatenate([ys, [ys[0]]])
    plt.plot(xs, ys, 'o-'+color)


def output_path(path):
    arrow = '->'
    p = [str(i + 1) for i in path]
    return arrow.join(p)


def path_length(distances, path):
    return np.sum(distances[path[:-1], path[1:]])


def metropolis(current_path, new_path, distances, temperature):
    current_length = path_length(distances, current_path)
    new_length = path_length(distances, new_path)
    if new_length < current_length:
        return new_path, new_length
    else:
        delta = new_length - current_length
        acceptance_probability = np.exp(-delta / temperature)
        if random.random() < acceptance_probability:
            return new_path, new_length
        else:
            return current_path, current_length


def new_answer(path):
    np.random.seed(1)
    i = random.randint(1, len(path) - 1)
    j = random.randint(1, len(path) - 1)
    new_path = path.copy()
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def simulated_annealing(coords, temperature, cooling_rate, improve):
    np.random.seed(1)
    distances = squareform(pdist(coords))
    n = len(coords)
    current_path = np.random.permutation(n)
    current_length = path_length(distances, current_path)
    count = 0
    obj = [current_length]
    path = [current_path]

    # 动态定义初始温度和初始冷却速率
    temperature = initial_temperature
    alpha = cooling_rate
    iterations = 20000
    with imageio.get_writer('path_animation.gif', mode='I', fps=90) as writer:
        pbar = tqdm(total=iterations, desc="Eval_eps", unit="it")  # 初始化进度条
        while count < iterations:
        # while temperature > 1e-8:  # 1e-8
            count += 1
            new_path = new_answer(current_path)
            current_path, current_length = metropolis(current_path, new_path, distances, temperature)
            if current_length < obj[-1]:
                obj.append(current_length)
                path.append(current_path)
            else:
                obj.append(obj[-1])
                path.append(path[-1])

            if imporve:
                # 动态调整温度和冷却速率
                if count % 10 == 0:
                    delta = obj[-1] - obj[-11]
                    t = -delta / np.log(alpha)
                    temperature = min(max(0.5 * temperature, t), 1.5 * temperature)
                    alpha = max(0.95 * alpha, 0.5)

                else:
                    temperature *= cooling_rate
            else:
                temperature *= cooling_rate

            # # Dynamically plot the  path
            # plt.clf()
            # draw_path(path[-1], city_coords)
            # plt.title('Iteration {}, Distance {:.2f}'.format(count, obj[-1]))
            # plt.draw()
            # plt.pause(0.0001)
            # image = plt.gcf()
            # if isinstance(image, np.ndarray):
            #     writer.append_data(image)
            # else:
            #     plt.savefig('temp.png')
            #     image = imageio.imread('temp.png')
            #     writer.append_data(image)

            pbar.update()
    return path[-1], obj, path

def load_data(data_path):
    """
    导入数据，得到城市坐标信息
    :param data_path: 数据文件地址 str
    :return: 所有城市的坐标信息 二维 list
    """
    cities = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            x_str, y_str = line.split()[1:]
            x, y = int(x_str), int(y_str)
            cities.append((x, y))
    return cities

# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data

# Generate random city coordinates
np.random.seed(1)
num_cities = 20
# city_coords = np.array(load_data('./cities.txt'))
city_coords = np.array(read_tsp('./oliver30.tsp'))[:,1:3]

# np.random.rand(num_cities, 2)

# Run simulated annealing
initial_temperature = 100000.0#1.0
cooling_rate = 0.995#0.8
epoch = 4#20

colormap = ['r', 'c', 'y', 'g', 'b', 'm']
# results = []
# results2 = []
# for i in range(epoch):
#     imporve = True
#     best_path, distances, paths = simulated_annealing(city_coords, initial_temperature, cooling_rate, imporve)
#     results.append(distances)
#     imporve = False
#     best_path2, distances2, paths2 = simulated_annealing(city_coords, initial_temperature, cooling_rate, imporve)
#     results2.append(distances2)
#
#
# # Plot the results
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# colormap = ['r', 'c', 'y', 'g', 'b', 'm']
# # Plot the distance vs iteration
#
# ax1.plot(np.arange(len(distances)), np.mean(results, axis=0),'-' + colormap[1],label='Adaptive-SA')
# ax1.fill_between(np.arange(len(distances)), np.min(results, axis=0), np.max(results, axis=0),where=np.ones(len(distances)), color=colormap[1], alpha=0.2)
# ax1.plot(np.arange(len(distances2)), np.mean(results2, axis=0),'-' + colormap[0],label='SA')
# ax1.fill_between(np.arange(len(distances2)), np.min(results2, axis=0), np.max(results2, axis=0),where=np.ones(len(distances2)), color=colormap[0], alpha=0.2)
# ax1.set_xlabel('Iteration')
# ax1.set_ylabel('Distance')
# ax1.set_title('Distance vs Iteration')
# ax1.legend(loc='center right')
#
# # Plot the best path
# draw_path(best_path, city_coords,colormap[1])
# #draw_path(best_path2, city_coords,colormap[0])
# ax2.set_title('Best Path')
# plt.show()
#
# # Output the best path and distance
# print('Best Path:', output_path(best_path))
# print('Total Distance:', path_length(squareform(pdist(city_coords)), best_path))


result3 = []
result4 = []
cooling_ratelist = [0.5,0.7,0.9,0.99,0.995,0.999]
for cooling_rate in cooling_ratelist:
    imporve = True
    best_path3, distances3, paths3 = simulated_annealing(city_coords, initial_temperature, cooling_rate, imporve)
    imporve = False
    best_path4, distances4, paths4 = simulated_annealing(city_coords, initial_temperature, cooling_rate, imporve)
    result3.append(distances3[-1])
    result4.append(distances4[-1])
    # 绘制对比图
plt.plot(cooling_ratelist, result3, color=colormap[1],label='Adaptive-SA')
plt.plot(cooling_ratelist, result4, color=colormap[0],label='SA')
plt.legend()
plt.xlabel('Cooling rate')
plt.ylabel('Optimal path length')
plt.title('Comparison of two algorithms with different cooling rate')
plt.show()




