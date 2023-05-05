import random
import math
import matplotlib.pyplot as plt


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def total_distance(path, cities):
    return sum(distance(cities[path[i]], cities[path[i + 1]]) for i in range(len(path) - 1)) + distance(
        cities[path[-1]], cities[path[0]])


def simulated_annealing(cities, temp, temp_min, alpha, visual=True):
    num_cities = len(cities)
    path = list(range(num_cities))
    random.shuffle(path)
    curr_distance = total_distance(path, cities)

    distances = [curr_distance]

    if visual:
        plt.ion()
        plt.show()

        x = [city[0] for city in cities]
        y = [city[1] for city in cities]

        plt.plot(x, y, 'bo-')
        plt.title('Initial Path')
        plt.draw()
        plt.pause(0.001)

    while temp > temp_min:
        for i in range(num_cities):
            j = random.randint(0, num_cities - 1)
            path[i], path[j] = path[j], path[i]
            new_distance = total_distance(path, cities)
            delta = new_distance - curr_distance
            if delta < 0 or random.random() < math.exp(-delta / temp):
                curr_distance = new_distance
            else:
                path[i], path[j] = path[j], path[i]
            distances.append(curr_distance)

        temp *= alpha

        if visual:
            plt.clf()
            plt.plot(x, y, 'bo-')
            for i in range(num_cities - 1):
                plt.plot([cities[path[i]][0], cities[path[i + 1]][0]], [cities[path[i]][1], cities[path[i + 1]][1]],
                         'r-')
            plt.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], 'r-')
            plt.title('Iteration {}, Distance {:.2f}'.format(len(distances), curr_distance))
            plt.draw()
            plt.pause(0.001)

    if visual:
        plt.ioff()
        plt.clf()
        plt.plot(x, y, 'bo-')
        for i in range(num_cities - 1):
            plt.plot([cities[path[i]][0], cities[path[i + 1]][0]], [cities[path[i]][1], cities[path[i + 1]][1]], 'r-')
        plt.plot([cities[path[-1]][0], cities[path[0]][0]], [cities[path[-1]][1], cities[path[0]][1]], 'r-')
        plt.title('Final Path, Distance {:.2f}'.format(curr_distance))
        plt.show()

    return path, curr_distance, distances


if __name__ == '__main__':
    cities = [
        (16.4700, 96.1000),
        (16.4700, 94.4400),
        (20.0900, 92.5400),
        (22.3900, 93.3700),
        (25.2300, 97.2400),
        (22.0000, 96.0500),
        (20.4700, 97.0200),
        (17.2000, 96.2900),
        (16.3000, 97.3800),
        (14.0500, 98.1200),
        (16.5300, 97.3800),
        (21.5200, 95.5900),
        (19.4100, 97.1300),
        (20.0900, 92.5500)
    ]
    temp = 1e10
    temp_min = 1e-30
    alpha = 0.99

    path, distance, distances = simulated_annealing(cities, temp, temp_min, alpha)

    print('Best path found:', path)
    print('Total distance:', distance)

    plt.plot(range(len(distances)), distances)
