import numpy as np
import tools.filer as filer


def function(x, y):
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def prepared_data(min_=-1, max_=1, n=100, func=function):
    x = np.random.uniform(min_, max_, n)
    y = np.random.uniform(min_, max_, n)
    z = func(x, y)

    indices = np.random.permutation(n)
    x, y, z = x[indices], y[indices], z[indices]
    xy = np.array([[x[i], y[i]] for i in range(n)], dtype=np.float32)

    return xy, z.astype(np.float32)


def processed_data():
    data = filer.read_from_dir()
    output_data = {}
    for network in data.keys():
        for statistics in data[network]:
            if statistics['batch_size'] == 1 and statistics['epochs'] == 100:
                output_data[network[0:3]]['batch'].append(statistics[''])
                output_data[network[0:3]]['batch'].append(statistics[''])
            print()
