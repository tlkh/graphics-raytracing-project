import numpy as np
import math


def normalize(x):
    return x / np.linalg.norm(x)


eijk = np.array(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
        [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


def einsum_cross(u, v):
    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    return np.einsum("ijk,uj,vk->uvi", eijk, u, v)[0, 0, :]


def xWave(j, time, height, squished):
    val_a = math.sin(squished * j + 2 * time)
    val_b = math.sin(squished * 2 * j + 0.5 * time + 1)
    val_c = math.sin(squished * 2 * j + time)
    val = val_a + val_b + val_c
    return height * val


def yWave(j, time, height, squished):
    return height * math.sin(squished * j + 0.5 * time)


def diaWave(i, j, time, weight):
    return weight * math.sin(weight * 3 * (j - i) + time)


def gety(i, j, time, big=2):
    y = 0
    y += yWave(j, time, 5, big)
    y += xWave(j, time, 1, 5)
    y += yWave(i, time, 5, big)
    y += xWave(i, time, 1, 5)
    return y

