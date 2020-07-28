import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import tensorflow as tf

tf.config.optimizer.set_jit(True)

w = 200
h = 200


@tf.function
def normalize(x):
    return x / tf.linalg.norm(x)


@tf.function
def tf_dot(A, B):
    return tf.tensordot(A, B, axes=1)


@tf.function
def intersect_triangle(O, D, a, b, c, epsilon=1e-9):
    N = -1 * normalize(tf.linalg.cross(c - a, b - a))
    d = tf_dot(N, a)
    t_num = d - tf_dot(N, O)
    t_dem = tf_dot(N, D) + epsilon
    t = t_num / t_dem
    P = O + t * D
    d = intersect_plane(O, D, P, N)
    if tf.math.is_inf(d):
        return np.inf
    # check edge 1
    e1 = b - a
    vp1 = P - a
    c1 = tf.linalg.cross(e1, vp1)
    if tf_dot(N, c1) < 0:
        return np.inf
    # check edge 2
    e2 = c - b
    vp2 = P - b
    c2 = tf.linalg.cross(e2, vp2)
    if tf_dot(N, c2) < 0:
        return np.inf
    # check edge 1
    e3 = a - c
    vp3 = P - c
    c3 = tf.linalg.cross(e3, vp3)
    if tf_dot(N, c3) < 0:
        return np.inf
    return d


@tf.function
def intersect_plane(O, D, P, N):
    """
    Return the distance from O to the intersection of the ray (O, D) with the
    plane (P, N), or +inf if there is no intersection.
    O and P are 3D points, D and N (normal) are normalized vectors.
    """
    denom = tf_dot(D, N)
    if tf.math.abs(denom) < 1e-6:
        return np.inf
    else:
        d = tf_dot(P - O, N) / denom
        if d < 0:
            return np.inf
        else:
            return d


@tf.function
def intersect_sphere(O, D, S, R):
    """
    Return the distance from O to the intersection of the ray (O, D) with the
    sphere (S, R), or +inf if there is no intersection.
    O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    """
    a = tf_dot(D, D)
    OS = O - S
    b = 2 * tf_dot(D, OS)
    c = tf_dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = tf.math.sqrt(disc)
        if b < 0:
            q = (-1 * b - distSqrt) / 2.0
        else:
            q = (-1 * b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = tf.math.minimum(t0, t1), tf.math.maximum(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
        else:
            return np.inf
    else:
        return np.inf


def intersect(O, D, obj):
    if obj["type"] == "plane":
        return intersect_plane(O, D, obj["position"], obj["normal"])
    elif obj["type"] == "sphere":
        return intersect_sphere(O, D, obj["position"], obj["radius"])
    elif obj["type"] == "triangle":
        return intersect_triangle(O, D, obj["a"], obj["b"], obj["c"])


def get_normal(obj, M):
    # Find normal.
    if obj["type"] == "sphere":
        N = normalize(M - obj["position"])
    elif obj["type"] == "plane":
        N = obj["normal"]
    elif obj["type"] == "triangle":
        N = obj["normal"]
    return N


def get_color(obj, M):
    color = obj["color"]
    if not hasattr(color, "__len__"):
        color = color(M)
    return color


@tf.function
def calculate_M(rayO, rayD, t):
    return rayO + rayD * t


@tf.function
def calculate_to_L_O(L, O, M):
    toL = normalize(L - M)
    toO = normalize(O - M)
    return toL, toO


def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = calculate_M(rayO, rayD, t)
    # Find properties of the object.
    N = get_normal(obj, M)
    color = get_color(obj, M)
    toL, toO = calculate_to_L_O(L, O, M)
    # Shadow: find if the point is shadowed or not.
    l = [
        intersect(M + N * 0.0001, toL, obj_sh)
        for k, obj_sh in enumerate(scene)
        if k != obj_idx
    ]
    if l and min(l) < np.inf:
        return
    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get("diffuse_c", diffuse_c) * max(tf_dot(N, toL), 0.0) * color
    # Blinn-Phong shading (specular).
    col_ray += (
        obj.get("specular_c", specular_c)
        * max(tf_dot(N, normalize(toL + toO)), 0) ** specular_k
        * color_light
    )
    return obj, M, N, col_ray


def add_triangle(a, b, c, color):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    return dict(
        type="triangle",
        a=a,
        b=b,
        c=c,
        normal=normalize(tf.linalg.cross(c - a, b - a)),
        color=np.array(color, dtype=np.float32),
        reflection=0.5,
    )


def add_sphere(position, radius, color):
    return dict(
        type="sphere",
        position=np.array(position, dtype=np.float32),
        radius=np.array(radius, dtype=np.float32),
        color=np.array(color, dtype=np.float32),
        reflection=0.5,
    )


def add_plane(position, normal):
    return dict(
        type="plane",
        position=np.array(position, dtype=np.float32),
        normal=np.array(normal, dtype=np.float32),
        color=lambda M: (
            color_plane0 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else color_plane1
        ),
        diffuse_c=0.75,
        specular_c=0.5,
        reflection=0.25,
    )


# List of objects.
color_plane0 = 1.0 * np.ones(3)
color_plane1 = 0.0 * np.ones(3)

scene = [
    add_sphere([0.0, 0.5, 0.0], 0.3, [1, 0, 0]),
]


def get_rand():
    return random.random() / 10


for i in range(-1, 2):
    for j in range(-1, 2):
        print(i, j)
        UL = [i + 1, j]
        UR = [i + 1, j + 1]
        LL = [i, j]
        LR = [i, j + 1]
        scene += [
            add_triangle(
                [LL[0], get_rand(), LL[1]],
                [UL[0], get_rand(), UL[1]],
                [UR[0], get_rand(), UR[1]],
                [0, 1, 1],
            )
        ]
        h_xy = random.random()
        scene += [
            add_triangle(
                [UR[0], get_rand(), UR[1]],
                [LR[0], get_rand(), LR[1]],
                [LL[0], get_rand(), LL[1]],
                [0, 1, 1],
            )
        ]
        # break
    # break

scene += [add_plane([0.0, -2.0, 0.0], [0.0, 1.0, 0.0])]

# Light position and color.
L = np.array([5.0, 5.0, -10.0], dtype=np.float32)
color_light = np.ones(3)

# Default light and material parameters.
ambient = 0.05
diffuse_c = 1.0
specular_c = 1.0
specular_k = 50

depth_max = 3  # Maximum number of light reflections.
col = np.zeros(3)  # Current color.
O = np.array([0.0, 1.0, -1.0], dtype=np.float32)  # Camera.
Q = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Camera pointing to.
img = np.zeros((h, w, 3), dtype=np.float32)

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1.0, -1.0 / r + 0.25, 1.0, 1.0 / r + 0.25)


@tf.function
def create_new_ray(rayD, M, N):
    rayO = M + N * 0.0001
    rayD = normalize(rayD - 2 * tf_dot(rayD, N) * N)
    return rayO, rayD


# Loop through all pixels.
for i, x in tqdm(enumerate(np.linspace(S[0], S[2], w)), total=w):
    for j, y in enumerate(np.linspace(S[1], S[3], h)):
        col[:] = 0
        Q[:2] = (x, y)
        D = normalize(Q - O)
        depth = 0
        rayO, rayD = O, D
        reflection = 1.0
        # Loop through initial and secondary rays.
        while depth < depth_max:
            traced = trace_ray(rayO, rayD)
            if not traced:
                break
            obj, M, N, col_ray = traced
            # Reflection: create a new ray.
            rayO, rayD = create_new_ray(rayD, M, N)
            depth += 1
            try:
                col += reflection * col_ray.numpy()
            except:
                col += reflection * col_ray
            reflection *= obj.get("reflection", 1.0)
        img[h - j - 1, i, :] = tf.clip_by_value(col, 0, 1)

plt.imsave("fig.png", img)

