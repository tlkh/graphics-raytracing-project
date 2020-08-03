import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import multiprocessing
import cv2


threads = multiprocessing.cpu_count()

w = 100
h = 100


def normalize(x):
    return x / np.linalg.norm(x)

eijk = np.array([[[ 0.,  0.,  0.],
                  [ 0.,  0.,  1.],
                  [ 0., -1.,  0.]],
                 [[ 0.,  0., -1.],
                  [ 0.,  0.,  0.],
                  [ 1.,  0.,  0.]],
                 [[ 0.,  1.,  0.],
                  [-1.,  0.,  0.],
                  [ 0.,  0.,  0.]]])

def einsum_cross(u, v):
    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    return np.einsum('ijk,uj,vk->uvi', eijk, u, v)[0, 0, :]

def intersect_triangle(O, D, a, b, c):
    N = -1*normalize(einsum_cross(c-a, b-a))
    d = N.dot(a)
    t_num = d - (N.dot(O))
    t_dem = N.dot(D)
    t = t_num / t_dem
    P = O + t * D
    d = intersect_plane(O, D, P, N)
    if d == np.inf:
        return np.inf
    # check edge 1
    e1 = b - a
    vp1 = P - a
    c1 = einsum_cross(e1, vp1)
    if N.dot(c1) < 0:
        return np.inf
    # check edge 2
    e2 = c-b
    vp2 = P-b
    c2 = einsum_cross(e2, vp2)
    if N.dot(c2) < 0:
        return np.inf
    # check edge 3
    e3 = a-c
    vp3 = P - c
    c3 = einsum_cross(e3, vp3)
    if N.dot(c3) < 0:
        return np.inf
    return d
        
def intersect_plane(O, D, P, N):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # plane (P, N), or +inf if there is no intersection.
    # O and P are 3D points, D and N (normal) are normalized vectors.
    denom = np.dot(D, N)
    if np.abs(denom) < 1e-6:
        return np.inf
    d = np.dot(P - O, N) / denom
    if d < 0:
        return np.inf
    return d


def intersect_sphere(O, D, S, R):
    # Return the distance from O to the intersection of the ray (O, D) with the
    # sphere (S, R), or +inf if there is no intersection.
    # O and S are 3D points, D (direction) is a normalized vector, R is a scalar.
    a = np.dot(D, D)
    OS = O - S
    b = 2 * np.dot(D, OS)
    c = np.dot(OS, OS) - R * R
    disc = b * b - 4 * a * c
    if disc > 0:
        distSqrt = np.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t0 < 0 and t1 > 0:
            return t1
    return np.inf


def intersect(O, D, obj):
    if obj['type'] == 'plane':
        return intersect_plane(O, D, obj['position'], obj['normal'])
    elif obj['type'] == 'sphere':
        return intersect_sphere(O, D, obj['position'], obj['radius'])
    elif obj['type'] == 'triangle':
        return intersect_triangle(O, D, obj['a'], obj['b'], obj['c'])


def get_normal(obj, M):
    # Find normal.
    if obj['type'] == 'sphere':
        N = -1*normalize(M - obj['position'])
    elif obj['type'] == 'plane':
        N = obj['normal']
    elif obj['type'] == 'triangle':
        N = obj['normal']
    return N


def get_color(obj):
    color = obj['color']
    return color


def trace_ray(rayO, rayD):
    # Find first point of intersection with the scene.
    t = np.inf
    for i, obj in enumerate(scene):
        t_obj = intersect(rayO, rayD, obj)
        if t_obj < t:
            t, obj_idx = t_obj, i
            break
    # Return None if the ray does not intersect any object.
    if t == np.inf:
        return
    # Find the object.
    obj = scene[obj_idx]
    # Find the point of intersection on the object.
    M = rayO + rayD * t
    # Find properties of the object.
    N = get_normal(obj, M)
    color = obj.get('color')
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Shadow: find if the point is shadowed or not.
    #l = [intersect(M + N * .0001, toL, obj_sh)
    #     for k, obj_sh in enumerate(scene) if k != obj_idx]
    #if l and min(l) < np.inf:
    #   return
    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get('diffuse_c', diffuse_c) * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get('specular_c', specular_c) * max(np.dot(N,
                                                              normalize(toL + toO)), 0) ** specular_k * color_light
    return obj, M, N, col_ray


def add_triangle(a, b, c, color):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    return dict(type='triangle', a=a, b=b, c=c,
                normal=normalize(einsum_cross(c-a, b-a)),
                color=np.array(color), reflection=0.6,
                diffuse_c=.1, specular_c=.8)


def add_sphere(position, radius, color):
    return dict(type='sphere', position=np.array(position),
                radius=np.array(radius), color=np.array(color),
                diffuse_c=.9, specular_c=.1, reflection=0.1)


def add_plane(position, normal, color):
    return dict(type='plane', position=np.array(position),
                normal=np.array(normal),
                color=np.array(color),
                diffuse_c=.75, specular_c=.5, reflection=.25)


# List of objects.
color_plane0 = 1. * np.ones(3)
color_plane1 = 0. * np.ones(3)

scene = []

def get_rand():
    return random.random()/20.0

for i in range(-3, 4):
    for j in range(-3, 4):
        UL = np.asarray([i+1, j], dtype=float)
        UR = np.asarray([i+1, j+1], dtype=float)
        LL = np.asarray([i, j], dtype=float)
        LR = np.asarray([i, j+1], dtype=float)
        scene += [add_triangle([LL[0], get_rand(), LL[1]], [UL[0], get_rand(), UL[1]], [UR[0], get_rand(), UR[1]],
                               [0, 0.2, 0.3])]
        h_xy = random.random()
        scene += [add_triangle([UR[0], get_rand(), UR[1]], [LR[0], get_rand(), LR[1]], [LL[0], get_rand(), LL[1]],
                               [0, 0.2, 0.3])]
        #break
    #break
    
scene += [add_sphere([0.0, 0.0, 0.0], 10.0, [135/255, 206/255, 235/255]), ]
    
print("Num objects:", len(scene))

#scene += [add_plane([0., -2.0, 0.], [0., 1., 0.])]

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)

# Default light and material parameters.
ambient = 0.01
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 3  # Maximum number of light reflections.
O = np.array([0.0, 0.5, -1.0])  # Camera.
Q = np.array([0.0, 0.0, 0.0])  # Camera pointing to.
img = np.zeros((h, w, 3))

r = float(w) / h
# Screen coordinates: x0, y0, x1, y1.
S = (-1., -1. / r + .25, 1., 1. / r + .25)

def shade_pixel(x, y, q_z=0, depth_max=3):
    col = np.zeros(3)
    Q = np.array([x, y, q_z])
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
        rayO = M + N * .0001
        rayD = normalize(rayD - 2 * np.dot(rayD, N) * N)
        depth += 1
        col += reflection * col_ray
        reflection *= obj.get('reflection')
    return col

THREAD = True

if THREAD:
    for i, x in tqdm(enumerate(np.linspace(S[0], S[2], w)), total=w):
        coords = []
        index_counter = 0
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            coords.append([x, y, Q[2], depth_max])
        with multiprocessing.Pool(processes=threads) as pool:
            results = pool.starmap(shade_pixel, coords)
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            img[h - j - 1, i, :] = results[index_counter]
            index_counter += 1
else:
    for i, x in tqdm(enumerate(np.linspace(S[0], S[2], w)), total=w):
        for j, y in enumerate(np.linspace(S[1], S[3], h)):
            img[h - j - 1, i, :] = shade_pixel(x, y, Q[2], depth_max)

img = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
img = np.clip(img, 0.0, 1.0).astype(np.float)
plt.imsave('fig.png', img)