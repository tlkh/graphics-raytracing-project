import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import multiprocessing
import cv2
import math


threads = multiprocessing.cpu_count()

w = 100
h = 100


def normalize(x):
    return x / np.linalg.norm(x)


eijk = np.array([[[0.,  0.,  0.],
                  [0.,  0.,  1.],
                  [0., -1.,  0.]],
                 [[0.,  0., -1.],
                  [0.,  0.,  0.],
                  [1.,  0.,  0.]],
                 [[0.,  1.,  0.],
                  [-1.,  0.,  0.],
                  [0.,  0.,  0.]]])


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
    if obj.type == 'sphere':
        return intersect_sphere(O, D, obj.center, obj.radius)
    elif obj.type == 'triangle':
        return intersect_triangle(O, D, obj.a, obj.b, obj.c)


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
    N = obj.get_normal(M)
    color = obj.get_color(M)
    toL = normalize(L - M)
    toO = normalize(O - M)
    # Shadow: find if the point is shadowed or not.
    # l = [intersect(M + N * .0001, toL, obj_sh)
    #     for k, obj_sh in enumerate(scene) if k != obj_idx]
    # if l and min(l) < np.inf:
    #   return
    # Start computing the color.
    col_ray = ambient
    # Lambert shading (diffuse).
    col_ray += obj.get_diffuse_c() * max(np.dot(N, toL), 0) * color
    # Blinn-Phong shading (specular).
    col_ray += obj.get_specular_c() * max(np.dot(N, normalize(toL + toO)),
                                          0) ** specular_k * color_light
    return obj, M, N, col_ray


class Triangle(object):
    def __init__(self, a, b, c, color, diffuse_c=0.1, specular_c=0.5, reflection=0.5):
        self.type = "triangle"
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.normal = self.compute_normal()
        self.color = np.array(color, dtype=float)
        self.reflection = reflection
        self.diffuse_c = diffuse_c
        self.specular_c = specular_c

    def compute_normal(self):
        return normalize(einsum_cross(self.c-self.a,
                                      self.b-self.a))

    def get_normal(self, coords=None):
        return self.normal

    def get_color(self, coords=None):
        return self.color

    def get_reflection(self, coords=None):
        return self.reflection

    def get_diffuse_c(self, coords=None):
        return self.diffuse_c

    def get_specular_c(self, coords=None):
        return self.specular_c


class Sphere(object):
    def __init__(self, center, radius, color, diffuse_c=.9, specular_c=.1, reflection=0.1):
        self.type = "sphere"
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.color = np.array(color, dtype=float)
        self.reflection = reflection
        self.diffuse_c = diffuse_c
        self.specular_c = specular_c

    def get_normal(self, coords):
        return -1.0 * normalize(coords - self.center)

    def get_color(self, coords=None):
        point_color = self.color
        point_color[0] = coords[1]/5
        return point_color

    def get_reflection(self, coords=None):
        return self.reflection

    def get_diffuse_c(self, coords=None):
        return self.diffuse_c

    def get_specular_c(self, coords=None):
        return self.specular_c


scene = []


def xWave(j, time, height,squished):
    val = math.sin(squished*j+2*time) + math.sin(squished*2*j +
                                               0.5*time + 1) + math.sin(squished*2*j+time)

    return (height)*val

def yWave(j, time, height,squished):
    return height*math.sin(squished*j+0.5*time)

def diaWave(i, j, time, weight):
    return (weight)*math.sin(weight*3*(j - i) + time)

# higher val for squish means more the wave more squished
def gety(i, j, time):
    y = 0
    #y += diaWave(i,j,time,0.3)
    #y += yWave(j,time,0.3)
    #y += xWave(i,time,0.3)

    big = 2
    #y += diaWave(i,j,time,big)
    #y += yWave(j,time,15, big)
    #y += xWave(i, time, big)
    y += yWave(j,time,5,big)
    y += xWave(j,time,1,5)
    y += yWave(i,time,5,big)
    y += xWave(i,time,1,5)
    return y


time = 1
denom = 50

x_coords = list(np.linspace(-2.0, 2.0, num=9, endpoint=True))
y_coords = list(np.linspace(-2.0, 2.0, num=9, endpoint=True))

for i, x in enumerate(x_coords[:-1]):
    for j, y in enumerate(y_coords[:-1]):
        UL = np.asarray([x_coords[i+1], y], dtype=float)
        UR = np.asarray([x_coords[i+1], y_coords[j+1]], dtype=float)
        LL = np.asarray([x, y], dtype=float)
        LR = np.asarray([x, y_coords[j+1]], dtype=float)

        ul = gety(x_coords[i+1], y, time)/denom
        ur = gety(x_coords[i+1], y_coords[j+1], time)/denom
        ll = gety(x, y, time)/denom
        lr = gety(x, y_coords[j+1], time)/denom

        scene += [Triangle(a=[LL[0], ll, LL[1]],
                           b=[UL[0], ul, UL[1]],
                           c=[UR[0], ur, UR[1]],
                           color=[0.0, 0.2, 0.3])]
        scene += [Triangle(a=[UR[0], ur, UR[1]],
                           b=[LR[0], lr, LR[1]],
                           c=[LL[0], ll, LL[1]],
                           color=[0.0, 0.2, 0.3])]
        # break
    # break

scene += [Sphere(center=[0.0, 0.0, 0.0],
                 radius=10.0,
                 color=[135/255, 0.7, 0.8])]

print("Num objects:", len(scene))

# Light position and color.
L = np.array([5., 5., -10.])
color_light = np.ones(3)

# Default light and material parameters.
ambient = 0.01
diffuse_c = 1.
specular_c = 1.
specular_k = 50

depth_max = 3  # Maximum number of light reflections.
O = np.array([0.0, 1.0, -2.0])  # Camera.
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
        reflection *= obj.get_reflection()
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
