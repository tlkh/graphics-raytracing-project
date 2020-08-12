import argparse
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import random
import cv2
import math
import ops
import objects

parser = argparse.ArgumentParser()

parser.add_argument(
    "--width", help="Width of output image", type=int, default=64, required=False
)
parser.add_argument(
    "--height", help="Height of output image", type=int, default=64, required=False
)
parser.add_argument(
    "--grid", help="Grid size of the wave mesh", type=int, default=10, required=False
)
parser.add_argument(
    "--duration",
    help="Duration of animation to export",
    type=int,
    default=1,
    required=False,
)
parser.add_argument(
    "--fps",
    help="Frame per second of duration to export",
    type=int,
    default=1,
    required=False,
)
parser.add_argument(
    "--threads",
    help="Number of threads to use",
    type=int,
    default=multiprocessing.cpu_count(),
    required=False,
)
parser.add_argument(
    "--reflection_depth",
    help="Maximum recursion depth for reflection",
    type=int,
    default=4,
    required=False,
)
parser.add_argument(
    "--transmission_depth",
    help="Maximum recursion depth for transmission",
    type=int,
    default=2,
    required=False,
)
parser.add_argument(
    "--mirrorball", help="Draw the mirror ball in the scene", action="store_true", default=False, required=False
)

args = parser.parse_args()


def trace_ray(rayO, rayD):
    t = np.inf
    for group in scene:
        # print("[debug] Testing intersection with", group)
        intersected = False
        bounds = scene[group]["bounds"]
        t_obj = bounds.intersect(rayO, rayD)
        if t_obj < t:
            # print("[debug] Intersected with", group)
            objects = scene[group]["objects"]
            for obj in objects:
                # print("[debug] Testing intersection with", obj.type)
                t_obj = obj.intersect(rayO, rayD)
                if t_obj < t:
                    # print("[debug] Intersected with", obj.type)
                    t = t_obj
                    intersected_object = obj
                    intersected = True
                    break
        if intersected:
            break
    if t == np.inf:
        return None
    obj = intersected_object
    # print("[debug]", obj.type)
    M = rayO + rayD * t
    # get properties of object
    N = obj.get_normal(M)
    color = obj.get_color(M)
    toL = ops.normalize(L - M)
    toO = ops.normalize(O - M)
    # compute color
    col_ray = ambient_c
    col_ray += obj.get_diffuse_c() * max(np.dot(N, toL), 0) * color
    col_ray += (
        obj.get_specular_c()
        * max(np.dot(N, ops.normalize(toL + toO)), 0) ** obj.get_specular_k()
        * color_light
    )
    return obj, M, N, col_ray


def shade_pixel(x, y, q_z, depth_max):
    col = np.zeros(3)
    Q = np.array([x, y, q_z])
    D = ops.normalize(Q - O)
    depth = 0
    rayO, rayD = O, D
    reflection = 1.0
    transmission_depth = 0
    while depth < depth_max:
        traced = reflect_transmit_rays(
            rayO, rayD, col, reflection, transmission_depth, args.transmission_depth
        )
        if traced:
            rayO, rayD, col, reflection, transmission_depth = traced
        else:
            break
        depth += 1
    return col


def reflect_transmit_rays(
    og_rayO, og_rayD, col, reflection, transmission_depth, max_transmission_depth
):
    traced = trace_ray(og_rayO, og_rayD)
    if not traced:
        return False
    # reflection: create a new ray
    obj, M, N, col_ray = traced
    rf_rayO = M + N * 0.001
    rf_rayD = ops.normalize(og_rayD - 2 * np.dot(og_rayD, N) * N)
    col_rf = reflection * col_ray
    transmission_depth += 1
    opacity = obj.get_opacity()
    transparency = 1.0 - opacity
    if transmission_depth <= max_transmission_depth:
        # transmission of ray through transparent object
        tr_rayO = M - N * 0.001
        tf_rayD = og_rayD
        transmitted = reflect_transmit_rays(
            tr_rayO,
            tf_rayD,
            transparency * col,
            reflection,
            transmission_depth,
            args.transmission_depth,
        )
        if transmitted:
            _, _, col_tr, _, _ = transmitted
            col += transparency * col_tr
    col += opacity * col_rf
    reflection *= obj.get_reflection()
    col = np.clip(col, 0.0, 1.0)
    return rf_rayO, rf_rayD, col, reflection, transmission_depth


# scene properties
ambient_c = 0.01
color_light = np.array([1.0, 1.0, 1.0])

# camera position
O = np.array([0.0, 0.75, -6.0])
# camera orientation
Q = np.array([0.0, 0.0, 0.0])

r = args.width / args.height
# screen coordinates: (x0, y0, x1, y1)
S = (-1.0, -1.0 / r + 0.25, 1.0, 1.0 / r + 0.25)

# wave mesh properties
wave_y_scale = 50
x_coords = list(np.linspace(-3.0, 3.0, num=args.grid, endpoint=True))
y_coords = list(np.linspace(-4.0, 9.0, num=args.grid, endpoint=True))

# animation properties
total_frames = args.fps * args.duration
h, w = args.height, args.width

print("Starting to render", total_frames, "frames...")

for frame in trange(total_frames):
    scene = {}
    wave_mesh = []
    time = frame * (1 / args.fps)
    min_x, max_x = 1e9, -1 * 1e9
    min_y, max_y = 1e9, -1 * 1e9
    min_z, max_z = 1e9, -1 * 1e9
    counter = 1

    if args.mirrorball:
        scene["mirrorball"] = {
        "bounds": objects.Sphere(center=[0.0, 0.7, -2.0], radius=0.2),
        "objects": [objects.Sphere(center=[0.0, 0.7, -2.0],
                                    color=[0.9, 0.9, 0.9],
                                    radius=0.2)],
        }

    for i, x in enumerate(x_coords[:-1]):
        for j, y in enumerate(y_coords[:-1]):
            # this `y` is misleading...
            # taking the ground plane (actually X-Z) as X-Y
            UL = np.asarray([x_coords[i + 1], y], dtype=float)
            UR = np.asarray([x_coords[i + 1], y_coords[j + 1]], dtype=float)
            LL = np.asarray([x, y], dtype=float)
            LR = np.asarray([x, y_coords[j + 1]], dtype=float)

            ul = ops.gety(x_coords[i + 1], y, time) / wave_y_scale
            ur = ops.gety(x_coords[i + 1], y_coords[j + 1], time) / wave_y_scale
            ll = ops.gety(x, y, time) / wave_y_scale
            lr = ops.gety(x, y_coords[j + 1], time) / wave_y_scale

            # calculate the bounding box
            max_y = max([max_y, ul, ur, ll, lr])
            min_y = min([min_y, ul, ur, ll, lr])
            max_x, min_x = (
                max([max_x, x, x_coords[i + 1]]),
                min([min_x, x, x_coords[i + 1]]),
            )
            max_z, min_z = (
                max([max_z, y, y_coords[j + 1]]),
                min([min_z, y, y_coords[j + 1]]),
            )

            wave_mesh += [
                objects.Triangle(
                    a=[LL[0], ll, LL[1]],
                    b=[UL[0], ul, UL[1]],
                    c=[UR[0], ur, UR[1]],
                    color=[0.0, 0.1, 0.2],
                ),
                objects.Triangle(
                    a=[UR[0], ur, UR[1]],
                    b=[LR[0], lr, LR[1]],
                    c=[LL[0], ll, LL[1]],
                    color=[0.0, 0.1, 0.2],
                ),
            ]

            if len(wave_mesh) > 4:
                bounding_box = objects.Box(
                    vmin=[min_x, min_y, min_z], vmax=[max_x, max_y, max_z]
                )
                scene_wave = {"bounds": bounding_box, "objects": wave_mesh}
                scene["wave_part_" + str(counter)] = scene_wave
                counter += 1
                wave_mesh = []
                min_x, max_x = 1e9, -1 * 1e9
                min_y, max_y = 1e9, -1 * 1e9
                min_z, max_z = 1e9, -1 * 1e9

    if len(wave_mesh) > 0:
        # remaining triangles
        bounding_box = objects.Box(
            vmin=[min_x, min_y, min_z], vmax=[max_x, max_y, max_z]
        )
        scene_wave = {"bounds": bounding_box, "objects": wave_mesh}
        scene["wave" + str(counter)] = scene_wave

    scene["ocean_floor"] = {
        "bounds": objects.Box(vmin=[-9.0, -0.31, -9.0], vmax=[9.9, -0.29, 9.0]),
        "objects": [
            objects.Plane(
                position=[0.0, -0.3, 0.0],
                normal=[0.0, 1.0, 0.0],
                color=[0.3, 0.1, 0.0],
                reflection=0.0,
            ),
        ],
    }

    class EnvSphere(objects.SceneObject):
        def __init__(
            self, center, radius, diffuse_c=0.8, specular_c=0.5, reflection=0.01
        ):
            self.type = "env_sphere"
            self.center = np.array(center, dtype=float)
            self.radius = radius
            self.reflection = reflection
            self.diffuse_c = diffuse_c
            self.specular_c = specular_c

        def get_normal(self, coords):
            return -1.0 * ops.normalize(coords - self.center)

        def get_color(self, coords=None):
            global time
            start_color = np.array([0.9, 0.9, 1.0])  # start bright blue-white
            end_color = np.array([0.8, 0.6, 0.3])  # become red-orange
            color_gradient = end_color - start_color
            k = (time / 12) ** 2
            point_color = (1 - k) * start_color + k * color_gradient
            point_color[2] = (1 - time / 12) * point_color[2] + time / 10 * coords[
                1
            ] / 3
            return np.clip(point_color, 0.1, 1.0)

        def intersect(self, rayO, rayD):
            a = np.dot(rayD, rayD)
            rayOS = rayO - self.center
            b = 2 * np.dot(rayD, rayOS)
            c = np.dot(rayOS, rayOS) - self.radius ** 2
            disc = b * b - 4 * a * c
            if disc > 0:
                sqrt_disc = np.sqrt(disc)
                if b < 0:
                    q = (-b - sqrt_disc) / 2.0
                else:
                    q = (-b + sqrt_disc) / 2.0
                t0 = q / a
                t1 = c / q
                t0, t1 = min(t0, t1), max(t0, t1)
                if t0 < 0 and t1 > 0:
                    return t1
            return np.inf

    scene["environment"] = {
        "bounds": EnvSphere(center=[0.0, 0.0, 0.0], radius=10.0),
        "objects": [EnvSphere(center=[0.0, 0.0, 0.0], radius=10.0)],
    }

    print("Scene groups:", len(scene))
    # for group in scene:
    #    print("Num objects in", group+":", len(scene[group]["objects"]))
    total_num = sum([len(scene[group]["objects"]) for group in scene])
    print("Total num objects:", total_num)

    # light position and animation
    light_t = time
    light_x = 0
    light_y = 5 * math.cos(light_t / 4) + 4.5
    light_z = np.clip(light_t, -1, 9.5)
    L = np.array([light_x, light_y, light_z])

    img = np.zeros((h, w, 3))

    if args.threads:
        coords = []
        index_counter = 0
        for i, x in enumerate(np.linspace(S[0], S[2], w)):
            for j, y in enumerate(np.linspace(S[1], S[3], h)):
                coords.append([x, y, Q[2], args.reflection_depth])
        print("Rendering frame with", args.threads, "threads...")
        with multiprocessing.Pool(processes=args.threads) as pool:
            results = pool.starmap(shade_pixel, coords)
        print("Completed frame!")
        for i, x in enumerate(np.linspace(S[0], S[2], w)):
            for j, y in enumerate(np.linspace(S[1], S[3], h)):
                img[h - j - 1, i, :] = results[index_counter]
                index_counter += 1
    else:
        # single threaded implementation
        for i, x in tqdm(enumerate(np.linspace(S[0], S[2], w)), total=w):
            for j, y in enumerate(np.linspace(S[1], S[3], h)):
                img[h - j - 1, i, :] = shade_pixel(x, y, Q[2], args.reflection_depth)

    # upscale and write out image
    output_img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    output_img = np.clip(output_img, 0.0, 1.0).astype(np.float)
    plt.imsave(str(frame) + ".png", output_img)

