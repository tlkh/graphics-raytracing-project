import numpy as np
import ops


class SceneObject(object):
    def __init__(self):
        self.color = None
        self.reflection = None
        self.diffuse_c = None
        self.specular_c = None
        self.specular_k = 50

    def get_color(self, coords=None):
        return self.color

    def get_reflection(self, coords=None):
        return self.reflection

    def get_diffuse_c(self, coords=None):
        return self.diffuse_c

    def get_specular_c(self, coords=None):
        return self.specular_c

    def get_specular_k(self, coords=None):
        try:
            return self.specular_k
        except:
            return 50

    def get_opacity(self, coords=None):
        # 1.0 means opaque
        return 1.0


class Triangle(SceneObject):
    def __init__(
        self,
        a,
        b,
        c,
        color=[1.0, 1.0, 1.0],
        diffuse_c=0.3,
        specular_c=0.4,
        reflection=0.5,
    ):
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
        return ops.normalize(ops.einsum_cross(self.c - self.a, self.b - self.a))

    def get_normal(self, coords=None):
        return self.normal

    def get_opacity(self, coords=None):
        return 0.9

    def intersect(self, rayO, rayD):
        N = -1.0 * ops.normalize(ops.einsum_cross(self.c - self.a, self.b - self.a))
        d = np.dot(self.normal, self.a)
        t_num = d - (np.dot(self.normal, rayO))
        t_dem = np.dot(self.normal, rayD)
        t = t_num / t_dem
        P = rayO + t * rayD
        d = self.intersect_plane(rayO, rayD, P, self.normal)
        if d == np.inf:
            return np.inf
        # check edge 1
        e1 = self.b - self.a
        vp1 = P - self.a
        c1 = ops.einsum_cross(e1, vp1)
        if np.dot(N, c1) < 0:
            return np.inf
        # check edge 2
        e2 = self.c - self.b
        vp2 = P - self.b
        c2 = ops.einsum_cross(e2, vp2)
        if np.dot(N, c2) < 0:
            return np.inf
        # check edge 3
        e3 = self.a - self.c
        vp3 = P - self.c
        c3 = ops.einsum_cross(e3, vp3)
        if np.dot(N, c3) < 0:
            return np.inf
        return d

    def intersect_plane(self, rayO, rayD, P, N):
        denom = np.dot(rayD, N)
        if np.abs(denom) < 1e-6:
            return np.inf
        d = np.dot(P - rayO, N) / denom
        if d < 0:
            return np.inf
        return d


class Box(SceneObject):
    def __init__(
        self,
        vmin,
        vmax,
        color=[1.0, 1.0, 1.0],
        diffuse_c=0.5,
        specular_c=0.5,
        reflection=0.5,
    ):
        self.type = "box"
        self.vmin = np.array(vmin, dtype=float)
        self.vmax = np.array(vmax, dtype=float)
        self.color = np.array(color, dtype=float)
        self.reflection = reflection
        self.diffuse_c = diffuse_c
        self.specular_c = specular_c

    def get_normal(self, coords=None):
        # placeholder, we not gonna render the box
        # since we only interested in bounding box
        return np.array([0.0, 1.0, 0.0])

    def intersect(self, rayO, rayD):
        t_1 = (self.vmin[0] - rayO[0]) / rayD[0]
        t_2 = (self.vmax[0] - rayO[0]) / rayD[0]
        tmin = min(t_1, t_2)
        tmax = max(t_1, t_2)

        t_1 = (self.vmin[1] - rayO[1]) / rayD[1]
        t_2 = (self.vmax[1] - rayO[1]) / rayD[1]
        tymin = min(t_1, t_2)
        tymax = max(t_1, t_2)

        t_1 = (self.vmin[2] - rayO[2]) / rayD[2]
        t_2 = (self.vmax[2] - rayO[2]) / rayD[2]
        tzmin = min(t_1, t_2)
        tzmax = max(t_1, t_2)

        if (tmin > tymax) or (tymin > tmax):
            return np.inf

        if tymin > tmin:
            tmin = tymin
        if tmax < tmax:
            tmax = tymax

        if (tmin > tzmax) or (tzmin > tmax):
            return np.inf

        if tzmin > tmin:
            tmin = tzmin

        return tmin


class Plane(SceneObject):
    def __init__(
        self,
        position,
        normal,
        color=[1.0, 1.0, 1.0],
        diffuse_c=0.9,
        specular_c=0.1,
        reflection=0.1,
    ):
        self.type = "plane"
        self.position = np.array(position, dtype=float)
        self.normal = np.array(normal, dtype=float)
        self.color = np.array(color, dtype=float)
        self.reflection = reflection
        self.diffuse_c = diffuse_c
        self.specular_c = specular_c

    def get_normal(self, coords=None):
        return self.normal

    def intersect(self, rayO, rayD):
        denom = np.dot(rayD, self.normal)
        if np.abs(denom) < 1e-6:
            return np.inf
        d = np.dot(self.position - rayO, self.normal) / denom
        if d < 0:
            return np.inf
        return d


class Sphere(SceneObject):
    def __init__(
        self,
        center,
        radius,
        color=[1.0, 0.0, 0.0],
        diffuse_c=0.5,
        specular_c=0.5,
        reflection=0.5,
    ):
        self.type = "sphere"
        self.center = np.array(center, dtype=float)
        self.radius = radius
        self.color = np.array(color)
        self.reflection = reflection
        self.diffuse_c = diffuse_c
        self.specular_c = specular_c

    def get_normal(self, coords):
        return ops.normalize(coords - self.center)

    def get_color(self, coords=None):
        return self.color

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
            if t1 >= 0:
                if t0 < 0:
                    return t1
                else:
                    return t0
        return np.inf

