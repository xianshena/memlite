"""Right-handed world; camera +Z forward, R = Rz(yaw) Ry(pitch) Rx(roll).

Visibility uses an explicitly idealized center ray and a 100 degree square FOV.
A partly exposed sphere whose center is blocked is not detected. This is an
auditable geometry sensor, not an image renderer or learned object tracker.
"""
from dataclasses import dataclass
import math


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True, slots=True)
class Pose:
    xyz: tuple
    rpy: tuple

    @classmethod
    def from_vector(cls, vector):
        return cls(tuple(vector[:3]), tuple(vector[3:]))

    def vector(self):
        return self.xyz + self.rpy


def rotation(pose):
    r, p, y = pose.rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
            (-sp, cp*sr, cp*cr))


def to_world(pose, local):
    return add(pose.xyz, tuple(dot(row, local) for row in rotation(pose)))


def to_local(pose, world):
    delta = sub(world, pose.xyz)
    mat = rotation(pose)
    return tuple(sum(mat[j][i] * delta[j] for j in range(3)) for i in range(3))


@dataclass(frozen=True, slots=True)
class Sphere:
    marker: int
    center: tuple
    radius: float
    rgb: tuple


@dataclass(frozen=True, slots=True)
class Box:
    lower: tuple
    upper: tuple


def box_hit(origin, endpoint, box):
    """Segment/slab intersection; box must lie strictly before endpoint."""
    near, far = 0.0, 1.0 - 1e-9
    for a, b, lo, hi in zip(origin, endpoint, box.lower, box.upper):
        d = b - a
        if abs(d) < 1e-12:
            if a < lo or a > hi:
                return False
        else:
            u, v = sorted(((lo-a)/d, (hi-a)/d))
            near, far = max(near, u), min(far, v)
            if near > far:
                return False
    return far >= 0 and near <= far


def sphere_hit(origin, endpoint, sphere):
    direction, delta = sub(endpoint, origin), sub(origin, sphere.center)
    a, b = dot(direction, direction), 2 * dot(delta, direction)
    c = dot(delta, delta) - sphere.radius ** 2
    disc = b*b - 4*a*c
    if disc < 0 or a == 0:
        return False
    u, v = (-b-math.sqrt(disc))/(2*a), (-b+math.sqrt(disc))/(2*a)
    return v > 1e-9 and u < 1-1e-9


def visible(pose, target, objects, boxes):
    x, y, z = to_local(pose, target.center)
    extent = math.tan(math.radians(50))
    if z <= 0 or abs(x/z) > extent or abs(y/z) > extent:
        return False
    if any(box_hit(pose.xyz, target.center, box) for box in boxes):
        return False
    return not any(obj.marker != target.marker and sphere_hit(pose.xyz, target.center, obj)
                   for obj in objects)
