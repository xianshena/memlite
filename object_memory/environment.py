"""Deterministic 1m scene and experienced-value episodes; truth stays here.

Each RNG stream is SHA256-domain-separated. No seed, instance ID, label or
reward table crosses ModelInput. Tracking tags are resampled identity aids.
"""
from dataclasses import dataclass, replace, asdict
import hashlib
import math
import random

from .geometry import Pose, Sphere, Box, visible, to_local
from .interface import ModelInput, NO_FEEDBACK


def rng(seed, domain):
    digest = hashlib.sha256(f'object-memory-v1/{domain}/{seed}'.encode()).digest()
    return random.Random(int.from_bytes(digest, 'big'))


@dataclass(frozen=True, slots=True)
class Scene:
    objects: tuple
    boxes: tuple
    supports: tuple


@dataclass(frozen=True, slots=True)
class Episode:
    seed: int
    scene: Scene
    frames: tuple
    truth: tuple  # evaluator-only ((marker, experienced value), ...)
    subject: int
    scenes: tuple

    def record(self):
        return {'seed': self.seed, 'evaluator_only': {'scenes': [asdict(s) for s in self.scenes], 'truth': self.truth},
                'model_inputs': [f.as_dict() for f in self.frames]}


def sense(scene, pose, order_rng, subject, targets, action=(0, 0), feedback=NO_FEEDBACK, budget=0):
    rows = []
    for obj in scene.objects:
        if visible(pose, obj, scene.objects, scene.boxes + scene.supports):
            rows.append((obj.marker,) + to_local(pose, obj.center) + (0, 0, 0, 0)
                        + obj.rgb + (obj.radius, -1, -1))
    order_rng.shuffle(rows)
    return ModelInput(tuple(rows), pose.vector(), action, feedback, (subject, 0), (2, *targets, budget))


def make_episode(seed, distractors=4, reward_seed=None):
    if distractors < 1:
        raise ValueError('at least one interference event required')
    app, track = rng(seed, 'appearance'), rng(seed, 'tracking_marker')
    subj, layout = rng(seed, 'subject_tag'), rng(seed, 'layout')
    order = rng(seed, 'candidate_order')
    reward = rng(seed if reward_seed is None else reward_seed, 'hidden_reward')
    interference = rng(seed, 'interference')
    count = layout.randint(4, 16)
    markers = track.sample(range(1, 2**52), count)
    subject = subj.randrange(1, 2**52)
    rgb = tuple(app.random() for _ in range(3))
    # Two matched-looking targets; hidden taste independent of all geometry.
    radius = .025
    objects = [Sphere(markers[0], (.25, .75, .35), radius, rgb),
               Sphere(markers[1], (.75, .75, .35), radius, rgb)]
    for i in range(2, count):
        z = (.15, .35, .65)[i % 3]
        objects.append(Sphere(markers[i], (layout.uniform(.46, .54), layout.uniform(.74, .95), z),
                              .008, tuple(app.random() for _ in range(3))))
    supports = tuple(Box((o.center[0]-.03, o.center[1]-.03, 0),
                         (o.center[0]+.03, o.center[1]+.03, o.center[2]-o.radius)) for o in objects)
    scene = Scene(tuple(objects), (Box((.43, .4, .02), (.57, .5, .98)),), supports)
    pose = Pose((layout.uniform(.46, .54), .02, layout.uniform(.32, .4)), (-math.pi/2, 0, 0))
    final_pose = Pose((.5, .04, .36), (-math.pi/2, .02, .015))
    targets = tuple(markers[:2])
    values = (1, -1) if reward.getrandbits(1) else (-1, 1)
    frames = [sense(scene, pose, order, subject, targets, budget=2)]
    scenes = [scene]
    for marker, value in zip(targets, values):
        frames.append(sense(scene, pose, order, subject, targets, (1, marker), (1, marker, value), 1))
        scenes.append(scene)
    # New, visibly probed distractors enter the foreground. Their independent
    # feedback creates genuine competing records, rather than repeated empty steps.
    for i in range(distractors):
        angle = .01 * math.sin(i)
        p = Pose((.5 + .02*math.cos(i), .03, .36), (-math.pi/2, angle, 0))
        marker = track.randrange(1, 2**52)
        while marker in markers:
            marker = track.randrange(1, 2**52)
        markers.append(marker)
        distractor = Sphere(marker, (.5, .23, .32), .008, tuple(app.random() for _ in range(3)))
        ds = replace(scene, objects=scene.objects[:-1] + (distractor,))
        v = interference.choice((-1, 1))
        frames.append(sense(ds, p, order, subject, targets, (1, marker), (1, marker, v)))
        scenes.append(ds)
    frames.append(sense(scene, final_pose, order, subject, targets))
    scenes.append(scene)
    if not all(set(targets).issubset(f.markers) for f in frames):
        raise AssertionError('generator produced an unobservable probe/choice')
    return Episode(seed, scene, tuple(frames), tuple(zip(targets, values)), subject, tuple(scenes))


def swapped_episode(episode):
    frames = tuple(replace(f, observed_feedback=(1, f.observed_feedback[1], -f.observed_feedback[2]))
                   if f.observed_feedback[0] and f.observed_feedback[1] in dict(episode.truth) else f
                   for f in episode.frames)
    return replace(episode, frames=frames, truth=tuple((m, -v) for m, v in episode.truth))
