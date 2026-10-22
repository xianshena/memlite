"""The only model boundary: six immutable numeric arrays, strictly parsed.

visible row: marker, camera_xyz[3], observed_rpy[3], shape(0=sphere),
rgb[3], radius, support_evidence, contact_evidence (-1=unobserved).
The center-only sensor does not observe support/contact, so both are -1.
Sphere rpy is zero
because spheres have no identifiable orientation. No material/hardness/taste.
ego_pose: world xyz + roll/pitch/yaw. previous_action: kind, target marker
(0=observe, 1=probe completed, 2=select completed, 3=move completed).
feedback: valid, marker, experienced value (-1/+1); invalid always (0,0,0).
subject_context: independently sampled tag, observed need enum.
task_context: task enum, target marker A, target marker B, interaction budget.
Target markers identify the given candidate set; neither is the answer.
"""
from dataclasses import dataclass, replace
import math

NO_FEEDBACK = (0, 0, 0)
FIELDS = ('visible_observations', 'ego_pose', 'previous_action', 'observed_feedback',
          'subject_context', 'task_context')


def numeric_vector(value, length):
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise ValueError('wrong numeric vector shape')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError('only finite numeric inputs allowed')
    return tuple(value)


def tag(x):
    return type(x) is int and 0 < x < 2**53


@dataclass(frozen=True, slots=True)
class ModelInput:
    visible_observations: tuple
    ego_pose: tuple
    previous_action: tuple
    observed_feedback: tuple
    subject_context: tuple
    task_context: tuple

    def __post_init__(self):
        rows = tuple(numeric_vector(row, 14) for row in self.visible_observations)
        object.__setattr__(self, 'visible_observations', rows)
        for name, size in (('ego_pose', 6), ('previous_action', 2), ('observed_feedback', 3),
                           ('subject_context', 2), ('task_context', 4)):
            object.__setattr__(self, name, numeric_vector(getattr(self, name), size))
        markers = [r[0] for r in rows]
        if len(markers) != len(set(markers)) or any(not tag(m) for m in markers):
            raise ValueError('markers must be unique episode-local numeric tags')
        for row in rows:
            if row[7] != 0 or any(not 0 <= c <= 1 for c in row[8:11]) or not 0 < row[11] <= .5:
                raise ValueError('unsupported shape, color or size')
            if row[12] != -1 or row[13] != -1:
                raise ValueError('center-ray sensor cannot assert unobserved support/contact')
        if not tag(self.subject_context[0]) or self.subject_context[1] not in (0, 1):
            raise ValueError('invalid observable subject context')
        kind, target = self.previous_action
        if kind not in (0, 1, 2, 3) or (kind in (1, 2) and target not in markers):
            raise ValueError('illegal or invisible action target')
        valid, marker, value = self.observed_feedback
        if valid == 0:
            if self.observed_feedback != NO_FEEDBACK:
                raise ValueError('invalid feedback must use common no-feedback sentinel')
        elif valid == 1:
            if kind != 1 or marker != target or marker not in markers or value not in (-1, 1):
                raise ValueError('feedback requires an already completed visible probe')
        else:
            raise ValueError('feedback valid flag invalid')
        task, a, b, budget = self.task_context
        if task not in (1, 2, 3, 4, 5) or not tag(a) or not tag(b) or a == b:
            raise ValueError('invalid task or candidate set')
        if type(budget) is not int or budget < 0:
            raise ValueError('invalid interaction budget')

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(FIELDS):
            raise ValueError('input field whitelist violation')
        return cls(**data)

    def as_dict(self):
        return {name: [list(r) for r in self.visible_observations] if name == 'visible_observations'
                else list(getattr(self, name)) for name in FIELDS}

    @property
    def markers(self):
        return tuple(int(row[0]) for row in self.visible_observations)

    def permuted(self, order):
        if sorted(order) != list(range(len(self.visible_observations))):
            raise ValueError('not a permutation')
        return replace(self, visible_observations=tuple(self.visible_observations[i] for i in order))

    def with_subject(self, subject):
        return replace(self, subject_context=(subject, self.subject_context[1]))
