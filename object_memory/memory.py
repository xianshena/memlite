"""Nonlearned baselines. Methods accept ModelInput only, never Episode/Scene.

All retained Python containers/keys/values/counters count toward state budget.
Heap bytes (CPython getsizeof) and packed serialization bytes are separate.
Parameters and trainable parameters are zero. No hidden tracker/event cache.
"""
from collections import OrderedDict
import json
import sys

from .geometry import Pose, to_world
from .interface import ModelInput


def deep_bytes(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_bytes(k, seen) + deep_bytes(v, seen) for k, v in obj.items())
    elif isinstance(obj, (tuple, list)):
        size += sum(deep_bytes(v, seen) for v in obj)
    else:
        for name in getattr(obj, '__slots__', ()):
            if hasattr(obj, name):
                size += deep_bytes(getattr(obj, name), seen)
    return size


def decision(frame, value_lookup, position_lookup, attribute_lookup, force=False):
    if not isinstance(frame, ModelInput):
        raise TypeError('only ModelInput may cross model boundary')
    targets = set(frame.task_context[1:3])
    candidates = [m for m in frame.markers if m in targets]
    vals = {m: value_lookup(frame.subject_context, m) for m in candidates}
    known = len(candidates) == 2 and all(v is not None for v in vals.values())
    best = sorted(m for m, v in vals.items() if v == max(vals.values())) if known else sorted(candidates)
    probabilities = [((1/len(best)) if m in best else 0.0) for m in frame.markers]
    # Deterministic tie-break independent of list order, independent of rewards.
    selected = best[0] if best else 0
    index = frame.markers.index(selected) if selected else -1
    action_kind = 2 if known or force else (1 if frame.task_context[3] > 0 else 0)
    return {'candidate_probabilities': probabilities, 'selected_index': index if action_kind == 2 else -1,
            'selected_marker': selected if action_kind == 2 else 0,
            'expected_return': [vals.get(m) for m in frame.markers],
            'attribute_belief': [attribute_lookup(m) for m in frame.markers],
            'position_distribution': [{'last_known_point': position_lookup(m), 'calibrated': False}
                                      for m in frame.markers],
            'insufficient_evidence': int(not known), 'action': [action_kind, selected]}


class ExplicitStore:
    __slots__ = ('budget_bytes', 'records', 'tick', 'evictions')
    parameter_count = 0
    trainable_parameter_count = 0

    def __init__(self, budget_kib=192):
        if budget_kib not in (32, 64, 192):
            raise ValueError('registered budgets: 32, 64, 192 KiB')
        self.budget_bytes = budget_kib * 1024
        self.records = OrderedDict()
        self.tick = 0
        self.evictions = 0

    def _put(self, key, payload):
        self.records.pop(key, None)
        self.records[key] = (self.tick, payload)

    def _get(self, key):
        rec = self.records.get(key)
        return rec[1] if rec is not None else None

    def observe(self, frame):
        if not isinstance(frame, ModelInput):
            raise TypeError('only ModelInput may cross model boundary')
        self.tick += 1
        pose = Pose.from_vector(frame.ego_pose)
        for row in frame.visible_observations:
            self._put((0, int(row[0])), (to_world(pose, row[1:4]), row[7:]))
        if frame.observed_feedback[0]:
            _, marker, value = frame.observed_feedback
            self._put((1, *frame.subject_context, marker), value)
        while self.state_bytes() > self.budget_bytes and self.records:
            self.records.popitem(last=False)
            self.evictions += 1

    def value(self, subject, marker):
        return self._get((1, *subject, marker))

    def position(self, marker):
        record = self._get((0, marker))
        return record[0] if record else None

    def attribute(self, marker):
        record = self._get((0, marker))
        return record[1] if record else None

    def decide(self, frame, force=False):
        return decision(frame, self.value, self.position, self.attribute, force)

    def invalidate(self, subject, marker):
        self.records.pop((1, *subject, marker), None)

    def clear(self):
        self.records.clear()
        self.tick = self.evictions = 0

    @property
    def record_count(self):
        return len(self.records)

    def state_bytes(self):
        return deep_bytes(self)

    def serialized_bytes(self):
        return len(json.dumps([(list(k), v) for k, v in self.records.items()], separators=(',', ':')).encode())


class NoMemory:
    __slots__ = ()
    parameter_count = 0
    trainable_parameter_count = 0

    def observe(self, frame):
        if not isinstance(frame, ModelInput):
            raise TypeError('only ModelInput may cross model boundary')

    def decide(self, frame, force=False):
        return decision(frame, lambda s, m: None, lambda m: None, lambda m: None, force)

    def clear(self):
        pass

    def state_bytes(self):
        return deep_bytes(self)


class LastObservation:
    __slots__ = ('last',)
    parameter_count = 0
    trainable_parameter_count = 0

    def __init__(self):
        self.last = None

    def observe(self, frame):
        if not isinstance(frame, ModelInput):
            raise TypeError('only ModelInput may cross model boundary')
        self.last = frame

    def value(self, subject, marker):
        if self.last and self.last.subject_context == subject:
            valid, m, v = self.last.observed_feedback
            if valid and m == marker:
                return v
        return None

    def position(self, marker):
        if self.last:
            for row in self.last.visible_observations:
                if row[0] == marker:
                    return to_world(Pose.from_vector(self.last.ego_pose), row[1:4])
        return None

    def attribute(self, marker):
        if self.last:
            return next((row[7:] for row in self.last.visible_observations if row[0] == marker), None)
        return None

    def decide(self, frame, force=False):
        return decision(frame, self.value, self.position, self.attribute, force)

    def clear(self):
        self.last = None

    def state_bytes(self):
        return deep_bytes(self)
