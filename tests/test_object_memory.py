"""G0/G1a executable contract, written before the implementation."""
import copy
import math
import unittest
import tempfile
from pathlib import Path

from object_memory.geometry import Pose, to_world, to_local, visible, Sphere, Box
from object_memory.interface import ModelInput, NO_FEEDBACK
from object_memory.environment import make_episode, swapped_episode
from object_memory.memory import ExplicitStore, LastObservation, NoMemory
from object_memory.evaluate import run_history, score_choice, chance_control, spatial_control, canonical, sha


class GeometryTests(unittest.TestCase):
    def test_full_rotation_roundtrip(self):
        p = Pose((.2, .3, .4), (.4, -.7, 1.2))
        q = (.13, -.2, .8)
        for a, b in zip(q, to_local(p, to_world(p, q))):
            self.assertAlmostEqual(a, b, places=12)

    def test_known_rotation_not_only_roundtrip(self):
        p = Pose((1, 2, 3), (0, 0, math.pi / 2))
        for a, b in zip(to_world(p, (1, 0, 0)), (1, 3, 3)):
            self.assertAlmostEqual(a, b)
        for rpy, expected in (((math.pi/2, 0, 0), (0, -1, 0)),
                              ((0, math.pi/2, 0), (1, 0, 0))):
            for a, b in zip(to_world(Pose((0, 0, 0), rpy), (0, 0, 1)), expected):
                self.assertAlmostEqual(a, b)

    def test_box_sphere_occlusion_and_behind(self):
        p = Pose((0, 0, 0), (0, 0, 0))
        target = Sphere(1, (0, 0, .8), .03, (1, 0, 0))
        self.assertTrue(visible(p, target, [target], []))
        self.assertFalse(visible(p, target, [target], [Box((-.1, -.1, .3), (.1, .1, .4))]))
        blocker = Sphere(2, (0, 0, .4), .08, (0, 0, 0))
        self.assertFalse(visible(p, target, [target, blocker], []))
        behind = Sphere(3, (0, 0, .95), .05, (0, 0, 0))
        self.assertTrue(visible(p, target, [target, behind], []))

    def test_camera_motion_disoccludes_and_fov(self):
        target = Sphere(1, (0, 0, .8), .03, (1, 0, 0))
        box = Box((-.1, -.1, .3), (.1, .1, .4))
        self.assertTrue(visible(Pose((.4, 0, 0), (0, 0, 0)), target, [target], [box]))
        self.assertFalse(visible(Pose((0, 0, 1), (0, 0, 0)), target, [target], []))


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.ep = make_episode(0)

    def test_strict_numeric_whitelist(self):
        data = self.ep.frames[-1].as_dict()
        self.assertEqual(ModelInput.from_dict(data), self.ep.frames[-1])
        for key in ('hidden_taste', 'correct_candidate', 'occluded_ground_truth', 'extra'):
            bad = copy.deepcopy(data)
            bad[key] = 0
            with self.assertRaises(ValueError):
                ModelInput.from_dict(bad)
        bad = copy.deepcopy(data)
        bad['visible_observations'][0][0] = 'apple'
        with self.assertRaises(ValueError):
            ModelInput.from_dict(bad)

    def test_feedback_needs_completed_visible_action(self):
        bad = self.ep.frames[-1].as_dict()
        bad['observed_feedback'] = [1, 999, 1]
        with self.assertRaises(ValueError):
            ModelInput.from_dict(bad)
        bad = self.ep.frames[1].as_dict()
        bad['previous_action'] = [0, 0]
        with self.assertRaises(ValueError):
            ModelInput.from_dict(bad)

    def test_hidden_truth_changes_no_preexperience_inputs(self):
        other = swapped_episode(self.ep)
        self.assertEqual(self.ep.frames[0], other.frames[0])
        self.assertEqual(self.ep.frames[-1], other.frames[-1])
        self.assertEqual(self.ep.frames[-1].observed_feedback, NO_FEEDBACK)
        self.assertNotEqual(self.ep.truth, other.truth)
        for a, b in zip(self.ep.frames, other.frames):
            aa, bb = a.as_dict(), b.as_dict()
            aa.pop('observed_feedback'); bb.pop('observed_feedback')
            self.assertEqual(aa, bb)

    def test_episode_reproducibility_and_independent_reward_stream(self):
        self.assertEqual(make_episode(33), make_episode(33))
        changed = make_episode(33, reward_seed=12345)
        for a, b in zip(make_episode(33).frames, changed.frames):
            aa, bb = a.as_dict(), b.as_dict()
            aa.pop('observed_feedback'); bb.pop('observed_feedback')
            self.assertEqual(aa, bb)
        self.assertNotEqual(make_episode(33).frames[0], make_episode(34).frames[0])

    def test_scene_bounds_count_partial_visibility_and_observer_motion(self):
        for seed in range(100):
            ep = make_episode(seed)
            self.assertTrue(4 <= len(ep.scene.objects) <= 16)
            self.assertLess(len(ep.frames[0].visible_observations), len(ep.scene.objects))
            self.assertNotEqual(ep.frames[0].ego_pose, ep.frames[-1].ego_pose)
            for obj in ep.scene.objects:
                self.assertTrue(all(0 <= x <= 1 for x in obj.center))

    def test_every_frame_matches_only_visible_geometric_objects(self):
        for seed in range(10):
            ep = make_episode(seed)
            for frame, scene in zip(ep.frames, ep.scenes):
                pose = Pose.from_vector(frame.ego_pose)
                expected = {o.marker for o in scene.objects
                            if visible(pose, o, scene.objects, scene.boxes + scene.supports)}
                self.assertEqual(set(frame.markers), expected)
                for row in frame.visible_observations:
                    obj = next(o for o in scene.objects if o.marker == row[0])
                    self.assertLess(math.dist(to_world(pose, row[1:4]), obj.center), 1e-12)

    def test_bad_shapes_nan_and_sentinels_rejected(self):
        for name, value in (('ego_pose', [0]), ('ego_pose', [float('nan')]*6),
                            ('observed_feedback', [0, 123, -1]), ('subject_context', [1, 'happy'])):
            data = self.ep.frames[-1].as_dict()
            data[name] = value
            with self.assertRaises(ValueError):
                ModelInput.from_dict(data)

    def test_diagnostic_markers_independent_and_not_indices(self):
        a, b = make_episode(0), make_episode(1)
        self.assertTrue(set(a.frames[0].markers).isdisjoint(b.frames[0].markers))
        orders = [tuple(m for m in f.markers if m in dict(a.truth)) for f in a.frames]
        self.assertGreater(len(set(orders)), 1)


class MemoryTests(unittest.TestCase):
    def test_feedback_swap_clear_and_permutation(self):
        for seed in range(20):
            ep = make_episode(seed)
            model = run_history(ExplicitStore(), ep)
            final = ep.frames[-1]
            decision = model.decide(final)
            self.assertEqual(score_choice(ep, decision), 1)
            flipped = run_history(ExplicitStore(), swapped_episode(ep)).decide(final)
            self.assertNotEqual(decision['selected_marker'], flipped['selected_marker'])
            shuffled = final.permuted(tuple(reversed(range(len(final.visible_observations)))))
            permuted = model.decide(shuffled)
            self.assertEqual(decision['selected_marker'], permuted['selected_marker'])
            self.assertEqual(dict(zip(final.markers, decision['candidate_probabilities'])),
                             dict(zip(shuffled.markers, permuted['candidate_probabilities'])))
            model.clear()
            self.assertEqual(model.decide(final)['insufficient_evidence'], 1)

    def test_last_observation_and_no_memory_have_no_experience(self):
        ep = make_episode(9)
        for cls in (LastObservation, NoMemory):
            model = run_history(cls(), ep)
            self.assertEqual(model.decide(ep.frames[-1])['insufficient_evidence'], 1)
            self.assertEqual(model.decide(ep.frames[-1])['action'][0], 0)
            self.assertEqual(model.decide(ep.frames[0])['action'][0], 1)
            forced = model.decide(ep.frames[-1], force=True)
            self.assertEqual(forced['action'][0], 2)

    def test_subject_isolation_update_and_invalidation(self):
        ep = make_episode(1)
        model = run_history(ExplicitStore(), ep)
        final = ep.frames[-1]
        before = model.decide(final)['selected_marker']
        other = final.with_subject(final.subject_context[0] + 1)
        self.assertEqual(model.decide(other)['insufficient_evidence'], 1)
        for frame in swapped_episode(ep).frames[1:3]:
            model.observe(frame.with_subject(other.subject_context[0]))
        self.assertNotEqual(before, model.decide(other)['selected_marker'])
        self.assertEqual(before, model.decide(final)['selected_marker'])
        for frame in swapped_episode(ep).frames[1:3]:
            model.observe(frame)
        self.assertNotEqual(before, model.decide(final)['selected_marker'])
        loser = before
        model.invalidate(final.subject_context, loser)
        self.assertIsNone(model.value(final.subject_context, loser))
        winner = next(m for m, v in swapped_episode(ep).truth if v == 1)
        model.invalidate(final.subject_context, winner)
        self.assertIsNone(model.value(final.subject_context, winner))
        self.assertEqual(model.decide(final)['insufficient_evidence'], 1)

    def test_geometric_scene_motion_occlusion_not_synthetic_row_only(self):
        for seed in range(20):
            self.assertTrue(spatial_control(seed)['passed'])

    def test_episode_boundary_clear_and_invalidation_preserve_other_records(self):
        ep = make_episode(5)
        model = run_history(ExplicitStore(), ep)
        subject = ep.frames[0].subject_context
        target = ep.truth[0][0]
        others = {k: v for k, v in model.records.items() if k != (1, *subject, target)}
        model.invalidate(subject, target)
        self.assertEqual(dict(model.records), others)
        model.clear()
        self.assertTrue(model.decide(ep.frames[-1])['insufficient_evidence'])

    def test_model_rejects_evaluator_truth_container(self):
        for model in (ExplicitStore(), NoMemory(), LastObservation()):
            with self.assertRaises(TypeError):
                model.observe(make_episode(0))

    def test_state_stays_bounded_at_all_registered_capacities(self):
        for budget in (32, 64, 192):
            model = ExplicitStore(budget)
            run_history(model, make_episode(2, distractors=256))
            self.assertLessEqual(model.state_bytes(), budget*1024)
            self.assertGreater(model.serialized_bytes(), 0)

    def test_observed_motion_then_occlusion_retains_last_known_position(self):
        model = ExplicitStore()
        ep = make_episode(3)
        model.observe(ep.frames[0])
        row = ep.frames[0].visible_observations[0]
        marker = int(row[0])
        expected = to_world(Pose.from_vector(ep.frames[0].ego_pose), row[1:4])
        self.assertEqual(model.position(marker), expected)
        moved = ep.frames[0].as_dict()
        moved['visible_observations'][0][1] += .08
        moved = ModelInput.from_dict(moved)
        model.observe(moved)
        expected = to_world(Pose.from_vector(moved.ego_pose), moved.visible_observations[0][1:4])
        hidden = moved.as_dict()
        hidden['visible_observations'] = []
        model.observe(ModelInput.from_dict(hidden))
        self.assertEqual(model.position(marker), expected)
        self.assertIsNone(model.position(999999))

    def test_state_budget_counts_every_retained_container_and_evicts(self):
        model = ExplicitStore(32)
        for seed in range(20):
            run_history(model, make_episode(seed))
            self.assertLessEqual(model.state_bytes(), 32 * 1024)
        self.assertGreater(model.evictions, 0)
        model.clear()
        self.assertEqual(model.record_count, 0)
        self.assertEqual(model.parameter_count, 0)

    def test_no_experience_2000_balanced_independent_trials(self):
        report = chance_control(2000)
        self.assertEqual(report['n'], 2000)
        self.assertEqual(sum(report['label_counts']), 2000)
        self.assertTrue(.45 <= report['accuracy'] <= .55)
        self.assertTrue(report['ci95'][0] <= .5 <= report['ci95'][1])


if __name__ == '__main__':
    unittest.main()
