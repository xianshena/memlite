"""Executable environment contracts, written before life_choice implementation."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from life_choice.environment import (ACTIVITIES, FEATURE_NAMES, MEMORY_COLUMNS,
    generate_episode, execute, freeze_dataset, schema_audit, features_without_memory,
    with_history, reorder_episode)


class DailyEnvironmentTests(unittest.TestCase):
    def test_reproducibility_and_strict_feature_whitelist(self):
        a, b = generate_episode(17), generate_episode(17)
        np.testing.assert_array_equal(a.features, b.features)
        self.assertEqual(a.as_record(), b.as_record())
        self.assertEqual(a.features.shape, (12, len(FEATURE_NAMES)))
        self.assertEqual(a.features.dtype, np.float32)
        self.assertEqual(a.mask.dtype, np.bool_)
        self.assertEqual(schema_audit(a.as_record()), [])
        bad = copy.deepcopy(a.as_record())
        bad['inputs']['hidden_preference'] = 1
        self.assertTrue(schema_audit(bad))
        bad = copy.deepcopy(a.as_record())
        bad['inputs']['features'][0][0] = '0.5'
        self.assertTrue(schema_audit(bad))
        self.assertFalse(any(any(token in n for token in ('seed', 'target', 'utility', 'hidden', 'subject'))
                             for n in FEATURE_NAMES))

    def test_navigation_paths_and_completion_in_100_square_metres(self):
        completed = 0
        for seed in range(12):
            ep = generate_episode(seed)
            for i, c in enumerate(ep.candidates):
                result = execute(ep, i)
                self.assertEqual(result['completed'], bool(ep.mask[i]))
                if result['completed']:
                    completed += 1
                    path = result['path']
                    self.assertTrue(result['arrived'])
                    self.assertEqual(tuple(path[-1][:2]), tuple(c.destination[:2]))
                    self.assertTrue(0 <= c.destination[2] <= 1.8)
                    for p in path:
                        self.assertTrue(0 <= p[0] < 10 and 0 <= p[1] < 10)
                        self.assertNotIn([int(p[0]), int(p[1])], ep.meta['blocked_cells'])
                    for p, q in zip(path, path[1:]):
                        self.assertAlmostEqual(abs(p[0]-q[0])+abs(p[1]-q[1]), 1)
                    np.testing.assert_allclose(np.asarray(result['end_body'])-ep.body,
                                               ep.outcome_targets[i], atol=1e-6)
        self.assertGreater(completed, 80)

    def test_unavailable_action_cannot_complete_or_emit_feedback(self):
        found = False
        for seed in range(25):
            ep = generate_episode(seed)
            for i in np.flatnonzero(~ep.mask):
                found = True
                result = execute(ep, int(i))
                self.assertFalse(result['completed'])
                self.assertIsNone(result['feedback'])
                np.testing.assert_array_equal(result['end_body'], ep.body)
        self.assertTrue(found)

    def test_history_counterfactual_changes_only_memory_and_flips_choice(self):
        keep = [i for i in range(len(FEATURE_NAMES)) if i not in MEMORY_COLUMNS]
        for seed in range(20):
            a = generate_episode(seed, variant='counterfactual')
            b = generate_episode(seed, variant='counterfactual', history_flip=True)
            np.testing.assert_array_equal(a.features[:, keep], b.features[:, keep])
            np.testing.assert_array_equal(a.mask, b.mask)
            self.assertEqual(a.candidates, b.candidates)
            self.assertEqual(a.meta['current_feedback'], b.meta['current_feedback'])
            self.assertEqual(a.meta['current_feedback'], [0, 0.0])
            self.assertNotEqual(a.target, b.target)
            self.assertNotEqual(a.history, b.history)

    def test_unknown_has_uniform_teacher_and_no_preference_leak(self):
        for seed in range(20):
            ep = generate_episode(seed, variant='unknown')
            self.assertEqual(int(ep.mask.sum()), 2)
            self.assertEqual(len(ep.history), 0)
            legal = ep.teacher_utilities[ep.mask]
            self.assertEqual(float(legal[0]),float(legal[1]))
            np.testing.assert_array_equal(ep.features[:, MEMORY_COLUMNS], np.zeros((12, 2)))
            self.assertEqual(ep.meta['teacher_probabilities'].count(.5), 2)

    def test_unobserved_truth_changes_no_current_features_or_expected_teacher(self):
        ep = generate_episode(9)
        changed_meta = copy.deepcopy(ep.meta)
        changed_meta['hidden_preferences'] = {a:-v for a,v in ep.meta['hidden_preferences'].items()}
        changed = replace(ep,meta=changed_meta)
        np.testing.assert_array_equal(ep.features,changed.features)
        np.testing.assert_array_equal(ep.teacher_utilities,changed.teacher_utilities)
        np.testing.assert_array_equal(ep.outcome_targets,changed.outcome_targets)
        a,b = execute(ep,ep.target),execute(changed,ep.target)
        self.assertEqual(a['completed'],b['completed'])
        self.assertNotEqual(a['feedback'],b['feedback'])
        self.assertEqual(ep.as_record()['inputs'],changed.as_record()['inputs'])

    def test_candidate_order_randomized_and_split_entities_isolated(self):
        orders = {tuple(c.activity for c in generate_episode(s).candidates) for s in range(12)}
        self.assertGreater(len(orders), 8)
        a, b, c = [generate_episode(5, split=s) for s in ('train', 'validation', 'test')]
        for key in ('subject_group', 'scene_group', 'object_group'):
            self.assertEqual(len({e.meta[key] for e in (a,b,c)}), 3)
        for x,y in ((a,b),(b,c),(a,c)):
            self.assertNotEqual(x.meta['blocked_cells'],y.meta['blocked_cells'])
        reordered = reorder_episode(a,np.arange(11,-1,-1))
        self.assertEqual(reordered.candidates[reordered.target].activity,a.candidates[a.target].activity)
        for i,candidate in enumerate(a.candidates):
            j = [c.activity for c in reordered.candidates].index(candidate.activity)
            self.assertEqual(a.teacher_utilities[i],reordered.teacher_utilities[j])
            self.assertEqual(execute(a,i),execute(reordered,j))

    def test_explicit_history_is_observed_and_subject_scoped(self):
        ep = generate_episode(4)
        self.assertTrue(ep.history)
        for event in ep.history:
            self.assertEqual(event['subject_group'], ep.meta['subject_group'])
            self.assertTrue(event['arrived'] and event['completed'])
            self.assertTrue(event['observed'])
        self.assertEqual(ep.meta['current_feedback'], [0, 0.0])
        foreign = generate_episode(4, split='test').history
        isolated = with_history(ep, foreign)
        np.testing.assert_array_equal(isolated.features[:, MEMORY_COLUMNS], np.zeros((12,2)))
        cleared = features_without_memory(ep)
        keep = [i for i in range(len(FEATURE_NAMES)) if i not in MEMORY_COLUMNS]
        np.testing.assert_array_equal(cleared[:,keep],ep.features[:,keep])
        np.testing.assert_array_equal(cleared[:, MEMORY_COLUMNS],np.zeros((12,2)))

    def test_small_freeze_has_hashes_and_disjoint_split_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = freeze_dataset(tmp, train_n=20, val_n=20, test_n=20)
            self.assertEqual(manifest['split_counts'], {'train':20, 'validation':20, 'test':20})
            self.assertEqual(manifest['independent_contexts_total'],60)
            self.assertEqual(manifest['feature_names'], list(FEATURE_NAMES))
            self.assertTrue(manifest['source_sha256'])
            self.assertTrue((Path(tmp)/'manifest.json').exists())
            entities = []
            for split in ('train', 'validation', 'test'):
                with np.load(Path(tmp)/f'{split}.npz') as arr:
                    self.assertEqual(arr['features'].shape[1:], (12, len(FEATURE_NAMES)))
                    self.assertEqual(arr['outcomes'].shape[-1], 6)
                    self.assertEqual(arr['variants'].dtype,np.uint8)
                    self.assertEqual(np.bincount(arr['variants']).tolist(),[16,3,1])
                    self.assertEqual(manifest['splits'][split]['variant_counts'],
                                     {'daily':16,'counterfactual':3,'unknown':1})
                entities.append(set(manifest['splits'][split]['subject_groups']))
            self.assertFalse(entities[0] & entities[1] or entities[1] & entities[2] or entities[0] & entities[2])
            with self.assertRaises(FileExistsError):
                freeze_dataset(tmp,train_n=1,val_n=1,test_n=1)

    def test_cooking_prepares_meal_and_values_known_next_eating_step(self):
        found = False
        for seed in range(100):
            ep = generate_episode(seed)
            cook = next(i for i,c in enumerate(ep.candidates) if c.activity=='cook')
            if ep.mask[cook] and not ep.meta['resources']['meal_prepared']:
                found = True
                result = execute(ep,cook)
                self.assertTrue(result['resources_after']['meal_prepared'])
                self.assertFalse(result['resources_after']['ingredients_available'])
                self.assertEqual(result['end_body'][1],float(ep.body[1]))
                self.assertGreaterEqual(ep.meta['cook_two_step_need_value'],0)
                self.assertEqual(ep.features[cook,FEATURE_NAMES.index('meal_prepared')],0)
        self.assertTrue(found)

    def test_scene_changes_have_real_geometry_and_destination_variation(self):
        eps = [generate_episode(seed) for seed in range(30)]
        self.assertGreater(len({tuple(e.meta['door_cells']) for e in eps}),1)
        self.assertGreater(len({tuple(tuple(p) for p in e.meta['blocked_cells']) for e in eps}),15)
        for activity in ACTIVITIES:
            places = {next(c.destination for c in e.candidates if c.activity==activity) for e in eps}
            self.assertGreater(len(places),8)
        train = {e.meta['scene_geometry_sha256'] for e in eps}
        test = {generate_episode(seed,split='test').meta['scene_geometry_sha256'] for seed in range(30)}
        self.assertFalse(train & test)

    def test_external_goal_scenarios_cover_work_read_exercise_without_forcing_labels(self):
        counts = {activity:0 for activity in ('work','read','exercise')}
        scenes = set()
        for seed in range(1200):
            ep = generate_episode(seed)
            scenes.add(ep.meta['scenario'])
            chosen = ep.candidates[ep.target].activity
            if chosen in counts:
                counts[chosen] += 1
        self.assertEqual(len(scenes),12)
        self.assertTrue(all(count > 15 for count in counts.values()),counts)

    def test_expected_teacher_and_execution_reward_differ_only_in_new_preference(self):
        for seed in range(20):
            ep = generate_episode(seed)
            for i in np.flatnonzero(ep.mask):
                result = execute(ep,int(i))
                known_mean = float(ep.features[i,MEMORY_COLUMNS[0]])
                difference = .65*(result['feedback']-known_mean)
                self.assertAlmostEqual(result['realized_reward']-float(ep.teacher_utilities[i]),difference,places=5)
                self.assertAlmostEqual(result['realized_reward'],
                    result['myopic_realized_reward']+result['planning_value'],places=6)


if __name__ == '__main__':
    unittest.main()
