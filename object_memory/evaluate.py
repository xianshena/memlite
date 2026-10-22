"""Registered CPU-only G0/G1a runner. No optimizer/model-training dependency."""
import argparse
import dataclasses
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import tracemalloc

from .environment import make_episode, swapped_episode, rng, sense, Scene
from .geometry import Pose, Sphere, Box, to_world
from .memory import ExplicitStore, NoMemory, LastObservation
from .interface import NO_FEEDBACK

ROOT = Path(__file__).resolve().parents[1]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(obj, indent=2, ensure_ascii=False).encode('utf-8') + b'\n')


def wilson(successes, n):
    z = 1.959963984540054
    denom = 1 + z*z/n
    center = (successes/n + z*z/(2*n))/denom
    half = z*math.sqrt(successes/n*(1-successes/n)/n + z*z/(4*n*n))/denom
    return [center-half, center+half]


def proportion(successes, n):
    return {'successes': successes, 'n': n, 'accuracy': successes/n, 'ci95': wilson(successes, n)}


def run_history(model, episode):
    for frame in episode.frames[:-1]:
        model.observe(frame)
    return model


def score_choice(episode, decision):
    return int(dict(episode.truth).get(decision['selected_marker']) == 1)


def chance_control(n=2000):
    if n < 2000:
        raise ValueError('registered minimum: 2000 independent episodes')
    successes, labels, input_hashes = 0, [0, 0], set()
    for i in range(n):
        ep = make_episode(1_000_000 + i, distractors=1)
        frame = ep.frames[0]  # no previous interaction, no feedback
        pred = NoMemory().decide(frame, force=True)
        successes += score_choice(ep, pred)
        labels[int(ep.truth[0][1] == 1)] += 1
        input_hashes.add(sha(canonical(frame.as_dict())))
    return {**proportion(successes, n), 'label_counts': labels,
            'unique_independent_inputs': len(input_hashes), 'seed_range': [1_000_000, 1_000_000+n-1],
            'sampling': 'independent fair hidden bit per episode; one +1 and one -1 target',
            'forced_choice': True, 'chance_accuracy_gate': .45 <= successes/n <= .55,
            'probability_uncertainty_gate': 'NOT_ESTABLISHED_FOR_LEARNED_MODELS'}


def spatial_control(seed):
    """Real scene movement, observed update, new occluder, and disocclusion."""
    ep = make_episode(seed)
    marker = ep.truth[0][0]
    original = ep.scene.objects[0]
    moved = dataclasses.replace(original, center=(.25, .7, .55))
    scene = dataclasses.replace(ep.scene, objects=(moved,) + ep.scene.objects[1:])
    # Remove the old support under the moved object; it has been lifted.
    scene = dataclasses.replace(scene, supports=scene.supports[1:])
    pose = Pose.from_vector(ep.frames[0].ego_pose)
    subject, targets = ep.subject, tuple(m for m, v in ep.truth)
    observed = sense(scene, pose, rng(seed, 'spatial_order'), subject, targets)
    middle = tuple((a+b)/2 for a, b in zip(pose.xyz, moved.center))
    occluder = Box(tuple(x-.055 for x in middle), tuple(x+.055 for x in middle))
    hidden_scene = dataclasses.replace(scene, boxes=scene.boxes + (occluder,))
    hidden = sense(hidden_scene, pose, rng(seed, 'spatial_order2'), subject, targets)
    moved_pose = Pose((.05, .02, .4), pose.rpy)
    revealed = sense(hidden_scene, moved_pose, rng(seed, 'spatial_order3'), subject, targets)
    store = ExplicitStore()
    store.observe(ep.frames[0])
    old = store.position(marker)
    store.observe(observed)
    updated = store.position(marker)
    store.observe(hidden)
    retained = store.position(marker)
    error = math.dist(retained, moved.center)
    return {'passed': marker in observed.markers and marker not in hidden.markers
            and marker in revealed.markers and old != updated and updated == retained and error < 1e-10,
            'last_known_error_m': error,
            'inputs': [f.as_dict() for f in (ep.frames[0], observed, hidden, revealed)],
            'expected_last_known': moved.center, 'old_position': old}


def evaluate_pilot(episodes):
    records, counts, latencies = [], {}, []
    names = ('explicit_object_store', 'last_observation', 'no_memory', 'swapped_feedback',
             'cleared_memory', 'cross_episode_shuffled_memory')
    for name in names:
        counts[name] = 0
    paired, permuted, isolated, updated, invalidated, spatial = 0, 0, 0, 0, 0, 0
    positions, state_max, serialized_max = [], 0, 0
    stores = [run_history(ExplicitStore(), ep) for ep in episodes]
    # A fixed random derangement; donors from other episodes retain unrelated tags.
    donor_ids = list(range(len(episodes)))
    rng(0, 'cross_episode_shuffle').shuffle(donor_ids)
    if any(i == d for i, d in enumerate(donor_ids)):
        donor_ids = [(i+1) % len(episodes) for i in range(len(episodes))]
    for i, ep in enumerate(episodes):
        final = ep.frames[-1]
        store = stores[i]
        start = time.perf_counter_ns()
        pred = store.decide(final)
        latencies.append((time.perf_counter_ns()-start)/1e6)
        alternate = swapped_episode(ep)
        alt = run_history(ExplicitStore(), alternate).decide(final)
        paired += int(final == alternate.frames[-1] and final.observed_feedback == NO_FEEDBACK
                      and score_choice(alternate, alt) and pred['selected_marker'] != alt['selected_marker'])
        order = list(range(len(final.markers)))
        rng(ep.seed, 'permutation_control').shuffle(order)
        reordered = final.permuted(order)
        perm = store.decide(reordered)
        permuted += int(pred['selected_marker'] == perm['selected_marker'] and
                        dict(zip(final.markers, pred['candidate_probabilities'])) ==
                        dict(zip(reordered.markers, perm['candidate_probabilities'])))
        foreign = final.with_subject(final.subject_context[0] + 1)
        for f in alternate.frames[1:3]:
            store.observe(f.with_subject(foreign.subject_context[0]))
        isolated += int(store.decide(final)['selected_marker'] == pred['selected_marker']
                        and store.decide(foreign)['selected_marker'] == alt['selected_marker'])
        for f in alternate.frames[1:3]:
            store.observe(f)
        updated += score_choice(alternate, store.decide(final))
        for marker, _ in ep.truth:
            store.invalidate(final.subject_context, marker)
        invalidated += int(store.decide(final)['insufficient_evidence'] == 1 and
                           all(store.value(final.subject_context, m) is None for m, _ in ep.truth)
                           and store.decide(foreign)['selected_marker'] == alt['selected_marker'])
        store = run_history(ExplicitStore(), ep)
        cleared = run_history(ExplicitStore(), ep)
        cleared.clear()
        # Donor store is rebuilt: update/invalidation tests cannot contaminate donors.
        donor = run_history(ExplicitStore(), episodes[donor_ids[i]])
        decisions = {'explicit_object_store': pred, 'swapped_feedback': alt,
                     'last_observation': run_history(LastObservation(), ep).decide(final, force=True),
                     'no_memory': run_history(NoMemory(), ep).decide(final, force=True),
                     'cleared_memory': cleared.decide(final, force=True),
                     'cross_episode_shuffled_memory': donor.decide(final, force=True)}
        for name, dec in decisions.items():
            counts[name] += score_choice(alternate if name == 'swapped_feedback' else ep, dec)
        for m, _ in ep.truth:
            actual = next(o.center for o in ep.scene.objects if o.marker == m)
            positions.append(math.dist(store.position(m), actual))
        sc = spatial_control(ep.seed)
        spatial += int(sc['passed'])
        state_max = max(state_max, store.state_bytes())
        serialized_max = max(serialized_max, store.serialized_bytes())
        records.append({'episode': ep.seed, 'decisions': decisions, 'all_final_inputs_identical': final == alternate.frames[-1],
                        'final_input_sha256': sha(canonical(final.as_dict())), 'donor_episode': episodes[donor_ids[i]].seed,
                        'spatial_passed': sc['passed']})
    n = len(episodes)
    checks = {'paired_feedback_flip': paired, 'candidate_permutation': permuted,
              'subject_isolation': isolated, 'observed_value_update': updated,
              'explicit_invalidation': invalidated, 'spatial_motion_occlusion': spatial}
    return {'baselines': {k: proportion(v, n) for k, v in counts.items()},
            'controls': {k: proportion(v, n) for k, v in checks.items()},
            'known_position_p95_m': sorted(positions)[math.ceil(.95*len(positions))-1],
            'max_persistent_heap_bytes': state_max, 'max_serialized_payload_bytes': serialized_max,
            'latency_ms': {'median': sorted(latencies)[n//2], 'p95': sorted(latencies)[math.ceil(.95*n)-1]},
            'parameter_count': 0, 'trainable_parameter_count': 0,
            'deterministic_pass': all(v == n for v in checks.values()) and counts['explicit_object_store'] == n}, records


def capacity_curve():
    rows = []
    for budget in (32, 64, 192):
        for distractors in (4, 16, 64, 256):
            successes, evictions, heap, packed, total_ms = 0, 0, 0, 0, 0
            # Capacity diagnostics only; 10 seeds do not satisfy future G3 statistics.
            for seed in range(10):
                ep = make_episode(2_000_000+seed, distractors=distractors)
                model = ExplicitStore(budget)
                begin = time.perf_counter_ns()
                run_history(model, ep)
                total_ms += (time.perf_counter_ns()-begin)/1e6
                successes += score_choice(ep, model.decide(ep.frames[-1], force=True))
                heap, packed = max(heap, model.state_bytes()), max(packed, model.serialized_bytes())
                evictions += model.evictions
                if heap > budget*1024:
                    raise AssertionError('persistent state budget violation')
            rows.append({'budget_kib': budget, 'distractor_events': distractors, **proportion(successes, 10),
                         'max_heap_bytes': heap, 'max_serialized_bytes': packed, 'total_evictions': evictions,
                         'mean_history_latency_ms': total_ms/10,
                         'scope': 'capacity diagnostic; not G3 learning acceptance'})
    return rows


def freeze_dataset(data_dir, protocol):
    """Freeze disjoint episode groups without scoring any held-out decisions.

    Gzip mtime=0 for reproducibility. All splits are deterministic generator
    realizations; identity instances are unique per episode. Generator source
    hash is part of the recipe. Test data is not used for gate tuning.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / 'frozen_manifest.json'
    source_hashes = {p.name: sha(p.read_bytes()) for p in sorted((ROOT/'object_memory').glob('*.py'))}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing['protocol_sha256'] != sha(canonical(protocol)) or existing['source_hashes'] != source_hashes:
            raise ValueError('frozen data protocol/source changed; choose a new --data directory')
        for entry in existing['splits']:
            if sha((data_dir/entry['file']).read_bytes()) != entry['compressed_sha256']:
                raise ValueError('frozen data hash mismatch; existing data is never overwritten')
        return existing
    entries, start = [], 10_000
    seen = {'scene_seed': set(), 'object_instance': set(), 'subject_preference_instance': set()}
    for split, n in (('train', 7000), ('validation', 1000), ('test', 2000)):
        path = data_dir / f'{split}.jsonl.gz'
        if path.exists():
            raise ValueError('unmanifested split already exists; choose a fresh --data directory')
        lines_hash = hashlib.sha256()
        with path.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
                for seed in range(start, start+n):
                    ep = make_episode(seed)
                    row = ep.record()
                    all_markers = sorted({o.marker for scene in ep.scenes for o in scene.objects})
                    row['group'] = {'scene_seed': seed, 'object_instance': [f'{seed}:{m}' for m in all_markers],
                                    'subject_preference_instance': f'{seed}:0'}
                    line = canonical(row) + b'\n'
                    stream.write(line)
                    lines_hash.update(line)
                    # Explicit group disjointness checks, independent of frame indexing.
                    for key in seen:
                        values = row['group'][key] if key == 'object_instance' else [row['group'][key]]
                        if any(v in seen[key] for v in values):
                            raise AssertionError('dataset group collision')
                        seen[key].update(values)
        entries.append({'split': split, 'n': n, 'seed_start': start, 'seed_end': start+n-1,
                        'file': path.name, 'compressed_sha256': sha(path.read_bytes()),
                        'canonical_jsonl_sha256': lines_hash.hexdigest()})
        start += n
    manifest = {'schema_version': 1, 'protocol_sha256': sha(canonical(protocol)), 'splits': entries,
                'group_disjoint': True, 'heldout_scored': False,
                'source_hashes': source_hashes,
                'evaluation': 'G0/G1a pilot only; learned split evaluation remains closed'}
    write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/object-memory')
    parser.add_argument('--data', type=Path, default=ROOT/'runs/object-memory-data')
    parser.add_argument('--freeze', action='store_true', help='freeze 7000/1000/2000 only after both CPU gates pass')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.data.mkdir(parents=True, exist_ok=True)
    protocol = {'version': 1, 'scope': ['G0', 'G1a'], 'pilot_seeds': [0, 99], 'pilot_n': 100,
                'chance_seeds': [1_000_000, 1_001_999], 'chance_n': 2000,
                'capacity_seeds': [2_000_000, 2_000_009], 'budgets_kib': [32, 64, 192],
                'distractors': [4, 16, 64, 256], 'source_commit_base': '51bbc74',
                'chance_range': [.45, .55], 'deterministic_controls_required': 1.0,
                'hidden_truth_policy': 'evaluation only; ModelInput accepts six whitelisted numeric fields',
                'training_started': False, 'learned_probability_calibration_claimed': False}
    write_json(args.output/'protocol.json', protocol)  # before scores
    tracemalloc.start()
    begin = time.perf_counter()
    episodes = [make_episode(i) for i in range(100)]
    pilot_bytes = b''.join(canonical(ep.record())+b'\n' for ep in episodes)
    (args.data/'pilot.jsonl').write_bytes(pilot_bytes)
    write_json(args.output/'pilot_manifest.json', {'n': 100, 'seeds': [0, 99], 'sha256': sha(pilot_bytes),
                                                'file': 'object_memory_data/pilot.jsonl'})
    chance = chance_control(2000)
    pilot, actions = evaluate_pilot(episodes)
    capacity = capacity_curve()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    passed = chance['chance_accuracy_gate'] and pilot['deterministic_pass']
    report = {'scope': 'structured 3D environment / deterministic memory controls only',
              'gates': {'G0': 'PASS' if passed else 'FAIL', 'G1a': 'PASS' if pilot['deterministic_pass'] else 'FAIL',
                        'G1b': 'NOT_RUN', 'G2_G6': 'NOT_ACCEPTED'},
              'chance': chance, 'pilot': pilot, 'capacity': capacity,
              'runtime': {'python': sys.version, 'platform': platform.platform(), 'device': 'CPU',
                          'elapsed_seconds': time.perf_counter()-begin, 'tracemalloc_peak_bytes': peak,
                          'peak_scope': 'Python allocations for pilot/chance/capacity; excludes later freeze'},
              'limitations': ['diagnostic identity tags; no learned tracking', 'center-ray sphere sensor; no RGB-D or pointcloud',
                             'deterministic lookup probabilities; not calibrated learned predictions',
                             '100 pilot episodes and 10 capacity seeds; not G2/G3/G4 statistical acceptance',
                             'no training or hardware test; no Jev reproduction']}
    write_json(args.output/'summary.json', report)
    write_json(args.output/'decisions.json', actions)
    write_json(args.output/'spatial_example.json', spatial_control(0))
    write_json(args.output/'failure_cases.json', {'unknown': NoMemory().decide(episodes[0].frames[0]),
                                               'capacity_failures': [r for r in capacity if r['accuracy'] < 1],
                                               'unknown_semantics': 'action 0 abstains; action 1 probes within budget; force=True is scored separately'})
    if args.freeze and passed:
        manifest = freeze_dataset(args.data, protocol)
        write_json(args.output/'frozen_manifest.json', manifest)
    print(json.dumps({'gates': report['gates'], 'chance': chance, 'pilot': pilot,
                      'output': str(args.output)}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
