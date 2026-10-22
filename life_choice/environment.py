"""A 10 x 10 metre numerical household, with observed experience and needs.

This is structured state, not RGB-D perception, biological hormones, or learned
visual tracking. A candidate is navigate-to-place AND finish-activity. The
teacher sees the same known conditions and observed history as the learner.
Unobserved preferences are evaluator-only and never teacher supervision.
"""
from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ACTIVITIES = ('drink', 'cook', 'eat', 'sleep', 'rest', 'toilet', 'wash',
              'shower', 'work', 'read', 'exercise', 'entertainment')
NEED_NAMES = ('thirst', 'hunger', 'fatigue', 'bladder', 'hygiene', 'boredom')
CUE_NAMES = ('food_video', 'water_cue', 'phone_cue', 'work_cue', 'fitness_cue', 'quiet_cue')
FEATURE_NAMES = (tuple('body_'+x for x in NEED_NAMES)
                 + tuple('cue_'+x for x in CUE_NAMES)
                 + ('time_sin', 'time_cos', 'ego_x', 'ego_y', 'ego_z')
                 + tuple('activity_'+x for x in ACTIVITIES)
                 + ('relative_x', 'relative_y', 'relative_z', 'resource_available',
                    'prerequisite_met', 'path_distance', 'meal_prepared',
                    'ingredients_available', 'future_eat_path_distance', 'memory_mean', 'memory_known'))
MEMORY_COLUMNS = (38, 39)
INPUT_KEYS = frozenset(('feature_names', 'features', 'mask', 'current_feedback'))
VERSION = 'household-choice-v1'
SPLIT_BASE = {'train':0, 'validation':100000, 'test':200000}
SPLIT_DOORS = {'train':((1,2),(6,7)), 'validation':((3,),(6,7,8,9)), 'test':((4,),(6,7,8,9))}
# A negative entry relieves a need; these are explicit simulator assumptions.
# Cooking prepares a meal; it does not count as having eaten it.
EFFECTS = np.array([
    [-.75,0,0,.12,0,-.02], [0,0,.04,0,0,-.12],
    [0,-.75,0,.02,0,-.08], [.04,.06,-.9,.02,0,-.04],
    [0,0,-.35,0,0,-.12], [0,0,0,-.85,0,0],
    [0,0,0,0,-.35,0], [0,0,-.06,0,-.85,-.08],
    [.04,.04,.12,.02,0,-.15], [0,0,.03,0,0,-.35],
    [.18,.12,.2,0,.12,-.55], [0,0,.04,0,0,-.65]
], dtype=np.float32)
CUE_AFFINITY = np.array([
    [0,1,0,0,0,0], [.8,0,0,0,0,0], [1,0,0,0,0,0],
    [0,0,0,0,0,.25], [0,0,0,0,0,.45], [0,0,0,0,0,0],
    [0,0,0,0,0,0], [0,0,0,0,.1,0], [0,0,0,1,0,0],
    [0,0,0,0,0,1], [0,0,0,0,1,0], [0,0,1,0,0,0]
], dtype=np.float32)
CUE_VALUE_SCALE = .9
VARIANT_NAMES = ('daily','counterfactual','unknown')


def _rng(seed, split, domain):
    digest = hashlib.sha256(f'{VERSION}/{split}/{domain}/{int(seed)}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], 'little'))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    activity: str
    destination: tuple
    available: bool
    prerequisite_met: bool
    path_distance: float


@dataclass(frozen=True)
class Episode:
    features: np.ndarray
    mask: np.ndarray
    teacher_utilities: np.ndarray
    target: int
    outcome_targets: np.ndarray  # observable expected post-completion need delta
    body: np.ndarray
    candidates: tuple
    history: tuple
    meta: dict  # evaluator and diagnostics only, never fed to model

    def as_record(self):
        return {
            'inputs': {'feature_names':list(FEATURE_NAMES), 'features':self.features.tolist(),
                       'mask':self.mask.tolist(), 'current_feedback':self.meta['current_feedback']},
            'observed_history':list(self.history),
            'evaluator_only':dict(self.meta, candidates=[asdict(c) for c in self.candidates],
                                 body=self.body.tolist(), teacher_utilities=self.teacher_utilities.tolist(),
                                 target=self.target, outcome_targets=self.outcome_targets.tolist())}


def schema_audit(record):
    """Fail closed on unknown model-input fields and malformed numeric arrays."""
    errors = []
    inputs = record.get('inputs', {})
    if set(inputs) != INPUT_KEYS:
        errors.append('input key whitelist mismatch')
    if inputs.get('feature_names') != list(FEATURE_NAMES):
        errors.append('feature order mismatch')
    try:
        raw_features = np.asarray(inputs['features'])
        features = np.asarray(inputs['features'], dtype=np.float32)
        mask = np.asarray(inputs['mask'])
        if (features.shape != (12,len(FEATURE_NAMES)) or not np.isfinite(features).all()
                or raw_features.dtype.kind not in 'fiu'):
            errors.append('feature numeric shape/finiteness')
        if mask.shape != (12,) or mask.dtype != np.bool_:
            errors.append('mask shape/type')
        if inputs.get('current_feedback') != [0, 0.0]:
            errors.append('final current feedback must be no-feedback sentinel')
    except (KeyError, TypeError, ValueError):
        errors.append('non-numeric or missing model input')
    return errors


def _tree(start, blocked):
    """Unit-cost, four-neighbour base navigation; z handled by arm reach."""
    parent = {start:None}
    queue = deque([start])
    distance = {start:0}
    while queue:
        x,y = queue.popleft()
        for q in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= q[0] < 10 and 0 <= q[1] < 10 and q not in blocked and q not in parent:
                parent[q] = (x,y)
                distance[q] = distance[x,y]+1
                queue.append(q)
    return parent, distance


def _history_means(history, subject):
    groups = {}
    for event in history:
        if (event['subject_group'] == subject and event['observed']
                and event['arrived'] and event['completed']):
            groups.setdefault(event['activity'], []).append(float(event['feedback']))
    return {key:float(np.mean(values)) for key,values in groups.items()}


def _history_event(subject, activity, feedback):
    return {'subject_group':subject, 'activity':activity, 'arrived':True,
            'completed':True, 'observed':True, 'feedback':float(feedback)}


def generate_episode(seed, split='train', variant='daily', history_flip=False):
    """Independent household decision with a short personally observed history.

    Split is a whole scene/subject/object cohort, never a frame split. Separate
    RNG domains prevent preference or candidate order acting as accidental IDs.
    Counterfactual and unknown use matched drink/eat opportunities. They are
    diagnostic tasks, deliberately distinct from normal multi-need decisions.
    """
    if split not in SPLIT_BASE:
        raise ValueError('split must be train, validation, or test')
    if variant not in ('daily','counterfactual','unknown'):
        raise ValueError('unknown episode variant')
    layout = _rng(seed,split,'layout')
    needs = _rng(seed,split,'needs')
    cue_rng = _rng(seed,split,'cue')
    hidden = _rng(seed,split,'unobserved_preference')
    history_rng = _rng(seed,split,'experience')
    order_rng = _rng(seed,split,'candidate_order')
    identity = f'{split}/{seed}'
    subject = 'subject/'+_sha(identity)[:20]
    scenario = int(seed)%12
    # Quotas balance situation prompts, not the teacher's eventual labels.
    # A work notification, reading opportunity or exercise invitation occurs
    # in a low-need state often enough to be a meaningful competing goal.
    body = (needs.beta(1.4,1.8,6)*.35).astype(np.float32)
    scenario_need = {0:(0,.8),1:(1,.8),2:(1,.8),3:(2,.8),4:(2,.45),
                     5:(3,.85),6:(4,.45),7:(4,.85),11:(5,.65)}
    if scenario in scenario_need:
        need_index,level = scenario_need[scenario]
        body[need_index] = min(1.0,float(needs.uniform(level,level+.14)))
    cues = np.zeros(6, dtype=np.float32)
    scenario_cue = {0:1,1:0,2:0,3:5,4:5,5:1,6:5,7:5,8:3,9:5,10:4,11:2}
    cue_index = scenario_cue[scenario]
    cues[cue_index] = float(cue_rng.uniform(.7,1.0))
    if cue_rng.random() < .25:
        extra_cue = int(cue_rng.integers(6))
        cues[extra_cue] = max(cues[extra_cue],float(cue_rng.uniform(.1,.4)))
    hour = float(needs.uniform(0,24))
    time = (math.sin(hour*math.pi/12),math.cos(hour*math.pi/12))
    # Interior walls have split-specific door geometries: held-out scenes
    # differ physically, not only in a diagnostic scene label.
    doors = tuple(int(layout.choice(choices)) for choices in SPLIT_DOORS[split])
    blocked = {(5,y) for y in range(10) if y not in doors}
    blocked |= {(x,5) for x in range(10) if x not in doors}
    starts = [(x,y) for x in range(10) for y in range(10) if (x,y) not in blocked]
    start = starts[int(layout.integers(len(starts)))]
    rooms = ([(x,y) for x in range(5) for y in range(5)],
             [(x,y) for x in range(6,10) for y in range(6,10)],
             [(x,y) for x in range(5) for y in range(6,10)],
             [(x,y) for x in range(6,10) for y in range(5)])
    cells = []
    for room,count in zip(rooms,(3,2,3,4)):
        cells.extend(room[i] for i in layout.choice(len(room),size=count,replace=False))
    # All rooms remain accessible under this layout. Vary furniture harmlessly.
    reserved = set(cells) | {start} | {(5,d) for d in doors} | {(d,5) for d in doors}
    furniture = [p for room in rooms for p in room]
    layout.shuffle(furniture)
    furniture = furniture[:int(layout.integers(3,9))]
    for cell in furniture:
        if cell not in reserved:
            trial = blocked | {cell}
            _,reachable = _tree(start,trial)
            if all(c in reachable for c in cells):
                blocked.add(cell)
    parents, distances = _tree(start,blocked)
    heights = (.95,.9,.78,.48,.5,.45,.9,1.65,.75,.75,.1,.75)
    available = history_rng.random(12) > .08
    prerequisites = history_rng.random(12) > .12
    # Rest/toilet are always physically available in the household.
    available[[4,5]] = True
    prerequisites[[4,5]] = True
    resources = {'meal_prepared':bool(history_rng.random() < .55),
                 'ingredients_available':bool(history_rng.random() < .9),
                 'water_available':bool(history_rng.random() < .94)}
    if scenario == 1:
        resources.update(meal_prepared=False,ingredients_available=True)
    elif scenario == 2:
        resources['meal_prepared'] = True
    available[0] = resources['water_available']
    available[1] = resources['ingredients_available']
    available[2] = resources['meal_prepared']
    prerequisites[[0,1,2]] = True
    preferences = hidden.uniform(-1,1,12).astype(np.float32)
    history = []
    for i,activity in enumerate(ACTIVITIES):
        if history_rng.random() < .7:
            for repeat in range(int(history_rng.integers(1,4))):
                feedback = float(np.clip(preferences[i]+history_rng.normal(0,.04),-1,1))
                history.append(_history_event(subject,activity,feedback))
    if variant != 'daily':
        body[:] = 0
        cues[:] = 0
        # Both candidates have exactly matched position and known resources.
        cells[2] = cells[0]
        available[:] = False
        prerequisites[:] = False
        available[[0,2]] = True
        prerequisites[[0,2]] = True
        resources = {'meal_prepared':True,'ingredients_available':True,'water_available':True}
        history = []
        if variant == 'counterfactual':
            sign = -1 if history_flip else 1
            history = [_history_event(subject,'drink',sign), _history_event(subject,'eat',-sign)]
            preferences[0],preferences[2] = sign,-sign
    means = _history_means(history,subject)
    candidates, features, mask, outcomes, utilities = [],[],[],[],[]
    ego = (start[0]+.5,start[1]+.5,.9)
    _,from_cook = _tree(cells[1],blocked)
    future_eat_distance = float(from_cook.get(cells[2],100))
    # A finite two-step simulator prior: prepare, then eat. No immediate hunger
    # relief is credited to cooking's physical outcome. The planning bonus is
    # used only if no meal is currently ready, from known meal/resource state.
    future_eat_delta = np.clip(body+EFFECTS[2],0,1)-body
    cook_future_value = (.85*2.5*float(np.dot(body,-future_eat_delta))
                         - .012*future_eat_distance)
    cook_future_value = max(0.0,cook_future_value) if not resources['meal_prepared'] else 0.0
    for i,activity in enumerate(ACTIVITIES):
        cell = cells[i]
        destination = (cell[0]+.5,cell[1]+.5,heights[i])
        distance = float(distances.get(cell,100))
        legal = bool(available[i] and prerequisites[i] and cell in parents and destination[2] <= 1.8)
        after = np.clip(body+EFFECTS[i],0,1).astype(np.float32)
        delta = after-body if legal else np.zeros(6,dtype=np.float32)
        # Need relief is weighted by the current observable need. Known cue
        # attraction and travel cost are deliberate finite teacher assumptions.
        # Experienced value is the mean of this subject's observed feedback.
        utility = (float(np.dot(body,-delta))*2.5
                   + CUE_VALUE_SCALE*float(np.dot(cues,CUE_AFFINITY[i]))
                   + .65*means.get(activity,0.0) - .012*distance)
        if activity == 'cook':
            utility += cook_future_value
        if variant == 'unknown':
            # Equal known travel costs, unknown balanced preference mean zero.
            # Keeping the common cost retains expected-reward consistency.
            utility = -.012*distance
        if not legal:
            utility = -10.0
        onehot = [float(j==i) for j in range(12)]
        row = list(body)+list(cues)+list(time)+[x/10 for x in ego]+onehot
        row += [(destination[j]-ego[j])/10 for j in range(3)]
        row += [float(available[i]),float(prerequisites[i]),distance/20,
                float(resources['meal_prepared']),float(resources['ingredients_available']),future_eat_distance/20,
                means.get(activity,0.0),float(activity in means)]
        candidates.append(Candidate(activity,destination,bool(available[i]),bool(prerequisites[i]),distance))
        features.append(row); mask.append(legal); outcomes.append(delta); utilities.append(utility)
    permutation = order_rng.permutation(12)
    candidates = tuple(candidates[i] for i in permutation)
    features = np.asarray(features,dtype=np.float32)[permutation]
    mask = np.asarray(mask,dtype=np.bool_)[permutation]
    utilities = np.asarray(utilities,dtype=np.float32)[permutation]
    outcomes = np.asarray(outcomes,dtype=np.float32)[permutation]
    target = int(np.argmax(utilities))
    legal_probabilities = [float(1/mask.sum()) if v else 0.0 for v in mask]
    scene_geometry = {'blocked_cells':[list(p) for p in sorted(blocked)],
                      'destinations':[list(c.destination) for c in candidates]}
    meta = {'version':VERSION,'seed':int(seed),'split':split,'variant':variant,
            'subject_group':subject,'scene_group':'scene/'+_sha(identity+'/scene')[:20],
            'object_group':'objects/'+_sha(identity+'/objects')[:20],
            'start_cell':list(start),'ego':list(ego),'blocked_cells':[list(p) for p in sorted(blocked)],
            'scene_area_m2':100,'ceiling_m':2.6,'current_feedback':[0,0.0],
            'scene_geometry_sha256':_sha(scene_geometry),'door_cells':list(doors),
            'resources':resources,'cook_two_step_need_value':cook_future_value,
            'two_step_discount':.85,'future_eat_path_distance_m':future_eat_distance,
            'external_stimulus':CUE_NAMES[cue_index] if variant=='daily' else 'none',
            'scenario':ACTIVITIES[scenario]+'_opportunity' if variant=='daily' else variant,
            'known_cue_value':{a:CUE_VALUE_SCALE*float(np.dot(cues,CUE_AFFINITY[i]))
                               for i,a in enumerate(ACTIVITIES)},
            'hidden_preferences':{a:float(v) for a,v in zip(ACTIVITIES,preferences)},
            'teacher_probabilities':legal_probabilities if variant=='unknown' else None,
            'target_is_arbitrary_tiebreak':variant=='unknown',
            'unknown_hidden_winner':('drink','eat')[int(hidden.integers(2))] if variant=='unknown' else None}
    return Episode(features,mask,utilities,target,outcomes,body,candidates,tuple(history),meta)


def features_without_memory(episode):
    """Clear experience features while preserving every current observation."""
    features = episode.features.copy()
    features[:,MEMORY_COLUMNS] = 0
    return features


def reorder_episode(episode, permutation):
    """Candidate order is presentation only; labels and results follow it."""
    permutation = np.asarray(permutation,dtype=np.int64)
    if permutation.shape != (12,) or sorted(permutation.tolist()) != list(range(12)):
        raise ValueError('candidate permutation must contain 0..11 exactly once')
    target = int(np.flatnonzero(permutation == episode.target)[0])
    meta = dict(episode.meta)
    if meta['teacher_probabilities'] is not None:
        meta['teacher_probabilities'] = [meta['teacher_probabilities'][i] for i in permutation]
    return replace(episode,features=episode.features[permutation].copy(),mask=episode.mask[permutation].copy(),
                   teacher_utilities=episode.teacher_utilities[permutation].copy(),target=target,
                   outcome_targets=episode.outcome_targets[permutation].copy(),
                   candidates=tuple(episode.candidates[i] for i in permutation),meta=meta)


def with_history(episode, history, subject=None):
    """Rebuild only the memory columns; useful for subject-isolation controls.

    Teacher arrays are retained as the original evaluation reference. This is
    an ablation input transformation, not a newly relabelled training episode.
    """
    features = episode.features.copy()
    means = _history_means(history,subject or episode.meta['subject_group'])
    for i,candidate in enumerate(episode.candidates):
        features[i,MEMORY_COLUMNS[0]] = means.get(candidate.activity,0)
        features[i,MEMORY_COLUMNS[1]] = float(candidate.activity in means)
    return replace(episode,features=features,history=tuple(history))


def execute(episode, index):
    """Navigate before action; feedback becomes observable only on completion.

    Reward = observed weighted need relief + known cue satisfaction + new
    preference - travel cost + the explicitly labelled finite planning value.
    Only the newly experienced preference differs from expected teacher value.
    No pre-action input contains this newly realised result.
    """
    if not 0 <= int(index) < len(episode.candidates):
        raise IndexError(index)
    index = int(index)
    candidate = episode.candidates[index]
    start = tuple(episode.meta['start_cell'])
    blocked = {tuple(p) for p in episode.meta['blocked_cells']}
    parents,_ = _tree(start,blocked)
    goal = (int(candidate.destination[0]),int(candidate.destination[1]))
    arrived = goal in parents
    path_cells = []
    if arrived:
        p = goal
        while p is not None:
            path_cells.append(p); p = parents[p]
        path_cells.reverse()
    legal = (arrived and candidate.available and candidate.prerequisite_met
             and candidate.destination[2] <= 1.8 and bool(episode.mask[index]))
    feedback = None
    end_body = episode.body.copy()
    reward = -.1
    planning_value = 0.0
    myopic_reward = -.1
    resources_after = dict(episode.meta['resources'])
    if legal:
        feedback = episode.meta['hidden_preferences'][candidate.activity]
        if episode.meta['variant'] == 'unknown':
            feedback = 1.0 if candidate.activity == episode.meta['unknown_hidden_winner'] else -1.0
        action_id = ACTIVITIES.index(candidate.activity)
        end_body = np.clip(episode.body+EFFECTS[action_id],0,1).astype(np.float32)
        myopic_reward = (2.5*float(np.dot(episode.body,episode.body-end_body))
                         + episode.meta['known_cue_value'][candidate.activity]
                         +.65*float(feedback)-.012*candidate.path_distance)
        planning_value = episode.meta['cook_two_step_need_value'] if candidate.activity=='cook' else 0.0
        reward = myopic_reward+planning_value
        if candidate.activity == 'cook':
            resources_after['meal_prepared'] = True
            resources_after['ingredients_available'] = False
        elif candidate.activity == 'eat':
            resources_after['meal_prepared'] = False
    return {'arrived':bool(arrived),'completed':bool(legal),
            'path':[[x+.5,y+.5,0.0] for x,y in path_cells],
            'end_body':end_body.tolist(),'feedback':feedback,'realized_reward':float(reward),
            'myopic_realized_reward':float(myopic_reward),'planning_value':float(planning_value),
            'action':candidate.activity,'destination':list(candidate.destination),
            'resources_after':resources_after,
            'completion_requires':['valid_path','arrived','resource_available','prerequisite_met','vertical_reach']}


def freeze_dataset(outdir, train_n=10000, val_n=1000, test_n=2000):
    """Freeze disjoint complete episodes; refuse to overwrite existing data."""
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty dataset: {outdir}')
    outdir.mkdir(parents=True,exist_ok=True)
    counts = {'train':int(train_n),'validation':int(val_n),'test':int(test_n)}
    if min(counts.values()) < 1:
        raise ValueError('every split needs at least one complete episode')
    normalized_source = Path(__file__).read_text(encoding='utf-8').replace('\r\n','\n')
    manifest = {'version':VERSION,'scene_area_m2':100,'split_counts':counts,
                'independent_contexts_total':sum(counts.values()),
                'variant_codes':{name:i for i,name in enumerate(VARIANT_NAMES)},
                'variant_ratio':{'daily':.8,'counterfactual':.15,'unknown':.05},
                'feature_names':list(FEATURE_NAMES),'memory_columns':list(MEMORY_COLUMNS),
                'source_sha256':hashlib.sha256(normalized_source.encode()).hexdigest(),
                'source_hash_eol':'LF-normalized UTF-8','splits':{},
                'teacher':'known needs + known cue attraction + observed personal experience - path cost',
                'unvalidated':['visual perception','biological hormone dynamics','real robot safety','human behavior realism']}
    with (outdir/'representative.jsonl').open('w',encoding='utf-8',newline='\n') as example_file:
        for split,count in counts.items():
            seeds = list(range(SPLIT_BASE[split],SPLIT_BASE[split]+count))
            # Every complete block of 20 has 16 daily, 3 history-sensitive and
            # 1 no-experience decision. History signs alternate independently.
            variants = np.asarray([0 if i%20<16 else 1 if i%20<19 else 2
                                   for i in range(count)],dtype=np.uint8)
            episodes = []
            counterfactual_index = 0
            for seed,variant in zip(seeds,variants):
                flip = bool(counterfactual_index%2) if variant==1 else False
                episodes.append(generate_episode(seed,split=split,variant=VARIANT_NAMES[variant],history_flip=flip))
                if variant==1:
                    counterfactual_index += 1
            arrays = {'features':np.stack([e.features for e in episodes]),
                      'mask':np.stack([e.mask for e in episodes]),
                      'utilities':np.stack([e.teacher_utilities for e in episodes]),
                      'targets':np.asarray([e.target for e in episodes],dtype=np.int64),
                      'outcomes':np.stack([e.outcome_targets for e in episodes]),
                      'seeds':np.asarray(seeds,dtype=np.int64),'variants':variants}
            target = outdir/f'{split}.npz'
            np.savez_compressed(target,**arrays)
            for ep in episodes[:3]:
                example_file.write(json.dumps(ep.as_record(),ensure_ascii=False,sort_keys=True)+'\n')
            groups = {name:[e.meta[name[:-1]] for e in episodes]
                      for name in ('subject_groups','scene_groups','object_groups')}
            manifest['splits'][split] = dict(groups,seeds=seeds,
                physical_scene_hashes=sorted({e.meta['scene_geometry_sha256'] for e in episodes}),
                entity_groups_sha256=_sha(groups), file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                variant_counts={name:int((variants==i).sum()) for i,name in enumerate(VARIANT_NAMES)},
                counterfactual_history_sign_counts={'positive':(counterfactual_index+1)//2,
                                                    'negative':counterfactual_index//2},
                count=count)
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest
