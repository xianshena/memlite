"""Small deterministic memory demonstration; Python standard library only."""
import json
from object_memory.environment import make_episode, swapped_episode
from object_memory.memory import ExplicitStore
from object_memory.evaluate import run_history, score_choice

def demo(n=8):
    correct = flips = unknown_after_clear = 0
    for seed in range(n):
        ep = make_episode(seed, distractors=4)
        original = run_history(ExplicitStore(), ep)
        first = original.decide(ep.frames[-1])
        other = run_history(ExplicitStore(), swapped_episode(ep)).decide(ep.frames[-1])
        assert ep.frames[-1] == swapped_episode(ep).frames[-1]
        correct += score_choice(ep, first)
        flips += first['selected_marker'] != other['selected_marker']
        original.clear()
        unknown_after_clear += original.decide(ep.frames[-1])['insufficient_evidence'] == 1
    assert correct == flips == unknown_after_clear == n
    return {'kind':'deterministic CPU smoke, not a trained model result', 'episodes':n,
            'history_choices_correct':correct,'history_swaps_flip_choice':flips,
            'cleared_memory_reports_unknown':unknown_after_clear}

if __name__ == '__main__':
    print(json.dumps(demo(), indent=2))
