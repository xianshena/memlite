"""Train the user-selected ~308M decision model, CUDA only, resumably."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time

import numpy as np
import torch

from .environment import FEATURE_NAMES, MEMORY_COLUMNS, generate_episode, execute
from .gpu_model import LargeChoiceConfig, LargeChoiceNetwork, choice_loss, teacher_distribution


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def cuda_setup(seed):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA training is required; use the separate object-memory CPU example without a GPU')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Protocol requires BF16-capable CUDA')


@contextmanager
def exclusive_gpu(output):
    handles = []
    for path in (output.parent/'gpu.lock',):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('rb' if path.exists() else 'a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError('Another GPU experiment holds the shared lock')
        handles.append(handle)
    try:
        yield
    finally:
        for handle in handles:
            handle.close()


def load_data(path, manifest, split):
    if digest(path) != manifest['splits'][split]['file_sha256']:
        raise RuntimeError('Frozen data hash mismatch: '+split)
    with np.load(path) as arrays:
        return {key: torch.as_tensor(arrays[key].copy(), device='cuda')
                for key in ('features', 'mask', 'utilities', 'targets', 'outcomes', 'seeds', 'variants')}


def evaluate(model, data, batch=64, temperature=None, clear_memory=False):
    model.eval()
    collected = {k: [] for k in ('probabilities', 'utilities', 'outcomes')}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(data['features']), batch):
            features = data['features'][start:start+batch]
            if clear_memory:
                features = features.clone()
                features[:, :, MEMORY_COLUMNS] = 0
            result = model(features, data['mask'][start:start+batch], temperature=temperature)
            for key in collected:
                collected[key].append(result[key].detach())
    predictions = {key: torch.cat(value) for key, value in collected.items()}
    probability = predictions['probabilities']
    selected = probability.argmax(-1)
    selected_u = data['utilities'].gather(1, selected[:, None]).squeeze(1)
    best_u = data['utilities'].masked_fill(~data['mask'], -1e9).amax(-1)
    correct = (best_u-selected_u) <= 1e-5
    target = teacher_distribution(data['utilities'], data['mask'])
    nll = -(target*probability.clamp_min(1e-9).log()).sum(-1)
    brier = (probability-target).square().sum(-1)
    stats = {'n': len(selected), 'tie_aware_choice_accuracy': correct.float().mean().item(),
             'mean_utility_regret': (best_u-selected_u).mean().item(),
             'choice_nll': nll.mean().item(), 'multiclass_brier_sum': brier.mean().item(),
             'utility_mae': (predictions['utilities'][data['mask']]-data['utilities'][data['mask']]).abs().mean().item(),
             'need_delta_mae': (predictions['outcomes'][data['mask']]-data['outcomes'][data['mask']]).abs().mean().item(),
             'illegal_selection_count': int((~data['mask'].gather(1, selected[:, None])).sum().item()),
             'by_variant': {}}
    for name, variant in (('daily', 0), ('experience_pair', 1), ('unknown_tie', 2)):
        keep = data['variants'] == variant
        stats['by_variant'][name] = {'n': int(keep.sum().item()),
            'tie_aware_accuracy': correct[keep].float().mean().item(),
            'mean_regret': (best_u-selected_u)[keep].mean().item(), 'nll': nll[keep].mean().item()}
    return stats, predictions


def diagnostics(model, output, seed):
    """Separate fixed post-training episodes; no test-driven training changes."""
    model.eval()
    n = 200
    a = [generate_episode(300000+i, split='test', variant='counterfactual', history_flip=bool(i % 2)) for i in range(n)]
    b = [generate_episode(300000+i, split='test', variant='counterfactual', history_flip=not bool(i % 2)) for i in range(n)]
    unknown = [generate_episode(400000+i, split='test', variant='unknown') for i in range(2000)]

    def predict(episodes, memory=True, permute=False):
        result = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(episodes), 64):
                eps = episodes[start:start+64]
                x = torch.tensor(np.stack([e.features for e in eps]), device='cuda')
                mask = torch.tensor(np.stack([e.mask for e in eps]), device='cuda')
                if not memory:
                    x[:, :, MEMORY_COLUMNS] = 0
                order = torch.arange(11, -1, -1, device='cuda') if permute else torch.arange(12, device='cuda')
                p = model(x[:, order], mask[:, order])['probabilities']
                result.extend(p[:, torch.argsort(order)].float().cpu().tolist())
        return np.asarray(result)

    pa, pb = predict(a), predict(b)
    pc = predict(a, memory=False)
    pp = predict(a, permute=True)
    pu = predict(unknown)
    ca, cb = pa.argmax(1), pb.argmax(1)
    correct_a = [int(i == e.target) for i, e in zip(ca, a)]
    correct_b = [int(i == e.target) for i, e in zip(cb, b)]
    clear_accuracy = np.mean([int(i == e.target) for i, e in zip(pc.argmax(1), a)])
    true_unknown = [next(i for i, c in enumerate(e.candidates) if c.activity == e.meta['unknown_hidden_winner']) for e in unknown]
    unknown_correct = np.mean(pu.argmax(1) == np.asarray(true_unknown))
    u_target = pu[np.arange(len(unknown)), true_unknown]
    record = {'n_counterfactual_pairs': n, 'original_accuracy': float(np.mean(correct_a)),
              'swapped_accuracy': float(np.mean(correct_b)),
              'both_correct_and_flip': float(np.mean(np.asarray(correct_a)*np.asarray(correct_b)*(ca != cb))),
              'clear_memory_forced_accuracy': float(clear_accuracy),
              'candidate_permutation_probability_max_abs_diff': float(np.max(np.abs(pa-pp))),
              'unknown_n': len(unknown), 'unknown_forced_accuracy': float(unknown_correct),
              'unknown_mean_abs_p_minus_half': float(np.mean(np.abs(pu[np.stack([e.mask for e in unknown]) ]-.5))),
              'unknown_maxprob_ge_0_9_fraction': float(np.mean(pu.max(1) >= .9)),
              'unknown_binary_nll': float(-np.log(np.maximum(u_target, 1e-9)).mean()),
              'unknown_binary_brier': float(((u_target-1)**2).mean()),
              'all_nonmemory_current_fields_equal': all(np.array_equal(e.features[:, :38], f.features[:, :38])
                                                        and np.array_equal(e.mask, f.mask) for e, f in zip(a, b)),
              'feedback_sentinel_equal': all(e.meta['current_feedback'] == f.meta['current_feedback'] == [0, 0.0]
                                             for e, f in zip(a, b))}
    paths = []
    for ep, choice, p in zip(a[:12], ca[:12], pa[:12]):
        paths.append({'episode_seed': ep.meta['seed'], 'probabilities': p.tolist(),
                      'selected_index': int(choice), 'execution': execute(ep, int(choice))})
    dump(output/'diagnostics.json', record)
    dump(output/'action_examples.json', paths)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(0, 1, 2))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.epochs < 3:
        raise ValueError('The reference protocol requires at least three complete epochs')
    args.output.mkdir(parents=True, exist_ok=True)
    with exclusive_gpu(args.output):
        return train(args)


def train(args):
    cuda_setup(args.seed)
    manifest = json.loads((args.data/'manifest.json').read_text())
    config = LargeChoiceConfig()
    model = LargeChoiceNetwork(config).cuda()
    count = model.parameter_count()
    if not 290_000_000 <= count <= 330_000_000:
        raise RuntimeError('Model does not satisfy the approved ~300M parameter size')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.05, foreach=False)
    train_data = load_data(args.data/'train.npz', manifest, 'train')
    validation = load_data(args.data/'validation.npz', manifest, 'validation')
    n = len(train_data['features'])
    with torch.no_grad():
        visible = train_data['features'][train_data['mask']]
        model.feature_mean.copy_(visible.mean(0))
        model.feature_scale.copy_(visible.std(0).clamp_min(.1))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    data_hash = digest(args.data/'manifest.json')
    generator = torch.Generator(device='cuda').manual_seed(7000+args.seed)
    checkpoint = args.output/'latest.pt'
    epoch, cursor, steps, order = 0, 0, 0, None
    if args.resume:
        saved = torch.load(checkpoint, map_location='cuda')
        if saved['source_hashes'] != source_hashes or saved['data_hash'] != data_hash:
            raise RuntimeError('Checkpoint source/data mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, cursor, steps, order = saved['epoch'], saved['cursor'], saved['steps'], saved['order']
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        torch.set_rng_state(saved['cpu_rng'].cpu())
        generator.set_state(saved['order_rng'].cpu())
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        del saved
    elif checkpoint.exists():
        raise RuntimeError('Existing checkpoint requires --resume')
    device = {'name': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0),
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'bf16': True}
    parameter_initial = model.utility_head.weight.detach().clone()
    started = time.time()
    state = {'state': 'TRAINING_GPU', 'pid': os.getpid(), 'seed': args.seed, 'epochs_target': args.epochs,
             'parameters': count, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
             'device': device, 'source_hashes': source_hashes, 'data_hash': data_hash, 'cpu_training_fallback': False,
             'config': asdict(config), 'training_kind': 'supervised choice and value/outcome prediction; not RLCD or online RL',
             'wall_time_limit_seconds': None, 'epochs_complete': epoch, 'steps': steps, 'started_unix': started}
    dump(args.output/'state.json', state)
    stopping = False
    def request_stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    last_save = time.time()
    torch.cuda.reset_peak_memory_stats()

    def save():
        nonlocal last_save
        # FP32 weights and Adam states: single resumable checkpoint. Parent must
        # archive a finished seed before launching the next, preserving old runs.
        expected = count*12 + 64*1024*1024
        if shutil.disk_usage(args.output).free < expected + 256*1024*1024:
            raise RuntimeError('Insufficient disk for atomic resumable checkpoint; no old files deleted')
        tmp = checkpoint.with_suffix('.pt.tmp')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'config': asdict(config), 'source_hashes': source_hashes, 'data_hash': data_hash,
                    'epoch': epoch, 'cursor': cursor, 'steps': steps, 'order': order,
                    'cuda_rng': torch.cuda.get_rng_state_all(), 'cpu_rng': torch.get_rng_state(),
                    'order_rng': generator.get_state(), 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}, tmp)
        os.replace(tmp, checkpoint)
        last_save = time.time()
        state['checkpoint'] = {'file': str(checkpoint), 'bytes': checkpoint.stat().st_size, 'sha256': digest(checkpoint)}
        dump(args.output/'state.json', state)

    log = (args.output/'training.jsonl').open('a', buffering=1)
    try:
        while epoch < args.epochs and not stopping:
            if order is None:
                order = torch.randperm(n, device='cuda', generator=generator)
                cursor = 0
            model.train()
            while cursor < n and not stopping:
                idx = order[cursor:cursor+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                begin = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(train_data['features'][idx], train_data['mask'][idx])
                    loss, parts = choice_loss(prediction, train_data['mask'][idx], train_data['utilities'][idx], train_data['outcomes'][idx])
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Non-finite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError('Non-finite gradient')
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-begin
                cursor += len(idx)
                steps += 1
                record = {'epoch': epoch+1, 'cursor': cursor, 'n': n, 'step': steps,
                          'loss': loss.item(), **parts, 'gradient_norm': norm.item(), 'step_seconds': elapsed,
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
                log.write(json.dumps(record)+'\n')
                state.update(steps=steps, epoch_active=epoch+1, cursor=cursor, last_step=record,
                             elapsed_seconds=time.time()-started)
                if steps % 5 == 0 or steps == 1:
                    dump(args.output/'state.json', state)
                if time.time()-last_save >= 1800:
                    save()
            if cursor == n:
                epoch += 1
                order, cursor = None, 0
                metrics, _ = evaluate(model, validation)
                log.write(json.dumps({'validation_epoch': epoch, **metrics})+'\n')
                state.update(epochs_complete=epoch, validation=metrics)
                dump(args.output/'state.json', state)
                if epoch % 3 == 0 or epoch == args.epochs:
                    save()
        if stopping:
            state['state'] = 'SAVED_STOPPED'
            save()
            return 2
        # The frozen test split is opened only after all scheduled epochs.
        test = load_data(args.data/'test.npz', manifest, 'test')
        metrics, prediction = evaluate(model, test)
        clear_metrics, _ = evaluate(model, test, clear_memory=True)
        diag = diagnostics(model, args.output, args.seed)
        state.update(state='COMPLETE', epochs_complete=epoch, steps=steps, test=metrics,
                     test_clear_memory=clear_metrics, diagnostics=diag,
                     utility_head_l2_change=float(torch.linalg.vector_norm(model.utility_head.weight-parameter_initial).item()),
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.time()-started)
        dump(args.output/'state.json', state)
        dump(args.output/'test_metrics.json', metrics)
        print(json.dumps(state, indent=2))
        return 0
    except Exception as exc:
        state.update(state='FAILED', error=repr(exc), epochs_complete=epoch, steps=steps, cursor=cursor)
        dump(args.output/'state.json', state)
        # Do not erase the last successful atomic checkpoint on an error.
        raise
    finally:
        log.close()


if __name__ == '__main__':
    raise SystemExit(main())
