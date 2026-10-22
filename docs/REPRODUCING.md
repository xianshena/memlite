# Reproduction scope / 复现边界

This is an audit-and-source distribution, not a claim of newly reproduced GPU results. Commands run from the package root. `runs/` is the only default location for new CPU evaluation/training output. Nothing writes to the source project.

## CPU and existing-record audits

The standard-library object-memory demo does eight tiny episodes; it has zero trainable parameters. Existing object-memory tests include 2,000 independent no-experience trials. The NumPy household suite covers schemas, geometry, visible/history isolation, resources and small split freezing. These use fresh temporary directories and never change included frozen data.

`python -m scripts.audit_recorded_results` checks all 6,000 included decision rows, probability shape/sums, argmax selection, scenario buckets, action totals, and frozen-array hashes without a neural network. Add `--with-frozen-data` to recompute the reported daily counts/positive regrets using the included numerical teacher utilities (NumPy required). It does not establish human behavioral correctness or independent experimental replication.

## Frozen synthetic data

`data/life_choice/` contains the original frozen numerical arrays and manifest (10,000/1,000/2,000). Use those arrays for reference comparisons. Generator-only regeneration:

```sh
python -m scripts.freeze_household_data --output runs/regenerated-data
```

This writes a NEW directory and refuses a nonempty destination. Original generation used NumPy 2.4.5; the historical GPU runtime used NumPy 1.26.4. Release QA uses NumPy 2.3.5. Compressed NPZ container bytes and regenerated FP32 values may differ by version; do not claim a new byte-identical freeze without checking. Original post-action metadata reconstruction used atol=2e-6/rtol=0 and exact masks, while scoring used hash-verified frozen utilities.

## Optional CUDA reference (NOT run during release preparation)

Use a separately prepared Linux CUDA/BF16 environment. `fcntl` makes the training entrypoint Unix-only. Historical reference: NVIDIA Torch `2.3.0a0+6ddf5cf85e.nv24.04`, CUDA 12.2, NumPy 1.26.4. `requirements-gpu-reference.txt` documents part of this environment; it is not a complete lock or a substitute for a supported vendor Torch install. No new dependency installation is implied by release QA.

```sh
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
for seed in 0 1 2; do
  python -m life_choice.train_gpu --seed "$seed" --data data/life_choice --output "runs/seed$seed" --epochs 12 --batch-size 128 || break
done
```

The model has 308,487,175 trainable parameters, batch128, AdamW lr1e-4/weight decay.05/dropout.1/gradient clip1; loss = tie-aware CE + .25 utility Huber + .1 need-delta Huber. Each epoch validates; the fixed final epoch opens test data. Three 12-epoch runs are sequential, not 36 passes through one evolving network. About 6.85 decimal GB was historically allocated at peak, but target hardware requirements must be remeasured. A full resumable checkpoint is about 3.70 GB, with additional free space required for safe atomic saving. No weights are bundled.

The original global device-specific lock was replaced by `output.parent/gpu.lock`; place all competing new GPU runs under the same parent. Source hashes differ for this entrypoint and are recorded in the mapping; neither exact old-checkpoint compatibility nor GPU execution of the adaptation was verified here. On an interruption, `--resume` uses the same data/source hashes. Only load a trusted, self-generated Torch checkpoint; old Torch checkpoints use Python pickle. Full-size interrupted/continuous bitwise equivalence remains unverified.

Training writes `test_metrics.json`, history-swap/clear-memory/unknown diagnostics, 12 action examples, and per-step logs. To export every decision from a new self-trained checkpoint:

```sh
python -m scripts.export_checkpoint_decisions --checkpoint runs/seed0/latest.pt --data data/life_choice --output runs/seed0-actions
```

This CUDA export command was syntax-checked only. It has not been executed on a GPU during release preparation. It uses the fixed original test arrays, validates reconstructed metadata and writes a new audit directory. Do not overwrite the recorded evidence.

## Not reproduced or included

- GPU optimization, CUDA numerical smoke, fresh checkpoint export, large-model recovery equivalence.
- Object-memory G1b/G2–G6 learned-model acceptance, real images, hormones, motors or live robotics.
- Teacher inference/expansion automation, external accounts, paid APIs, large weights, personal data, remote transport or credentials.
- A complete portable vendor environment lock or a fair small/large-model comparison.
