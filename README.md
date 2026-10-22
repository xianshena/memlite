![MemLite](assets/header.svg)

# MemLite: Reproducible Memory-Conditioned Decision Learning

[中文](README.zh-CN.md) · [Results and limitations](docs/RESULTS.md) · [Reproduction](docs/REPRODUCING.md) · [Licenses](docs/RIGHTS_AND_LICENSES.md)

A learning-driven experiment in how explicit personal experience changes choices
in synthetic scenes. Start with a tiny CPU example, inspect three recorded 308M
training seeds, and audit individual decisions against frozen simulator utilities.
This is an early research project; external adoption and independent replication
have not been demonstrated.

## Run the CPU example

Python 3.11+; run from the repository root. These commands use the standard library:

```sh
python -m examples.cpu_quickstart
python -m unittest discover -s tests -p test_preview_smoke.py -v
python -m unittest discover -s tests -p test_object_memory.py -v
python -m scripts.audit_recorded_results
```

To check the household environment and re-score frozen records:

```sh
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -p test_life_environment.py -v
python -m scripts.audit_recorded_results --with-frozen-data
```

The object example has zero trainable parameters. CPU tests demonstrate software
contracts and recorded-data consistency, not new neural training.

## Recorded 308M experiment

A randomly initialized 308,487,175-parameter numerical relation encoder trained
for 12 epochs / 948 updates per seed on 10,000 independent synthetic contexts.
All three seeds reuse the same 2,000 test scenes: 1,600 daily + 300 experience +
100 unknown utility ties. Memory is supplied by an explicit subject-filtered frontend.

| Recorded result | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Daily correct / 1,600 | 1,506 | 1,505 | 1,506 |
| Daily accuracy | 94.125% | 94.0625% | 94.125% |
| Both sides correct + flipped / 200 history pairs | 200 | 200 | 200 |
| Clear-memory forced choice correct / 200 | 102 | 100 | 100 |
| Separate unknown-preference controls / 2,000 | 994 | 958 | 1,002 |

Daily accuracy uses the **daily subset n=1,600 per seed**, rather than all 2,000 test scenes.

Full tie-aware test scores are 95.30%, 95.25%, 95.30%; utility ties do not measure
hidden preferences. The history control counts pairs, while clear-memory scores
only the original side. The release includes 6,000 per-decision records; complete
pair-level prediction journals for the 200 history pairs were not retained.

## What this demonstrates — and what remains open

- Supplied history can affect learned numerical choices; clearing it removes that signal.
- Deterministic BFS and abstract execution supply navigation and task completion.
- Autonomous learned memory writing/forgetting, real perception and robot motor control remain unverified.
- No independent numerical candidate-reversal threshold was preregistered for the household run; report measured deviations rather than claiming strict invariance.
- G0/G1a CPU examples are limited diagnostics; later acceptance gates remain unverified.
- The Qwen 4B teacher / 1.85B student appendix loaded 272 training and 64 validation examples, scoring 42/64 teacher agreement. It has no independent test and does not inherit 308M results.

## Repository guide

| Path | Purpose |
|---|---|
| `object_memory/`, `examples/` | Low-resource explicit-memory example and controls |
| `life_choice/` | Numerical environment, 308M model and optional CUDA trainer |
| `data/`, `protocols/` | Frozen synthetic arrays and recorded protocol |
| `results/` | Original selected metrics, logs, failures and action audits |
| `scripts/` | Data freezing, record audit and optional new-checkpoint decision export |
| `docs/` | Reproduction boundaries, result interpretation and source provenance |

The CPU release checks passed on Python 3.12.14 with NumPy 2.3.5. Six GPU fixtures
were skipped because Torch/CUDA were unavailable. GPU training and the portable
checkpoint exporter were not rerun for packaging. Historical GPU dependencies
are documented, not fully locked. No large weights, private conversations, personal
application documents, credentials or company assets are included.

Own code: [MIT](LICENSE). Synthetic data and numerical records: [separate scope notice](DATA_LICENSE.md).
Dependencies retain [their own licenses](THIRD_PARTY_NOTICES.md).
