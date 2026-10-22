# Third-party dependencies and references

These are external dependencies/references, not vendored components. Installations
must retain the upstream distribution's complete notices. This file does not
relicense third-party code, model outputs or assets.

| Item | Upstream terms/reference | Included here |
|---|---|---|
| Python standard library | [PSF/Python licenses](https://docs.python.org/3.12/license.html) | Imports only; no runtime |
| NumPy | [BSD-3-Clause and distribution notices](https://numpy.org/doc/2.3/license.html) | Imports only; no wheels |
| PyTorch v2.3 family | [Upstream license and notices](https://github.com/pytorch/pytorch/blob/v2.3.0/LICENSE) | Optional GPU imports only |
| NVIDIA CUDA / historical vendor Torch build | Vendor-specific terms apply | Historical environment metadata only |
| Qwen3-4B-Instruct-2507 | [Official model card, Apache-2.0](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Exploratory model identity and aggregate measurements only |

The community Q4_K_M conversion and teacher outputs have not been independently
cleared for redistribution. They, teacher automation and all model checkpoints
are excluded. No character fan art is included or covered by MIT. No affiliation
with model vendors, fictional-character rights holders or OpenAI is implied.
