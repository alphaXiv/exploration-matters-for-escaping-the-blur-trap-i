# ExploreGS controlled reproduction

This repository is an autonomous, controlled reproduction of the core mechanisms
in *Exploration Matters for Escaping the Blur Trap in 3D Gaussian Splatting*
(arXiv:2607.17965).

The paper's official repository did not yet contain implementation code on
2026-07-21, so this project implements a compact differentiable Gaussian renderer
to test the two causal claims and their proposed interventions:

1. screen-space projection gradients are orthogonal to the viewing ray;
2. front-to-back alpha compositing attenuates gradients for later primitives;
3. random 3D seeding improves a weak-baseline far-side reconstruction task;
4. gradient-independent random splitting improves an occluded-detail task.

Each Kubernetes run uses two 8-GPU pods. Sixteen independent trials are gathered
with `torch.distributed`, and rank zero prints a compact final evidence block to
the run log. Experiment children vary only `config.json`; the command remains
`bash run.sh` throughout the tree.

## Publication package

The completed campaign provides 46 successful Kubernetes runs with terminal
measurement evidence. The controlled mechanisms reproduce strongly, while the
official-Graphdeco real-scene approximation provides modest, seed-sensitive
directional support; the overall verdict is **partially reproduced**.

- [Detailed reproduction report](reproduction/report.md)
- [Self-contained marimo notebook](reproduction/exploration_blur_trap.py)
- [Open the notebook in Molab](https://molab.marimo.io/github/alphaXiv/exploration-matters-for-escaping-the-blur-trap-i/blob/main/reproduction/exploration_blur_trap.py)

The report contains five distinct evidence figures covering the headline
result, experiment dynamics, seed robustness, efficiency/runtime, and a
controlled splitting diagnostic. Cancelled and failed attempts are excluded.
