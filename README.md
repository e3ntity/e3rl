# e3rl

Fast and simple implementation of RL algorithms, designed to run fully on GPU.

Currently, the following algorithms are implemented:
- Distributed Distributional DDPG (D4PG)
- Deep Deterministic Policy Gradient (DDPG)
- Distributional PPO (DPPO)
- Distributional Soft Actor Critic (DSAC)
- Proximal Policy Optimization (PPO)
- Soft Actor Critic (SAC)
- Twin Delayed DDPG (TD3)

**Maintainer**: Lukas Schneider <br/>
**Contact**: Lukas Schneider (schneider.lukas@protonmail.com)

This project was originally forked from [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (actively maintained).
See [CONTRIBUTORS.md](CONTRIBUTORS.md) for upstream attribution.

## Installation

To install the package, run the following command in the root directory of the repository:

```bash
pip install -e .
```

Optional extras are available for additional functionality:

```bash
pip install -e ".[gym,logging,export]"  # gymnasium, tensorboard/wandb, ONNX export
pip install -e ".[dev]"                 # linters and pre-commit
pip install -e ".[docs]"                # sphinx
```

## Device support

e3rl runs on CUDA, Apple Silicon (MPS), and CPU. Pass `device="cuda:0"`, `device="mps"`, or `device="cpu"` to envs, agents, and runners. The helper `e3rl.utils.resolve_device()` auto-selects the best available backend in the order **CUDA → MPS → CPU** and is used by the bundled examples.

## Examples

Examples can be run from the `examples/` directory. The example directory also includes hyperparameters tuned for some gym environments, which are loaded automatically. Videos of trained policies are periodically saved to `videos/`.

```bash
python examples/example.py
```

The clips below show the first and last recorded episodes from a single training run of `examples/example.py` (DPPO on `BipedalWalker-v3`), illustrating convergence from a randomly initialized policy to a walking gait:

| Untrained | After 5000 iterations |
| :---: | :---: |
| ![Untrained](assets/example-first-episode.gif) | ![Trained](assets/example-last-episode.gif) |

## Tests

```bash
cd tests/ && python -m unittest
```

## Documentation

```bash
pip install -e ".[docs]"
sphinx-apidoc -o docs/source . ./examples
cd docs/ && make html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{schneider2023learning,
  archivePrefix={arXiv},
  author={Lukas Schneider and Jonas Frey and Takahiro Miki and Marco Hutter},
  eprint={2309.14246},
  primaryClass={cs.RO},
  title={Learning Risk-Aware Quadrupedal Locomotion using Distributional Reinforcement Learning},
  year={2023},
}
```

## Contributing

The project uses [`ruff`](https://github.com/astral-sh/ruff) for linting and formatting, run via `pre-commit`:

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
```
