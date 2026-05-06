from hyperparams import hyperparams

from e3rl.algorithms import *
from e3rl.env.gym_env import GymEnv
from e3rl.runners.runner import Runner
from e3rl.utils import resolve_device

ALGORITHM = DPPO
DEVICE = resolve_device()
TASK = "BipedalWalker-v3"


def main():
    hp = hyperparams[ALGORITHM.__name__][TASK]

    print(f"Algorithm: {ALGORITHM.__name__}")
    print(f"Task:      {TASK}")
    print(f"Device:    {DEVICE}")

    env = GymEnv(name=TASK, device=DEVICE, draw=True, **hp["env_kwargs"])
    agent = ALGORITHM(env, benchmark=True, device=DEVICE, **hp["agent_kwargs"])
    runner = Runner(env, agent, device=DEVICE, **hp["runner_kwargs"])
    runner._learn_cb = [Runner._log]

    try:
        runner.learn(5000)
    finally:
        env.close()


if __name__ == "__main__":
    main()
