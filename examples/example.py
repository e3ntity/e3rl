import torch

from e3rl.algorithms import *
from e3rl.env.gym_env import GymEnv
from e3rl.runners.runner import Runner
from hyperparams import hyperparams


ALGORITHM = DPPO
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TASK = "BipedalWalker-v3"


def main():
    hp = hyperparams[ALGORITHM.__name__][TASK]

    env = GymEnv(name=TASK, device=DEVICE, draw=True, **hp["env_kwargs"])
    agent = ALGORITHM(env, benchmark=True, device=DEVICE, **hp["agent_kwargs"])
    runner = Runner(env, agent, device=DEVICE, **hp["runner_kwargs"])
    runner._learn_cb = [Runner._log]

    runner.learn(5000)


if __name__ == "__main__":
    main()
