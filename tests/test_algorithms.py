import unittest

import torch

from e3rl.algorithms import D4PG, DDPG, DPPO, DSAC, PPO, SAC, TD3
from e3rl.env.gym_env import GymEnv
from e3rl.modules import Network
from e3rl.runners.runner import Runner

DEVICES = ["cpu"]
if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda:0")


def _per_device(test_method):
    def wrapper(self):
        for device in DEVICES:
            with self.subTest(device=device):
                self.device = device
                test_method(self)

    wrapper.__name__ = test_method.__name__
    return wrapper


def per_device(cls):
    for name, attr in list(cls.__dict__.items()):
        if name.startswith("test_") and callable(attr):
            setattr(cls, name, _per_device(attr))
    return cls


class AlgorithmTestCaseMixin:
    algorithm_class = None
    device = "cpu"

    def _make_env(self, params={}):
        my_params = dict(name="LunarLanderContinuous-v3", device=self.device, environment_count=4)
        my_params.update(params)

        return GymEnv(**my_params)

    def _make_agent(self, env, agent_params={}):
        return self.algorithm_class(env, device=self.device, **agent_params)

    def _make_runner(self, env, agent, runner_params={}):
        if not runner_params or "num_steps_per_env" not in runner_params:
            runner_params["num_steps_per_env"] = 6

        return Runner(env, agent, device=self.device, **runner_params)

    def _learn(self, env, agent, runner_params={}):
        runner = self._make_runner(env, agent, runner_params)
        runner.learn(10)

    def test_default(self):
        env = self._make_env()
        agent = self._make_agent(env)

        self._learn(env, agent)

    def test_single_env_single_step(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env)

        self._learn(env, agent, dict(num_steps_per_env=1))


class RecurrentAlgorithmTestCaseMixin(AlgorithmTestCaseMixin):
    def test_recurrent(self):
        env = self._make_env()
        agent = self._make_agent(env, dict(recurrent=True))

        self._learn(env, agent)

    def test_single_env_single_step_recurrent(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env, dict(recurrent=True))

        self._learn(env, agent, dict(num_steps_per_env=1))


@per_device
class D4PGTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = D4PG


@per_device
class DDPGTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = DDPG


iqn_params = dict(
    critic_network=DPPO.network_iqn,
    iqn_action_samples=8,
    iqn_embedding_size=16,
    iqn_feature_layers=2,
    iqn_value_samples=4,
    value_loss=DPPO.value_loss_energy,
)

qrdqn_params = dict(
    critic_network=DPPO.network_qrdqn,
    qrdqn_quantile_count=16,
    value_loss=DPPO.value_loss_l1,
)


@per_device
class DPPOTest(RecurrentAlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = DPPO

    def test_qrdqn(self):
        env = self._make_env()
        agent = self._make_agent(env, qrdqn_params)

        self._learn(env, agent)

    def test_qrdqn_sing_env_single_step(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env, qrdqn_params)

        self._learn(env, agent, dict(num_steps_per_env=1))

    def test_qrdqn_energy_loss(self):
        my_agent_params = qrdqn_params.copy()
        my_agent_params["value_loss"] = DPPO.value_loss_energy

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)

    def test_qrdqn_huber_loss(self):
        my_agent_params = qrdqn_params.copy()
        my_agent_params["value_loss"] = DPPO.value_loss_huber

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)

    def test_qrdqn_transformer(self):
        my_agent_params = qrdqn_params.copy()
        my_agent_params["recurrent"] = True
        my_agent_params["critic_recurrent_layers"] = 2
        my_agent_params["critic_recurrent_module"] = Network.recurrent_module_transformer
        my_agent_params["critic_recurrent_tf_context_length"] = 8
        my_agent_params["critic_recurrent_tf_head_count"] = 2

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)

    def test_iqn(self):
        env = self._make_env()
        agent = self._make_agent(env, iqn_params)

        self._learn(env, agent)

    def test_iqn_single_step_single_env(self):
        env = self._make_env(dict(environment_count=1))
        agent = self._make_agent(env, iqn_params)

        self._learn(env, agent, dict(num_steps_per_env=1))

    def test_iqn_recurrent(self):
        my_agent_params = iqn_params.copy()
        my_agent_params["recurrent"] = True

        env = self._make_env()
        agent = self._make_agent(env, my_agent_params)

        self._learn(env, agent)


@per_device
class DSACTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = DSAC


@per_device
class PPOTest(RecurrentAlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = PPO


@per_device
class SACTest(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = SAC


@per_device
class TD3Test(AlgorithmTestCaseMixin, unittest.TestCase):
    algorithm_class = TD3


if __name__ == "__main__":
    unittest.main()
