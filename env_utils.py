from typing import Callable, Iterable

import gym

assert gym.__version__ == "0.21.0"
import numpy as np
import sys

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_robosuite import EnvRobosuite


from data_utils import OBS_KEYS


def make_robomimic_envs(
    hdf5_path: str,
    context_length: int,
    parallel_envs: int,
    target_returns: Iterable[float],
    render: bool,
    async_envs: bool,
    max_rollout_length: int,
) -> gym.Env:
    assert parallel_envs == len(target_returns)
    env_fns = [
        _make_env_fn(
            hdf5_path=hdf5_path,
            context_length=context_length,
            target_return=rtg,
            render=num == 0 and render,
            max_rollout_length=max_rollout_length,
        )
        for num, rtg in zip(range(parallel_envs), target_returns)
    ]
    if async_envs:
        envs = AsyncParallelEnvs(env_fns)
    else:
        envs = ParallelEnvs(env_fns)
    return envs


def _make_env_fn(
    hdf5_path: str,
    context_length: int,
    target_return: float,
    render=False,
    max_rollout_length=500,
) -> Callable:
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)

    def _make_env():
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=render,
            render_offscreen=False,
            use_image_obs=False,
        )
        env = RobomimicRCGymWrapper(
            env=env,
            target_return=target_return,
            context_length=context_length,
            max_timesteps=max_rollout_length,
        )
        return env

    return _make_env


class RobomimicRCGymWrapper(gym.Env):
    def __init__(
        self,
        env: EnvRobosuite,
        context_length: int,
        target_return: float,
        max_timesteps: int = 500,
    ):
        super().__init__()
        self.env = env
        self.max_timesteps = max_timesteps
        self.target_return = target_return
        self.context_length = context_length
        action_low, action_high = self.env.env.action_spec
        self.action_space = gym.spaces.Box(low=action_low, high=action_high)
        example_state = self.reset()
        self.reset_stats()
        self.observation_space = gym.spaces.Box(
            low=np.full_like(example_state, -float("inf")),
            high=np.full_like(example_state, float("inf")),
        )

    def reset_stats(self):
        self.return_history = []
        self.success_history = []

    def reset(self):
        obs = self.make_obs(self.env.reset())
        self._return = 0.0
        self.timestep = 0
        self.states = [obs]
        self.actions = []
        self.returns = [0.0]
        return obs

    def render(self, *args, **kwargs):
        self.env.render()

    def make_obs(self, dict_):
        obs = []
        for k in OBS_KEYS:
            obs.append(dict_[k])
        obs = np.concatenate(obs)
        return obs

    def step(self, action):
        self.timestep += 1
        obs_dict, rew, done, info = self.env.step(action)
        obs = self.make_obs(obs_dict)
        info["success"] = self.env.is_success()["task"]
        done = done or info["success"] or self.timestep >= self.max_timesteps
        if info["success"]:
            # semi-sparse reward (manually written into
            # offline robomimic data during training)
            rew = max(500.0 - self.timestep, 0.0)
        self._return += rew
        if done:
            self.return_history.append(self._return)
            self.success_history.append(info["success"])
        self.states.append(obs)
        self.returns.append(self._return)
        self.actions.append(action)
        return obs, rew, done, info

    def sequence(self):
        states = np.stack(self.states[-self.context_length :], axis=0)
        blank_action = np.zeros((1, self.action_space.shape[0]))
        if self.actions and self.context_length > 1:
            real_actions = np.stack(self.actions[-self.context_length + 1 :], axis=0)
            actions = np.concatenate((blank_action, real_actions), axis=0)
        else:
            actions = blank_action
        rtgs = (
            self.target_return
            - np.array(self.returns[-self.context_length :])[:, np.newaxis]
        )
        seq = np.concatenate((states, actions, rtgs), axis=-1)
        return seq


"""
Below is modified gym code with features from new version hacked 
ontop of the older version of gym (0.21.0).
"""


class ParallelEnvs(gym.vector.SyncVectorEnv):
    def __init__(self, env_fns):
        super().__init__(
            env_fns,
            observation_space=None,
            action_space=None,
            copy=True,
        )

    @property
    @property
    def return_history(self):
        return [e.return_history for e in self.envs]

    @property
    def success_history(self):
        return [e.success_history for e in self.envs]

    def reset_stats(self):
        for e in self.envs:
            e.reset_stats()

    def render(self, *args, **kwargs):
        return self.envs[0].render(*args, **kwargs)

    def sequence(self):
        return [e.sequence() for e in self.envs]


from gym.error import (
    AlreadyPendingCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
    NoAsyncCallError,
)
from gym.vector.async_vector_env import AsyncState


class AsyncParallelEnvs(gym.vector.AsyncVectorEnv):
    def __init__(self, env_fns):
        super().__init__(
            env_fns,
            observation_space=None,
            action_space=None,
            copy=False,
            shared_memory=False,
            worker=_worker,
        )

    def call_async(self, name: str, *args, **kwargs):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_STEP

    def call_wait(self, timeout) -> list:
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def _np_p(self, p):
        self.call_async(p)
        result = self.call_wait(20)
        return np.array(result)

    @property
    def return_history(self):
        self.call_async("return_history")
        return self.call_wait(20)

    @property
    def success_history(self):
        self.call_async("success_history")
        return self.call_wait(20)

    def reset_stats(self):
        self.call_async("reset_stats")
        self.call_wait(20)

    def sequence(self):
        self.call_async("sequence")
        seqs = self.call_wait(20)
        return seqs


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                pipe.send((observation, True))

            elif command == "step":
                (
                    observation,
                    reward,
                    done,
                    info,
                ) = env.step(data)
                if done:
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_observation_space":
                pipe.send((data == env.observation_space, True))
            elif command == "_check_spaces":
                pipe.send(
                    (
                        (data[0] == env.observation_space, data[1] == env.action_space),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
