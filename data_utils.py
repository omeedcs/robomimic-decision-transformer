from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.dataset import SequenceDataset
from robomimic.config import config_factory

config = config_factory(algo_name="bc")
ObsUtils.initialize_obs_utils_with_config(config)
OBS_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object")
# OBS_KEYS= ('object', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_vel_lin', 'robot0_eef_vel_ang', 'robot0_gripper_qpos', 'robot0_gripper_qvel')


class RTGSequenceDataset(SequenceDataset):
    def get_item(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset
        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length,
        )
        """
        Here is our hack to get the return-to-go data by loading
        all the rewards until the end of the demo, reverse cumsumming
        and then trimming to the regular seq length.

        The robomimic data has been remade with dense rewards enabled. However,
        the issue is that the dense reward return is very uncorrelated with dataset
        quality. For example in the lift dataset "proficient human" (ph) has the lowest
        return while "machine generated" (mg) has the highest. This seems to be due
        to the sequence length mattering more than success rate or speed. We manually 
        alter the reward function to add a semi-sparse success bonus that decreases every timestep. 
        This gives a wider distribution of target RTGs for decision transformer than the
        default binary option in robomimic.
        """

        # starting from this index, find all the future reward and dones
        future_info, _ = self.get_sequence_from_demo(
            demo_id, index_in_demo, keys=["rewards", "dones"], seq_length=demo_length
        )
        for t, d in enumerate(future_info["dones"]):
            if d:
                break
        # figure out which timestep of the demonstration leads to a success
        success_step = (
            index_in_demo + t + 1
        )  # extra +1 matches rew func in RobomimicRCGymWrapper
        meta["success_step"] = np.array([success_step], dtype=np.float32)
        future_rews = future_info["rewards"].copy()
        future_rews[t] = max(500.0 - success_step, 0.0)
        future_rews[t + 1 :] = 0.0
        # reverse cumsum
        rtgs = np.ascontiguousarray(np.flip(np.cumsum(np.flip(future_rews))))[
            :, np.newaxis
        ]
        meta["rtgs"] = rtgs[: self.seq_length]
        # include full demo return, which helps us pick the expert return at test-time
        if index_in_demo == 0:
            meta["full_demonstration_return"] = rtgs[0]
        else:
            # use nan as indicator to ignore this value in expert RTG calculation
            meta["success_step"] = np.full_like(meta["success_step"], np.nan)
            meta["full_demonstration_return"] = np.full_like(rtgs[0], np.nan)
        """
        back to standard robomimic SequenceDataset
        """

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs",
        )
        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(
                meta["obs"], obs_normalization_stats=self.obs_normalization_stats
            )

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                meta["next_obs"] = ObsUtils.normalize_obs(
                    meta["next_obs"],
                    obs_normalization_stats=self.obs_normalization_stats,
                )

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                goal = ObsUtils.normalize_obs(
                    goal, obs_normalization_stats=self.obs_normalization_stats
                )
            meta["goal_obs"] = {
                k: goal[k][0] for k in goal
            }  # remove sequence dimension for goal
        return meta


def torch_move_pad(samples: List[np.ndarray], device):
    states = [torch.from_numpy(s).float() for s in samples]
    seq_lengths = (
        torch.Tensor([len(s) for s in states]).to(device).view(-1, 1, 1).long()
    )
    states = pad_sequence(states, batch_first=True, padding_value=-4.0).to(device)
    return states, seq_lengths


def robomimic_collate(samples):
    # flatten observation dicts into an array (matches RobomimicRCGymWrapper)
    states = []
    for obs_key in OBS_KEYS:
        states_obs = np.stack([s_dict["obs"][obs_key] for s_dict in samples], axis=0)
        states.append(states_obs)
    seq_states = np.concatenate(states, axis=-1)
    pad_mask = np.stack([s_dict["obs"]["pad_mask"] for s_dict in samples], axis=0)
    # ground-truth actions become supervised learning labels
    actions = np.stack([s_dict["actions"] for s_dict in samples], axis=0)
    # actions shifted by a timestep are added to the input of the Decision Transformer
    B, L, D = actions.shape
    blank_action = np.zeros((B, 1, D))
    seq_actions = np.concatenate((blank_action, actions[:, :-1].copy()), axis=1)
    seq_rtgs = np.stack([s_dict["rtgs"] for s_dict in samples], axis=0)
    # input sequence includes states, actions, and rtgs. Note rtgs are in last
    # index. Ablations use this as an easy way to zero out the return-conditioned
    # part.
    seq = np.concatenate((seq_states, seq_actions, seq_rtgs), axis=-1)
    demo_returns = np.stack(
        [s_dict["full_demonstration_return"] for s_dict in samples], axis=0
    )
    demo_steps = np.stack([s_dict["success_step"] for s_dict in samples], axis=0)
    data = dict(
        seq=torch.from_numpy(seq).float(),
        actions=torch.from_numpy(actions).float(),
        # Note flipped sign here to be consistent w/ rest of code. True == padded
        pad_mask=~torch.from_numpy(pad_mask).bool(),
        demo_returns=torch.from_numpy(demo_returns).float().squeeze(1),
        demo_steps=torch.from_numpy(demo_steps).float().squeeze(1),
    )
    return data


def load_robomimic_dset(hdf5_path: str, context_length: str) -> RTGSequenceDataset:
    dset = RTGSequenceDataset(
        hdf5_path=hdf5_path,
        obs_keys=OBS_KEYS,
        dataset_keys=("actions", "rewards", "dones"),
        load_next_obs=False,
        seq_length=context_length,
        pad_seq_length=True,
        get_pad_mask=True,
        goal_mode=None,
        hdf5_cache_mode="all",
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
    )
    return dset
