from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BasePointcloudDataset

class RLBenchPointcloudDataset(BasePointcloudDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_lang=False
            ):
        super().__init__()
        self.use_lang = use_lang
        if use_lang:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'point_cloud', 'lang_goal_embed'])
        else:
            self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=['state', 'action', 'point_cloud'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        if self.use_lang:
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:],
                'point_cloud': self.replay_buffer['point_cloud'],
                'lang_goal_embed': self.replay_buffer['lang_goal_embed']
            }
        else:
            data = {
                'action': self.replay_buffer['action'],
                'agent_pos': self.replay_buffer['state'][...,:],
                'point_cloud': self.replay_buffer['point_cloud'],
            }           
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        # normalizer['point_cloud_robot'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 3)
        lang_goal_embed = None
        if self.use_lang:
            lang_goal_embed = sample['lang_goal_embed'][:,].astype(np.float32) # (T, lang_embed_dim) 
        # point_cloud_robot = sample['point_cloud_robot'][:,].astype(np.float32) # (T, 1024, 3)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 3
                # 'point_cloud_robot': point_cloud_robot, # T, 1024, 3
                'agent_pos': agent_pos, # T, D_pos
                'lang_goal_embed': lang_goal_embed # None or (T, lang_embed_dim)
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
