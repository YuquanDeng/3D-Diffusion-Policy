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
from termcolor import cprint

class DexArtPointcloudDataset(BasePointcloudDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'imagin_robot', 'img'])
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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            # 'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['imagin_robot'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 512, 3)
        imagin_robot = sample['imagin_robot'][:,].astype(np.float32) # (T, 96, 7)
        
        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 512, 3
                'imagin_robot': imagin_robot, # T, 96, 7
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


class DexArtPointcloudMultiTaskDataset(BasePointcloudDataset):
    """Support observation only"""
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name

        cprint("[MultiTaskDataset] used.", "yellow")
        
        self.replay_buffers = []
        self.samplers = []
        self.train_masks = []
        
        for path in zarr_path:
            replay_buffer = ReplayBuffer.copy_from_path(
                path, keys=['point_cloud', 'imagin_robot', 'img'])

            val_mask = get_val_mask(
                n_episodes=replay_buffer.n_episodes, 
                val_ratio=val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)

            sampler = SequenceSampler(
                replay_buffer=replay_buffer, 
                sequence_length=horizon,
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=train_mask)
            
            
            self.replay_buffers.append(replay_buffer)
            self.samplers.append(sampler)
            self.train_masks.append(train_mask)

        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_idx_sampler(self, idx):
        # 根据idx获得对应的sampler，以及其在sampler中的idx
        # 每个sampler的长度都是不一样的
        # return sampler_idx, idx
        current_count = 0
        for i in range(len(self.samplers)):
            if current_count + len(self.samplers[i]) > idx:
                return i, idx - current_count
            current_count += len(self.samplers[i])
        raise ValueError("idx out of range")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffers[0], 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_masks[0]
            )
        val_set.train_mask = ~self.train_masks[0]
        return val_set

  
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['imagin_robot'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return sum([len(self.samplers[i]) for i in range(len(self.samplers))])

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 512, 3)
        imagin_robot = sample['imagin_robot'][:,].astype(np.float32) # (T, 96, 7)
        
        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 512, 3
                'imagin_robot': imagin_robot, # T, 96, 7
                # 'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sampler_idx, idx = self.get_idx_sampler(idx)
        sample = self.samplers[sampler_idx].sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

