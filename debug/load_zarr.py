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
from PIL import Image
import os

tasks = ['drill_40demo_1024.zarr', 'dumpling_new_40demo_1024.zarr', 'pour_40demo_1024.zarr', 'roll_40demo_1024.zarr']

task = tasks[0]
save_path = f"/home/nil/manipulation/3D-Diffusion-Policy/debug/real_world_demo/{task}"
if not os.path.exists(save_path):
    os.makedirs(save_path)
zarr_path = f"/home/nil/manipulation/3D-Diffusion-Policy/3D-Diffusion-Policy/data/real_robot_demo/{task}"
replay_buffer = ReplayBuffer.copy_from_path(
    zarr_path, keys=['state', 'action', 'point_cloud', 'depth', 'img'])

# for idx in range(len(replay_buffer["state"])):
#     # import pdb;pdb.set_trace()
#     img = Image.fromarray(replay_buffer["img"][idx]) # H, W, 3
#     img.save(os.path.join(save_path, f'{idx}.png'))
    
import pdb;pdb.set_trace()

