if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
from diffusion_policy_3d.policy.diffusion_unet_hybrid_pointcloud_policy import DiffusionUnetHybridPointcloudPolicy
from diffusion_policy_3d.dataset.base_dataset import BasePointcloudDataset
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.json_logger import JsonLogger
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

from diffusion_policy_3d.workspace.train_diffusion_unet_hybrid_pointcloud_workspace import TrainDiffusionUnetHybridPointcloudWorkspace

class TrainDiffusionRLBenchPointcloudWorkspace(TrainDiffusionUnetHybridPointcloudWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
    
    def run(self):
        raise NotImplementedError("run() function in TrainDiffusionRLBenchPointcloudWorkspace.")
        return super().run()