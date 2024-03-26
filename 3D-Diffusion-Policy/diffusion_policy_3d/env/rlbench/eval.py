# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import torch
import numpy as np
import os
import yaml
import csv

from argparse import Namespace
from multiprocessing import Value, Manager
from tensorflow.python.summary.summary_iterator import summary_iterator
from copy import deepcopy

from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rvt.utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)

from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.log_writer import LogWriter
from yarr.agents.agent import VideoSummary

import mvt.config as default_mvt_cfg
import rvt.models.rvt_agent as peract2
import rvt.config as default_exp_cfg

from rvt.utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
from rvt.utils.ortho_cam.new_cameras_rlbench_env import NewCameraRLBenchEnv as OrthoCameraRLBenchEnv

from rvt.libs.peract.helpers import utils
from rvt.utils.peract_utils import IMAGE_SIZE, get_official_peract
from rvt.models.per_io import PerceiverIO
from rvt.models.peract import PerceiverActorAgent
from rvt.utils.rvt_utils import (
    TensorboardManager,
    get_eval_parser,
    RLBENCH_TASKS,
    get_time_stamp,
)
from rvt.utils.rvt_utils import load_agent_state
from rvt.train import get_mvt_cfg, get_rvt_agent
from mvt import MVT

from omegaconf import OmegaConf


@torch.no_grad()
def eval(
    agent,
    tasks,
    cameras,
    eval_datafolder,
    start_episode=0,
    eval_episodes=25,
    episode_length=25,
    record_every_n=-1,
    replay_ground_truth=False,
    # replay_ground_truth=True,
    device=0,
    headless=True,
    # logging=False,
    # log_dir=None,
    # save_video=False,
    # verbose=True,
    # use_mask=False,
    # qwen_path=None,
    # save_mask=False,
):

    # agent.eval()
    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(cameras, camera_resolution, method_name="")

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))

    if not logging:
        record_every_n = -1

    if save_video:
        record_every_n = 1

    if set(cameras) == set(["front", "left_shoulder", "right_shoulder", "wrist"]):
        RLBenchEnv = CustomMultiTaskRLBenchEnv
    elif set(cameras) == set(["front", "left", "right", "back", "top"]):
        RLBenchEnv = OrthoCameraRLBenchEnv
    else:
        assert False, "Unknown camera set: {}".format(cameras)

    eval_env = RLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=eval_datafolder,
        episode_length=episode_length,
        headless=headless,
        swap_task_every=eval_episodes,
        include_lang_goal_in_obs=True,
        time_in_state=True,
        record_every_n=record_every_n,
    )

    eval_env.eval = True

    device = f"cuda:{device}"

    rollout_generator = RolloutGenerator(device)
    stats_accumulator = SimpleAccumulator(eval_video_fps=30)

    eval_env.launch()

    current_task_id = -1

    num_tasks = len(tasks)
    step_signal = Value("i", -1)

    scores = []
    for task_id in range(num_tasks):
        task_rewards = []
        for ep in range(start_episode, start_episode + eval_episodes):
            episode_rollout = []
            generator = rollout_generator.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=episode_length,
                timesteps=1,
                eval=True,
                eval_demo_seed=ep,
                record_enabled=False,
                replay_ground_truth=replay_ground_truth,
            )
            try:
                for replay_transition in generator:
                    episode_rollout.append(replay_transition)
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            if verbose:
                print(
                    f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
                )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]

    eval_env.shutdown()
    return scores

if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()

    tasks_to_eval = ["light_bulb_in"]
    scores = eval(
        # agent=agent,
        agent=None,
        tasks=tasks_to_eval,
        cameras=args.cameras,
        eval_datafolder=args.eval_datafolder,
        start_episode=args.start_episode,
        eval_episodes=args.eval_episodes,
        episode_length=args.episode_length,
        record_every_n=args.record_every_n,
        replay_ground_truth=args.ground_truth,
        device=args.device,
        headless=args.headless,
        logging=True,
        # log_dir=agent_eval_log_dir,
        log_dir=None,
        save_video=args.save_video,
        verbose=True,
        use_mask=args.use_mask,
        qwen_path=args.qwen_path,
        save_mask=args.save_mask,
    )