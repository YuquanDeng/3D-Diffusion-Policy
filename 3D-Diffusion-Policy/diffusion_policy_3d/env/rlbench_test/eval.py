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


def load_agent(
    peract_official=False,
    model_path=None,
    peract_model_dir=None,
    device=0,
    mohits_model=False,
    eval_log_dir="",
    use_input_place_with_mean=False,
):
    """
    Load agent from model_path or peract_model_dir for testing. A single
        function that support both RVT and official Peract model.
    :param model_path: path to the model for RVT agent, example:
        'runs/rvt/model_14.pth'
    :param peract_official: whether to use official Peract model or not.
    :param peract_model_dir: path to the model for official Peract model,
        example: 'runs/peract/2021-05-20_15-00-00/0'. This folder should
        contain a file name QAttentionAgent_layer0.pt
    :param device: device to run the agent on
    :param mohits_model: whether to use Mohit's model or not, only useful if
        peract_official
    :param eval_log_dir: path to the log directory for evaluation, only useful
        if peract_official is False
    :param use_input_place_with_mean: whether to use place_with_mean in rvt
        config, Ideally the deafault should be True, but for backward
        comaptibility and avoiding unncessary bugs is kept at False.
        When True, the agent will use place_with_mean defined in config
        Otherwise, it will use True value for place_with_mean.
        This is only useful if peract_official is False
    """
    device = f"cuda:{device}"

    if not peract_official:

        assert model_path is not None
        assert peract_model_dir is None
        assert not mohits_model

        model_folder = os.path.join(os.path.dirname(model_path))
        exp_cfg_path = os.path.join(model_folder, "exp_cfg.yaml")
        mvt_cfg_path = os.path.join(model_folder, "mvt_cfg.yaml")

        # load exp_cfg
        exp_cfg = default_exp_cfg.get_cfg_defaults()
        exp_cfg.merge_from_file(exp_cfg_path)

        # WARNING NOTE: a temporary hack to use place_with_mean in evaluation
        # for RLBench tasks
        if not use_input_place_with_mean:
            exp_cfg.peract2.place_with_mean = True
        exp_cfg.freeze()

        mvt_cfg_path = os.path.join(model_folder, "mvt_cfg.yaml")
        mvt_cfg = get_mvt_cfg(mvt_cfg_path, "")

        assert exp_cfg.agent == "our", "Only support our agent if peract_official is False"

        agent = get_rvt_agent(
            exp_cfg=exp_cfg,
            mvt_cfg=mvt_cfg,
            device=device,
            training=False,
            ddp=False,
            log_dir=eval_log_dir,
        )
        load_agent_state(model_path, agent)

    else:
        assert peract_model_dir is not None
        assert model_path is None
        assert eval_log_dir == ""
        assert use_input_place_with_mean

        if mohits_model:
            # these checks are put to make sure that the models is correct
            # can be disable when you know what you are doing
            assert "seed0" in peract_model_dir
            assert "weights" in peract_model_dir
            assert "600000" in peract_model_dir

            model_folder = os.path.join(os.path.abspath(peract_model_dir), "..", "..")
            train_cfg_path = os.path.join(model_folder, "config.yaml")
            pe_fix = False
            real = False
            real_data_cfg = None
        else:
            # these checks are put to make sure that the models is correct
            # can be disable when you know what you are doing
            assert not "seed0" in peract_model_dir
            assert not "weights" in peract_model_dir
            assert not "600000" in peract_model_dir

            model_folder = os.path.join(os.path.abspath(peract_model_dir), "..")
            exp_cfg_path = os.path.join(model_folder, "exp_cfg.yaml")

            # load exp_cfg
            exp_cfg = default_exp_cfg.get_cfg_defaults()
            exp_cfg.merge_from_file(exp_cfg_path)

            train_cfg_path = os.path.join(os.path.dirname(__file__), exp_cfg.peract_official.cfg_path)
            pe_fix=exp_cfg.peract_official.pe_fix
            real = exp_cfg.dataset == "real"
            real_data_cfg = exp_cfg.real_data

        agent = get_official_peract(
            cfg_path=train_cfg_path,
            training=False,
            device=device,
            bs=1,
            pe_fix=pe_fix,
            real=real,
            real_data_cfg=real_data_cfg,
        )

        agent.load_weights(peract_model_dir)
        agent.eval()

    print(f"Agent Information: {agent}")
    return agent


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
    # replay_ground_truth=False,
    replay_ground_truth=True,
    device=0,
    headless=True,
    logging=False,
    log_dir=None,
    save_video=False,
    verbose=True,
    use_mask=False,
    qwen_path=None,
    save_mask=False,
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

    # if logging:
    #     assert log_dir is not None

    #     # create metric saving writer
    #     csv_file = "eval_results.csv"
    #     if not os.path.exists(os.path.join(log_dir, csv_file)):
    #         with open(os.path.join(log_dir, csv_file), "w") as csv_fp:
    #             fieldnames = ["task", "success rate", "length", "total_transitions"]
    #             csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
    #             csv_writer.writeheader()

    # evaluate agent
    # from utils.rollout_generator_mask import MaskRolloutGenerator
    # if use_mask:
    #     assert(qwen_path != None)
    #     rollout_generator = MaskRolloutGenerator(
    #         tasks=tasks,
    #         swap_task_every=eval_episodes,
    #         env_device=device, 
    #         log_dir=log_dir,
    #         # VLM cfg
    #         qwen_path=qwen_path,
    #         save_mask=save_mask,
    #     )
    # else:
    #     rollout_generator = RolloutGenerator(device)
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
        # if logging:
            # writer csv first
            # with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
            #     fieldnames = ["task", "success rate", "length", "total_transitions"]
            #     csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            #     csv_results = {"task": task_name}
            #     for s in summaries:
            #         if s.name == "eval_envs/return":
            #             csv_results["success rate"] = s.value
            #         elif s.name == "eval_envs/length":
            #             csv_results["length"] = s.value
            #         elif s.name == "eval_envs/total_transitions":
            #             csv_results["total_transitions"] = s.value
            #         if "eval" in s.name:
            #             s.name = "%s/%s" % (s.name, task_name)
            #     csv_writer.writerow(csv_results)
        # else:
        #     for s in summaries:
        #         if "eval" in s.name:
        #             s.name = "%s/%s" % (s.name, task_name)

        # if len(summaries) > 0:
        #     task_score = [
        #         s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
        #     ][0]
        # else:
        #     task_score = "unknown"

        # print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")

        # scores.append(task_score)

        # if save_video:
        #     import cv2
        #     import shutil

        #     video_image_folder = "./tmp"
        #     record_fps = 25
        #     record_folder = os.path.join(log_dir, "videos")
        #     os.makedirs(record_folder, exist_ok=True)
        #     video_success_cnt = 0
        #     video_fail_cnt = 0
        #     video_cnt = 0
        #     for summary in summaries:
        #         if isinstance(summary, VideoSummary):
        #             video = deepcopy(summary.value)
        #             video = np.transpose(video, (0, 2, 3, 1))
        #             video = video[:, :, :, ::-1]
        #             if task_rewards[video_cnt] > 99:
        #                 video_path = os.path.join(
        #                     record_folder,
        #                     f"{task_name}_success_{video_success_cnt}.mp4",
        #                 )
        #                 video_success_cnt += 1
        #             else:
        #                 video_path = os.path.join(
        #                     record_folder, f"{task_name}_fail_{video_fail_cnt}.mp4"
        #                 )
        #                 video_fail_cnt += 1
        #             video_cnt += 1
        #             os.makedirs(video_image_folder, exist_ok=True)
        #             for idx in range(len(video) - 10):
        #                 cv2.imwrite(
        #                     os.path.join(video_image_folder, f"{idx}.png"), video[idx]
        #                 )
        #             images_path = os.path.join(video_image_folder, r"%d.png")
        #             os.system(
        #                 "ffmpeg -i {} -vf palettegen palette.png -hide_banner -loglevel error".format(
        #                     images_path
        #                 )
        #             )
        #             os.system(
        #                 "ffmpeg -framerate {} -i {} -i palette.png -lavfi paletteuse {} -hide_banner -loglevel error".format(
        #                     record_fps, images_path, video_path
        #                 )
        #             )
        #             os.remove("palette.png")
        #             shutil.rmtree(video_image_folder)

    eval_env.shutdown()

    # if logging:
    #     csv_fp.close()

    # set agent to back train mode
    # agent.train()
    return scores


def get_model_index(filename):
    """
    :param filenam: path of file of format /.../model_idx.pth
    :return: idx or None
    """
    if len(filename) >= 9 and filename[-4:] == ".pth":
        try:
            index = int(filename[:-4].split("_")[-1])
        except:
            index = None
    else:
        index = None
    return index


def _eval(args):

    # model_paths = []
    # if not (args.peract_official):
    #     if args.mode == "final":
    #         model_paths.append(os.path.join(args.model_folder, "model.pth"))
    #     elif args.mode == "all":
    #         assert (
    #             False
    #         ), "Evaluation model 'all' can cause some bugs for now, please use mode 'single' or 'final' "
    #         epochs = []
    #         filenames = os.listdir(args.model_folder)
    #         for filename in filenames:
    #             epoch = get_model_index(filename)
    #             if epoch is not None:
    #                 epochs.append(epoch)
    #         epochs.sort()
    #         for epoch in epochs:
    #             if (epoch + 1) % 5 == 0:
    #                 model_paths.append(
    #                     os.path.join(args.model_folder, f"model_{epoch}.pth")
    #                 )
    #     elif args.mode == "single":
    #         assert args.model_name is not None
    #         model_paths.append(os.path.join(args.model_folder, args.model_name))
    #     else:
    #         raise NotImplementedError
    # else:
    #     model_paths.append(None)

    # skipping evaluated models
    # if args.skip:
    #     """
    #     to_skip: {
    #         0: {'light_bulb_in': False, .....}
    #         1: {'light_bulb_in': False, .....}
    #         .
    #         .
    #     }
    #     """
    #     to_skip = {
    #         get_model_index(x): {y: False for y in args.tasks} for x in model_paths
    #     }

    #     filenames = os.listdir(args.eval_log_dir)
    #     for filename in filenames:
    #         if not filename.startswith("events.out.tfevents."):
    #             continue
    #         summ = summary_iterator(f"{args.eval_log_dir}/{filename}")
    #         # skipping the time log of the summary
    #         try:
    #             next(summ)
    #         except:
    #             # moving to the next file
    #             continue
    #         for cur_summ in summ:
    #             cur_task = cur_summ.summary.value[0].tag[5:]
    #             cur_step = cur_summ.step
    #             if cur_step in to_skip:
    #                 to_skip[cur_step][cur_task] = True

    # tb = TensorboardManager(args.eval_log_dir)
    # for model_path in model_paths:
    #     tasks_to_eval = deepcopy(args.tasks)

    #     if args.peract_official:
    #         model_idx = 0
    #     elif args.mode == "final" or args.mode == "single":
    #         model_idx = get_model_index(model_path)
    #         if model_idx is None:
    #             model_idx = 0
    #     else:
    #         model_idx = get_model_index(model_path)

    #     if args.skip:
    #         for _task in args.tasks:
    #             if to_skip[model_idx][_task]:
    #                 tasks_to_eval.remove(_task)

    #         if len(tasks_to_eval) == 0:
    #             print(f"Skipping model_idx={model_idx} for args.tasks={args.tasks}")
    #             continue

    #     if not args.peract_official:
    #         if not args.use_mask:
    #             agent = load_agent(
    #                 model_path=model_path,
    #                 eval_log_dir=f"{args.eval_log_dir}/eval_run",
    #                 device=args.device,
    #                 use_input_place_with_mean=args.use_input_place_with_mean,
    #             )

    #             agent_eval_log_dir = os.path.join(
    #                 args.eval_log_dir, os.path.basename(model_path).split(".")[0]
    #             )
    #         else:
    #             # mask rvt2 
    #             from models.mask_rvt_agent import load_mask_rvt_agent
    #             agent = load_mask_rvt_agent(
    #                 model_path=model_path,
    #                 eval_log_dir=f"{args.eval_log_dir}/eval_run",
    #                 device=args.device,
    #                 use_input_place_with_mean=args.use_input_place_with_mean,
    #                 # VLM cfgs
    #                 same_lang_goal=args.same_lang_goal
    #             )

    #             agent_eval_log_dir = os.path.join(
    #                 args.eval_log_dir, os.path.basename(model_path).split(".")[0]
    #             )                
    #     else:
    #         agent = load_agent(
    #             peract_official=args.peract_official,
    #             peract_model_dir=args.peract_model_dir,
    #             device=args.device,
    #             mohits_model=args.mohits_model,
    #         )
    #         agent_eval_log_dir = os.path.join(args.eval_log_dir, "final")

        # os.makedirs(agent_eval_log_dir, exist_ok=True)
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
    #     print(f"model {model_path}, scores {scores}")
    #     task_scores = {}
    #     for i in range(len(tasks_to_eval)):
    #         task_scores[tasks_to_eval[i]] = scores[i]

    #     print("save ", task_scores)
    #     tb.update("eval", model_idx, task_scores)
    #     tb.writer.flush()

    # tb.close()


if __name__ == "__main__":
    parser = get_eval_parser()

    args = parser.parse_args()

    if args.log_name is None:
        args.log_name = get_time_stamp()

    # if not (args.peract_official):
    #     args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)
    # else:
    #     args.eval_log_dir = os.path.join(args.peract_model_dir, "eval", args.log_name)
    #     assert args.model_name == 'QAttentionAgent_layer0.pt', args.model_name

    # os.makedirs(args.eval_log_dir, exist_ok=True)

    # save the arguments for future reference
    # with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
    #     yaml.dump(args.__dict__, fp)

    _eval(args)