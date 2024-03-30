import wandb
import numpy as np
import torch
import collections
import tqdm
import os
# from diffusion_policy_3d.env import MetaWorldEnv
# from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
# from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_pointcloud_policy import BasePointcloudPolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
# import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

# EVALUATION
from multiprocessing import Value
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
# from rvt.utils.rlbench_planning import (
#     EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
# )

from diffusion_policy_3d.RLBench.utils.rlbench_planning import (
    EndEffectorPoseViaPlanning2 as EndEffectorPoseViaPlanning,
)

# from yarr.utils.rollout_generator import RolloutGenerator
from diffusion_policy_3d.RLBench.rollout_generator import RolloutGenerator
from yarr.utils.stat_accumulator import SimpleAccumulator
# from yarr.utils.log_writer import LogWriter
# from yarr.agents.agent import VideoSummary

# import mvt.config as default_mvt_cfg
# import rvt.models.rvt_agent as peract2
# import rvt.config as default_exp_cfg

# from rvt.utils.custom_rlbench_env import (
#     CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
# )
# from rvt.utils.ortho_cam.new_cameras_rlbench_env import NewCameraRLBenchEnv as OrthoCameraRLBenchEnv
from diffusion_policy_3d.RLBench.utils.custom_rlbench_env import (
    CustomMultiTaskRLBenchEnv2 as CustomMultiTaskRLBenchEnv,
)
# from rvt.libs.peract.helpers import utils
from diffusion_policy_3d.RLBench.rlbench_utils import create_obs_config
from diffusion_policy_3d.RLBench.utils.rvt_utils import get_eval_parser
# from rvt.utils.peract_utils import IMAGE_SIZE, get_official_peract
# from rvt.models.per_io import PerceiverIO
# from rvt.models.peract import PerceiverActorAgent
# from rvt.utils.rvt_utils import (
    # TensorboardManager,
    # get_eval_parser,
    # RLBENCH_TASKS,
    # get_time_stamp,
# )
# from rvt.utils.rvt_utils import load_agent_state
# from rvt.train import get_mvt_cfg, get_rvt_agent
# from mvt import MVT
from diffusion_policy_3d.RLBench.rlbench_utils import (
    RLBENCH_TASKS,
    IMAGE_SIZE,
)
# LOGGING
from omegaconf import OmegaConf
from yarr.agents.agent import VideoSummary
from copy import deepcopy

class RLBenchPointcloudRunner(BasePointcloudRunner):
    def __init__(self,
                 output_dir=None,
                #  eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 # RLBench configs
                 tasks_to_eval=None,
                 cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
                 eval_datafolder=None,
                 start_episode=0,
                 eval_episodes=5,
                 episode_length=25,
                 record_every_n=-1,
                 ground_truth=False,
                 headless=False,
                 save_video=False,
                 log_dir=None,
                 # representation config
                 rotation_euler=False,
                 task_bound=None,
                 rotation_resolution=None
                 ):
        super().__init__(output_dir)
        # self.task_name = task_name

        # steps_per_render = max(10 // fps, 1)

        # def env_fn(task_name):
        #     return MultiStepWrapper(
        #         SimpleVideoRecordingWrapper(
        #             MetaWorldEnv(task_name=task_name,device=device, 
        #                          use_point_crop=use_point_crop, num_points=num_points)),
        #         n_obs_steps=n_obs_steps,
        #         n_action_steps=n_action_steps,
        #         max_episode_steps=max_steps,
        #         reward_agg_method='sum',
        #     )
        # self.eval_episodes = eval_episodes
        # self.env = env_fn(self.task_name)

        # self.fps = fps
        # self.crf = crf
        # self.n_obs_steps = n_obs_steps
        # self.n_action_steps = n_action_steps
        # self.max_steps = max_steps
        # self.tqdm_interval_sec = tqdm_interval_sec

        # self.logger_util_test = logger_util.LargestKRecorder(K=3)
        # self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
        self.tasks_to_eval = tasks_to_eval
        self.cameras = cameras
        self.eval_datafolder = eval_datafolder
        self.start_episode = start_episode
        self.eval_episodes = eval_episodes
        self.episode_length = episode_length

        self.save_video = save_video
        if self.save_video:
            record_every_n = 1
        self.record_every_n = record_every_n
        self.log_dir = log_dir

        self.ground_truth = ground_truth
        self.device = device
        self.headless = headless
        self.task_name = task_name
        
        # representation config
        self.use_point_crop = use_point_crop
        self.rotation_euler = rotation_euler
        self.task_bound = task_bound
        self.num_points = num_points
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.rotation_resolution = rotation_resolution

    @torch.no_grad()
    def run(self, policy: BasePointcloudPolicy):
        agent=policy
        tasks=self.task_name
        cameras=self.cameras
        eval_datafolder=self.eval_datafolder
        start_episode=self.start_episode
        eval_episodes=self.eval_episodes
        episode_length=self.episode_length
        record_every_n=self.record_every_n
        replay_ground_truth=self.ground_truth
        device=self.device
        headless=self.headless
        verbose=True
        logging=False
        save_video=self.save_video

        # agent.eval()
        camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
        obs_config = create_obs_config(cameras, camera_resolution, method_name="")

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

        # if not logging:
        #     record_every_n = -1

        # if save_video:
        #     record_every_n = 1

        if set(cameras) == set(["front", "left_shoulder", "right_shoulder", "wrist"]):
            RLBenchEnv = CustomMultiTaskRLBenchEnv
        # elif set(cameras) == set(["front", "left", "right", "back", "top"]):
        #     RLBenchEnv = OrthoCameraRLBenchEnv
        # else:
        #     assert False, "Unknown camera set: {}".format(cameras)

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

        # device = f"cuda:{device}"

        rollout_generator = RolloutGenerator(
            env_device=device,
            use_point_crop=self.use_point_crop,
            rotation_euler = self.rotation_euler,
            task_bound=self.task_bound,
            num_points=self.num_points,
            rotation_resolution=self.rotation_resolution
            )
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
                    timesteps=self.n_obs_steps,
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

            if self.save_video:
                import cv2
                import shutil

                video_image_folder = "./tmp"
                record_fps = 25
                record_folder = os.path.join(self.log_dir, "videos")
                os.makedirs(record_folder, exist_ok=True)
                video_success_cnt = 0
                video_fail_cnt = 0
                video_cnt = 0
                for summary in summaries:
                    if isinstance(summary, VideoSummary):
                        video = deepcopy(summary.value)
                        video = np.transpose(video, (0, 2, 3, 1))
                        video = video[:, :, :, ::-1]
                        if task_rewards[video_cnt] > 99:
                            video_path = os.path.join(
                                record_folder,
                                f"{task_name}_success_{video_success_cnt}.mp4",
                            )
                            video_success_cnt += 1
                        else:
                            video_path = os.path.join(
                                record_folder, f"{task_name}_fail_{video_fail_cnt}.mp4"
                            )
                            video_fail_cnt += 1
                        video_cnt += 1
                        os.makedirs(video_image_folder, exist_ok=True)
                        for idx in range(len(video) - 10):
                            cv2.imwrite(
                                os.path.join(video_image_folder, f"{idx}.png"), video[idx]
                            )
                        images_path = os.path.join(video_image_folder, r"%d.png")
                        os.system(
                            "ffmpeg -i {} -vf palettegen palette.png -hide_banner -loglevel error".format(
                                images_path
                            )
                        )
                        os.system(
                            "ffmpeg -framerate {} -i {} -i palette.png -lavfi paletteuse {} -hide_banner -loglevel error".format(
                                record_fps, images_path, video_path
                            )
                        )
                        os.remove("palette.png")
                        shutil.rmtree(video_image_folder)
                        cprint(f"saved {video_path}", "green")

        eval_env.shutdown()
        return scores


if __name__ == "__main__":
    parser = get_eval_parser()
    args = parser.parse_args()

    pc_runner = RLBenchPointcloudRunner()
    tasks_to_eval = ["light_bulb_in"]
    pc_runner.run(
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
        verbose=True,
        logging=False,
        save_video=False
    )

