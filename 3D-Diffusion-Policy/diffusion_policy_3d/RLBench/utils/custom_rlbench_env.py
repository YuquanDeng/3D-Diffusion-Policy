# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# from rvt.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
# from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from diffusion_policy_3d.RLBench.utils.demo_loading_utils import keypoint_discovery
import numpy as np

# RLBenchEnv and CustomRLBenchEnv
from typing import Type, List

import numpy as np
from rlbench import ObservationConfig, ActionMode
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation
from rlbench.backend.task import Task
from yarr.agents.agent import ActResult, VideoSummary, TextSummary
from yarr.envs.rlbench_env import RLBenchEnv, MultiTaskRLBenchEnv
from yarr.utils.observation_type import ObservationElement
from yarr.utils.transition import Transition
from yarr.utils.process_str import change_case

from pyrep.const import RenderMode
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects import VisionSensor, Dummy

# Lang
import clip
import torch
from diffusion_policy_3d.CLIP.clip import build_model, load_clip, tokenize
from diffusion_policy_3d.CLIP.clip_utils import _clip_encode_text

# adapt from YARR/yarr/env/rlbench_env.py
ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                        'gripper_open', 'gripper_pose',
                        'gripper_joint_positions', 'gripper_touch_forces',
                        'task_low_dim_state', 'misc']

def _extract_obs(obs: Observation, channels_last: bool, observation_config):
    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    robot_state = obs.get_low_dim_data()
    # Remove all of the individual state elements
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in ROBOT_STATE_KEYS}
    if not channels_last:
        # Swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                    for k, v in obs_dict.items()}
    else:
        # Add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)
    obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for config, name in [
        (observation_config.left_shoulder_camera, 'left_shoulder'),
        (observation_config.right_shoulder_camera, 'right_shoulder'),
        (observation_config.front_camera, 'front'),
        (observation_config.wrist_camera, 'wrist'),
        (observation_config.overhead_camera, 'overhead')]:
        if config.point_cloud:
            obs_dict['%s_camera_extrinsics' % name] = obs.misc['%s_camera_extrinsics' % name]
            obs_dict['%s_camera_intrinsics' % name] = obs.misc['%s_camera_intrinsics' % name]
    return obs_dict

class CustomRLBenchEnv(RLBenchEnv):

    def __init__(self,
                 task_class: Type[Task],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20):
        super(CustomRLBenchEnv, self).__init__(
            task_class, observation_config, action_mode, dataset_root,
            channels_last, headless=headless,
            include_lang_goal_in_obs=include_lang_goal_in_obs)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        obs_dict = super(CustomRLBenchEnv, self).extract_obs(obs)

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        # obs_dict['gripper_pose'] = grip_pose
        return obs_dict

    def launch(self):
        super(CustomRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail'),
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i):
        self._i = 0
        # super(CustomRLBenchEnv, self).reset()

        self._task.set_variation(-1)
        d, = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict


class CustomMultiTaskRLBenchEnv(MultiTaskRLBenchEnv):

    def __init__(self,
                 task_classes: List[Type[Task]],
                 observation_config: ObservationConfig,
                 action_mode: ActionMode,
                 episode_length: int,
                 dataset_root: str = '',
                 channels_last: bool = False,
                 reward_scale=100.0,
                 headless: bool = True,
                 swap_task_every: int = 1,
                 time_in_state: bool = False,
                 include_lang_goal_in_obs: bool = False,
                 record_every_n: int = 20, 
                 device=None):
        super(CustomMultiTaskRLBenchEnv, self).__init__(
            task_classes, observation_config, action_mode, dataset_root,
            channels_last, headless=headless, swap_task_every=swap_task_every,
            include_lang_goal_in_obs=include_lang_goal_in_obs)
        self._reward_scale = reward_scale
        self._episode_index = 0
        self._record_current_episode = False
        self._record_cam = None
        self._previous_obs, self._previous_obs_dict = None, None
        self._recorded_images = []
        self._episode_length = episode_length
        self._time_in_state = time_in_state
        self._record_every_n = record_every_n
        self._i = 0
        self._error_type_counts = {
            'IKError': 0,
            'ConfigurationPathError': 0,
            'InvalidActionError': 0,
        }
        self._last_exception = None
        
        # Load pre-trained language model
        if self._include_lang_goal_in_obs:
            self.device = device
            model, _ = load_clip("RN50", self.device , jit=False)
            self.clip_rn50 = build_model(model.state_dict()).to(self.device )
            del model

        self.lang_goal_embed = None

    @property
    def observation_elements(self) -> List[ObservationElement]:
        obs_elems = super(CustomMultiTaskRLBenchEnv, self).observation_elements
        for oe in obs_elems:
            if oe.name == 'low_dim_state':
                oe.shape = (oe.shape[0] - 7 * 3 + int(self._time_in_state),)  # remove pose and joint velocities as they will not be included
                self.low_dim_state_len = oe.shape[0]
        return obs_elems

    def extract_obs(self, obs: Observation, t=None, prev_action=None):
        obs.joint_velocities = None
        grip_mat = obs.gripper_matrix
        grip_pose = obs.gripper_pose
        joint_pos = obs.joint_positions
        obs.gripper_pose = None
        # obs.gripper_pose = None
        obs.gripper_matrix = None
        obs.wrist_camera_matrix = None
        obs.joint_positions = None
        if obs.gripper_joint_positions is not None:
            obs.gripper_joint_positions = np.clip(
                obs.gripper_joint_positions, 0., 0.04)

        # obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)
        extracted_obs = _extract_obs(obs, self._channels_last, self._observation_config)
        
        # if not reset
        if isinstance(self.lang_goal_embed, np.ndarray):
            extracted_obs["lang_goal_embed"] = self.lang_goal_embed
            extracted_obs["lang_goal"] = self._lang_goal

        obs_dict = extracted_obs

        if self._time_in_state:
            time = (1. - ((self._i if t is None else t) / float(
                self._episode_length - 1))) * 2. - 1.
            obs_dict['low_dim_state'] = np.concatenate(
                [obs_dict['low_dim_state'], [time]]).astype(np.float32)

        obs.gripper_matrix = grip_mat
        # obs.gripper_pose = grip_pose
        obs.joint_positions = joint_pos
        obs.gripper_pose = grip_pose
        obs_dict['gripper_pose'] = obs.gripper_pose
        obs_dict['gripper_open'] = np.array(obs.gripper_open)
        return obs_dict

    def launch(self):
        super(CustomMultiTaskRLBenchEnv, self).launch()
        self._task._scene.register_step_callback(self._my_callback)
        if self.eval:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam_base = Dummy('cam_cinematic_base')
            cam_base.rotate([0, 0, np.pi * 0.75])
            self._record_cam = VisionSensor.create([320, 180])
            self._record_cam.set_explicit_handling(True)
            self._record_cam.set_pose(cam_placeholder.get_pose())
            self._record_cam.set_render_mode(RenderMode.OPENGL)

    def reset(self) -> dict:
        self._i = 0
        self._previous_obs_dict = super(CustomMultiTaskRLBenchEnv, self).reset()
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()
        return self._previous_obs_dict

    def register_callback(self, func):
        self._task._scene.register_step_callback(func)

    def _my_callback(self):
        if self._record_current_episode:
            self._record_cam.handle_explicitly()
            cap = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
            self._recorded_images.append(cap)

    def _append_final_frame(self, success: bool):
        self._record_cam.handle_explicitly()
        img = (self._record_cam.capture_rgb() * 255).astype(np.uint8)
        self._recorded_images.append(img)
        final_frames = np.zeros((10, ) + img.shape[:2] + (3,), dtype=np.uint8)
        # Green/red for success/failure
        final_frames[:, :, :, 1 if success else 0] = 255
        self._recorded_images.extend(list(final_frames))

    def step(self, act_result: ActResult) -> Transition:
        action = act_result.action
        success = False
        obs = self._previous_obs_dict  # in case action fails.

        try:
            obs, reward, terminal = self._task.step(action)
            if reward >= 1:
                success = True
                reward *= self._reward_scale
            else:
                reward = 0.0
            obs = self.extract_obs(obs)
            self._previous_obs_dict = obs
        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            terminal = True
            reward = 0.0

            if isinstance(e, IKError):
                self._error_type_counts['IKError'] += 1
            elif isinstance(e, ConfigurationPathError):
                self._error_type_counts['ConfigurationPathError'] += 1
            elif isinstance(e, InvalidActionError):
                self._error_type_counts['InvalidActionError'] += 1

            self._last_exception = e

        summaries = []
        self._i += 1
        if ((terminal or self._i == self._episode_length) and
                self._record_current_episode):
            self._append_final_frame(success)
            vid = np.array(self._recorded_images).transpose((0, 3, 1, 2))
            task_name = change_case(self._task._task.__class__.__name__)
            summaries.append(VideoSummary(
                'episode_rollout_' + ('success' if success else 'fail') + f'/{task_name}',
                vid, fps=30))

            # error summary
            error_str = f"Errors - IK : {self._error_type_counts['IKError']}, " \
                        f"ConfigPath : {self._error_type_counts['ConfigurationPathError']}, " \
                        f"InvalidAction : {self._error_type_counts['InvalidActionError']}"
            if not success and self._last_exception is not None:
                error_str += f"\n Last Exception: {self._last_exception}"
                self._last_exception = None

            summaries.append(TextSummary('errors', f"Success: {success} | " + error_str))
        return Transition(obs, reward, terminal, summaries=summaries)

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        # super(CustomMultiTaskRLBenchEnv, self).reset()

        # if variation_number == -1:
        #     self._task.sample_variation()
        # else:
        #     self._task.set_variation(variation_number)

        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        _, obs = self._task.reset_to_demo(d)
        self._lang_goal = self._task.get_task_descriptions()[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict
    
class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):

    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)        

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
                self.eval and self._record_every_n > 0 and self._episode_index % self._record_every_n == 0)
        return self._previous_obs_dict

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
                self.eval and self._record_every_n > 0 and self._episode_index % self._record_every_n == 0)
        self._episode_index += 1
        self._recorded_images.clear()

        
        # extract language embedding
        if self._include_lang_goal_in_obs:
            tokens = clip.tokenize([self._lang_goal]).numpy()
            token_tensor = torch.from_numpy(tokens).to(self.device)
            with torch.no_grad():
                lang_feats, lang_embs = _clip_encode_text(self.clip_rn50, token_tensor)
            self._previous_obs_dict["lang_goal_embed"] = lang_feats[0].float().detach().cpu().numpy() # shape (1024,)    
            self._previous_obs_dict["lang_goal"] = self._lang_goal
            self.lang_goal_embed = self._previous_obs_dict["lang_goal_embed"]

        return self._previous_obs_dict

    def get_ground_truth_action(self, i):
        self._task.set_variation(-1)
        demo = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i)[0]

        episode_keypoints = keypoint_discovery(demo)
        actions = []
        for keypoint in episode_keypoints:
            obs_tp1 = demo[keypoint]
            obs_tm1 = demo[max(0, keypoint - 1)]
            action_trans = obs_tp1.gripper_pose[:3]
            action_quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
            if action_quat[-1] < 0:
                action_quat = -action_quat
            action_grip = float(obs_tp1.gripper_open)
            action_ignore_collisions = float(obs_tm1.ignore_collisions)
            actions.append(np.concatenate((action_trans, action_quat, [action_grip], [action_ignore_collisions])))

        return actions

def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)