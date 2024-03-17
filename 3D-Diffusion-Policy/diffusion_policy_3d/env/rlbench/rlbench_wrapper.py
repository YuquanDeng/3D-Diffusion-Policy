# code adapted from RVT.
from typing import List
import os

from diffusion_policy_3d.env.rlbench.custom_rlbench_env import CustomMultiTaskRLBenchEnv

# RLBench & Pyrep
from rlbench.backend.utils import task_file_to_task_class
from rlbench.backend import task as rlbench_task
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.backend.observation import Observation

# EndEffectorPoseViaPlanning2
import numpy as np
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)

# Poin cloud processing
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling
import visualizer

IMAGE_SIZE =  128

RLBENCH_TASKS = \
        ['put_item_in_drawer', 'reach_and_drag', 'turn_tap',
         'slide_block_to_color_target', 'open_drawer',
         'put_groceries_in_cupboard', 'place_shape_in_shape_sorter',
         'put_money_in_safe', 'push_buttons', 'close_jar',
         'stack_blocks', 'place_cups', 'place_wine_at_rack_location',
         'light_bulb_in', 'sweep_to_dustpan_of_size',
         'insert_onto_square_peg', 'meat_off_grill', 'stack_cups']

TASK_BOUDNS = {
    # x_min, y_min, z_min, x_max, y_max, z_max
    'default': [-1, -100, 0, 100, 100, 100],
}

def create_obs_config(camera_names: List[str],
                       camera_resolution: List[int],
                       method_name: str):
    unused_cams = CameraConfig()
    unused_cams.set_all(False)
    used_cams = CameraConfig(
        rgb=True,
        point_cloud=True,
        mask=False,
        depth=False,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),
        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,
        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config

class EndEffectorPoseViaPlanning2(EndEffectorPoseViaPlanning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def action(self, scene: Scene, action: np.ndarray, ignore_collisions: bool = True):
        action[:3] = np.clip(
            action[:3],
            np.array(
                [scene._workspace_minx, scene._workspace_miny, scene._workspace_minz]
            )
            + 1e-7,
            np.array(
                [scene._workspace_maxx, scene._workspace_maxy, scene._workspace_maxz]
            )
            - 1e-7,
        )

        super().action(scene, action, ignore_collisions)

class RLBenchEnv(CustomMultiTaskRLBenchEnv):

    def __init__(self, 
            tasks,
            eval_datafolder,
            episode_length,
            headless,
            eval_episodes,
            record_every_n,
            cameras,
            verbose=True,
            use_point_crop=True,
            num_points=512
        ):

        # tasks class
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
            
        # obs_config
        camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
        obs_config = create_obs_config(cameras, camera_resolution, method_name="")   

        # action mode
        gripper_mode = Discrete()
        arm_action_mode = EndEffectorPoseViaPlanning2()
        action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode) 

        # customize for 3d diffusion policy
        # TODO: add use_point_crop
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        if self.use_point_crop:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
            self.min_bound = [x_min, y_min, z_min]
            self.max_bound = [x_max, y_max, z_max]

        super(RLBenchEnv, self).__init__(
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

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
                self.eval and self._record_every_n > 0 and self._episode_index % self._record_every_n == 0)
        return self._previous_obs_dict

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

        obs_dict = super(CustomMultiTaskRLBenchEnv, self).extract_obs(obs)

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

        # NOTE: customize observation for 3d diffusion policy evaluation.
        point_cloud = obs.front_point_cloud.reshape(-1, 3) # (H, W, 3) -> (H*W, 3)
        # NOTE: crop background.
        if self.use_point_crop:
            mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
            point_cloud = point_cloud[mask]

            mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
            point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps') # (num_points, 3)
        
        # debugging
        # visualizer.visualize_pointcloud(point_cloud)
        # import pdb; pdb.set_trace()

        obs_dict = {
            "image": obs.front_rgb,
            "agent_pos": np.concatenate([obs.gripper_pose[0:3], np.array([float(obs.gripper_open)])]),
            "point_cloud": point_cloud,
            # "depth": obs.front_depth,
        }

        return obs_dict
    
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

        return self._previous_obs_dict
