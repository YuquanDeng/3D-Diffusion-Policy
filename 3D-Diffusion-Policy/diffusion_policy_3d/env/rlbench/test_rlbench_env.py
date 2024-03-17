import argparse
import os
from copy import deepcopy

# RLBench
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper

from rlbench import CameraConfig, ObservationConfig
from pyrep.const import RenderMode
from typing import List

import numpy as np
from rlbench.action_modes.arm_action_modes import (
    EndEffectorPoseViaPlanning,
    Scene,
)

from rlbench_wrapper import RLBenchEnv



RLBENCH_TASKS = \
        ['put_item_in_drawer', 'reach_and_drag', 'turn_tap',
         'slide_block_to_color_target', 'open_drawer',
         'put_groceries_in_cupboard', 'place_shape_in_shape_sorter',
         'put_money_in_safe', 'push_buttons', 'close_jar',
         'stack_blocks', 'place_cups', 'place_wine_at_rack_location',
         'light_bulb_in', 'sweep_to_dustpan_of_size',
         'insert_onto_square_peg', 'meat_off_grill', 'stack_cups']

IMAGE_SIZE =  128

def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        default=["insert_onto_square_peg"])
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        default=["front", "left_shoulder", "right_shoulder", "wrist"])
    parser.add_argument(
        '--model-folder',
        type=str,
        default=None)
    parser.add_argument(
        '--eval-datafolder',
        type=str,
        default='./data/val/')
    parser.add_argument(
        '--start-episode',
        type=int,
        default=0,
        help='start to evaluate from which episode')
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='how many episodes to be evaluated for each task')
    parser.add_argument(
        '--episode-length',
        type=int,
        default=25,
        help='maximum control steps allowed for each episode')
    parser.add_argument(
        '--headless',
        action='store_true',
        default=False)
    parser.add_argument(
        '--ground-truth',
        action='store_true',
        default=False)
    parser.add_argument(
        '--peract_official',
        action='store_true')
    parser.add_argument(
        '--mohits_model',
        action='store_true')
    parser.add_argument(
        '--peract_model_dir',
        type=str,
        default='runs/peract_official/seed0/weights/600000')
    parser.add_argument(
        '--device',
        type=int,
        default=0)
    parser.add_argument(
        '--record-every-n',
        type=int,
        default=5)
    parser.add_argument(
        '--log-name',
        type=str,
        default=None)
    parser.add_argument(
        '--ngc',
        action='store_true')
    parser.add_argument(
        '--mode',
        type=str,
        default='final',
        choices=['final', 'all', 'single'])
    parser.add_argument(
        '--model-name',
        type=str,
        default=None)
    parser.add_argument(
        '--use-input-place-with-mean',
        action='store_true')
    parser.add_argument(
        '--save-video',
        action='store_true')
    parser.add_argument(
        '--skip',
        action='store_true')
    parser.add_argument(
        '--ortho_cam',
        help='use orthographic input camera system',
        action='store_true')

    # Variations
    parser.add_argument(
        '--variations',
        type=int,
        nargs='+',
        default=[-1])
    
    # TODO: add excluded_variations args

    # VLM
    parser.add_argument(
        '--use_mask',
        action='store_true',
        help='use mask generated from VLM in evaluation',
        default=False)
    parser.add_argument(
        '--qwen_path',
        type=str,
        default=None)
    parser.add_argument(
        '--save_mask',
        action='store_true',
        help='save mask generated from rollout under log_dir',
        default=False)
    parser.add_argument(
        '--same_lang_goal',
        action='store_true',
        help='use same lang tokens for same RLbench tasks',
        default=False
    )
    parser.add_argument(
        '--mask_type',
        type=str,
        default="qwen")
    parser.add_argument(
        '--traj-datafolder',
        type=str,
        default=None)
    return parser

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


def test_rlbench(tasks, record_every_n, eval_episodes, headless,episode_length, eval_datafolder, cameras, verbose=True):
    # task class
    task_classes = []
    if tasks[0] == "all":
        tasks = RLBENCH_TASKS
        if verbose:
            print(f"evaluate on {len(tasks)} tasks: ", tasks)

    task_files = [
        t.replace(".py", "")
        for t in os.listdir(rlbench_task.TASKS_PATH)
        if t != "__init__.py" and t.endswith(".py")
    ]

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

    # import pdb;pdb.set_trace()

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


if __name__ == "__main__":
    parser = get_eval_parser()

    args = parser.parse_args()

    tasks_to_eval = deepcopy(args.tasks)
    test_rlbench(
        tasks=tasks_to_eval,
        eval_datafolder = args.eval_datafolder,
        episode_length = args.episode_length,
        headless = args.headless,
        eval_episodes = args.eval_episodes,
        record_every_n = args.record_every_n,
        cameras=args.cameras
    )

    # if args.log_name is None:
    #     args.log_name = get_time_stamp()

    # if not (args.peract_official):
    #     args.eval_log_dir = os.path.join(args.model_folder, "eval", args.log_name)
    # else:
    #     args.eval_log_dir = os.path.join(args.peract_model_dir, "eval", args.log_name)
    #     assert args.model_name == 'QAttentionAgent_layer0.pt', args.model_name

    # os.makedirs(args.eval_log_dir, exist_ok=True)

    # # save the arguments for future reference
    # with open(os.path.join(args.eval_log_dir, "eval_config.yaml"), "w") as fp:
    #     yaml.dump(args.__dict__, fp)
    # _eval(args)