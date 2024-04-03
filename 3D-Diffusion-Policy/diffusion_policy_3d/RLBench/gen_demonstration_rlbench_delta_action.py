# bash scripts/metaworld/gen_demonstration_expert.sh reach 5
import argparse
import os
import zarr
import numpy as np
import matplotlib.pyplot as plt

from diffusion_policy_3d.RLBench.utils.config import ARGConfig
from diffusion_policy_3d.RLBench.utils.default_config import default_config, dmc_config
# from diffusion_policy_3d.env import MetaWorldEnv
# from model.algorithm import BAC
from termcolor import cprint
import copy
import imageio
# from metaworld.policies import *
# import faulthandler
# faulthandler.enable()

# RLBench
from diffusion_policy_3d.RLBench.rlbench_utils import get_stored_demo
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling
import visualizer
from scipy.spatial.transform import Rotation


"""
This script prepare RLBench data into action representation that match the real
world experiment in 3D diffusion policy deploying on Franka. 
"""


TASK_BOUDNS = {
    # x_min, y_min, z_min, x_max, y_max, z_max
    'default': [-1, -100, 0.78, 100, 100, 100],
    'remove_scene': [-1, -100, 0, 100, 100, 100],
    'remove_table': [-1, -100, 0.6, 100, 100, 100],
    'slide_block_to_color_target': [-0.4, -0.4, 0, 100, 0.4, 100]
}
ROTATION_RESOLUTION = 5 # degree increments per axis

seed = np.random.randint(0, 100)

# def load_mw_policy(task_name):
# 	if task_name == 'peg-insert-side':
# 		agent = SawyerPegInsertionSideV2Policy()
# 	else:
# 		task_name = task_name.split('-')
# 		task_name = [s.capitalize() for s in task_name]
# 		task_name = "Sawyer" + "".join(task_name) + "V2Policy"
# 		agent = eval(task_name)()
# 	return agent

def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)

def sensitive_gimble_fix(euler):
    """
    :param euler: euler angles in degree as np.ndarray in shape either [3] or
    [b, 3]
    """
    # selecting sensitive angle
    select1 = (89 < euler[..., 1]) & (euler[..., 1] < 91)
    euler[select1, 1] = 90
    # selecting sensitive angle
    select2 = (-91 < euler[..., 1]) & (euler[..., 1] < -89)
    euler[select2, 1] = -90

    # recalulating the euler angles, see assert
    r = Rotation.from_euler("xyz", euler, degrees=True)
    euler = r.as_euler("xyz", degrees=True)

    select = select1 | select2
    assert (euler[select][..., 2] == 0).all(), euler

    # uncomment the following to move values from x to z while keeping the
    # angle same i.e. x becomes 0 and z has the values
    # TODO: verify if this helps
    ######################################################################
    ## when y=90, x-z is invariant
    # euler[select1, 2] = -euler[select1, 0]
    # euler[select1, 0] = 0

    ## when y=-90, x+z is invariant
    # euler[select2, 2] = euler[select2, 0]
    # euler[select2, 0] = 0
    ######################################################################

    return euler

def _quaternion_to_discrete_euler(
    quaternion, resolution, gimble_fix=True,
    norm_operation=True):
    """
    :param quaternion: quaternion in shape [4]
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    :param norm_operation: if True, normalize the quaternion before converting
    just like peract. we haven't tested if this is necessarily required
    """
    if norm_operation:
        quaternion = normalize_quaternion(quaternion)
        if quaternion[-1] < 0:
            quaternion = -quaternion

    r = Rotation.from_quat(quaternion)

    euler = r.as_euler("xyz", degrees=True)
    if gimble_fix:
        euler = sensitive_gimble_fix(euler)

    euler += 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc

def quaternion_to_discrete_euler(
    quaternion, resolution, gimble_fix=True,
    norm_operation=True):
    """
    :param quaternion: quaternion in shape [4] or [b, 4]
    :param gimble_fix: the euler values for x and y can be very sensitive
        around y=90 degrees. this leads to a multimodal distribution of x and y
        which could be hard for a network to learn. When gimble_fix is true, around
        y=90, we change the mode towards x=0, potentially making it easy for the
        network to learn.
    :param norm_operation: if True, normalize the quaternion before converting
    just like peract. we haven't tested if this is necessarily required
    """
    assert quaternion.shape[-1] == 4
    assert quaternion.ndim in [1, 2]
    if quaternion.ndim == 1:
        return _quaternion_to_discrete_euler(
            quaternion, resolution, gimble_fix, norm_operation)
    else:
        out = np.array([
            _quaternion_to_discrete_euler(
                q, resolution, gimble_fix, norm_operation)
            for q in quaternion])
        return out

def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()

def main(args):
    env_name = args.env_name

    save_dir = os.path.join(args.save_dir, 'rlbench_'+args.env_name+f'_{args.rot_representation}'+'_delta_action_expert.zarr')
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    use_point_crop = args.use_point_crop
    num_points = args.num_points
    if use_point_crop:
        x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[args.env_name]
        min_bound = [x_min, y_min, z_min]
        max_bound = [x_max, y_max, z_max]

    # e = MetaWorldEnv(env_name, is_state_based=True, device="cuda:0", use_point_crop=use_point_crop, num_points=num_points)
    
    # create policy
    # policy_path = f"expert_ckpt/{env_name}.torch"
    # print("Policy path: ", policy_path)
    # config.update({"device": "3"})
    # pi = BAC(e.observation_space.shape[0], e.action_space, config)
    # pi.load_checkpoint(policy_path, evaluate=True)

    num_episodes = args.num_episodes
    cprint(f"Number of episodes : {num_episodes}", "yellow")
    

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    
    
    episode_idx = 0

    # RLBench raw data
    EPISODES_FOLDER = f"{env_name}/all_variations/episodes"
    data_path = os.path.join(args.root_dir, EPISODES_FOLDER)
    


    # mw_policy = load_mw_policy(env_name)
    # cprint(f"Using script policy: {args.use_script_policy}", "yellow")

    # loop over episodes
    while episode_idx < num_episodes:
        # raw_state = e.reset()

        # obs_dict = e.get_visual_obs()

        # done = False
        
        ep_reward = 0.
        ep_success = False
        ep_success_times = 0
        

        img_arrays_sub = []
        point_cloud_arrays_sub = []
        depth_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []
        total_count_sub = 0
        
        # NOTE: placeholder for calculating the first delta action.
        if args.rot_representation == "quat":
            prev_robot_state = np.zeros(8) # xyz(3) + quaternion(4) + gripper state(1)
        elif args.rot_representation == "euler":
            prev_robot_state = np.zeros(7) # xyz + euler(3) + gripper state

        # NOTE: demo type=rlbench.demo.demo; Details in https://github.com/stepjam/RLBench/blob/master/rlbench/demo.py
        #       obs type=rlbenc.backend.observation.Observation; Details in https://github.com/stepjam/RLBench/blob/master/rlbench/backend/observation.py
        demo = get_stored_demo(data_path=data_path, index=episode_idx)
        for idx in range(len(demo)):
            # NOTE: only use front camera information, add rgb information as well.
            obs = demo[idx]

            point_cloud = obs.front_point_cloud.reshape(-1, 3) # (H, W, 3) -> (H*W, 3)
            # NOTE: add rgb information; normalize the array 0-1 scale
            front_rgb = obs.front_rgb.astype("float32") / 255.0

            point_cloud = np.concatenate((point_cloud, front_rgb.reshape(-1, 3)), axis=1) # (H * W, 6)
            # NOTE: crop background.
            if use_point_crop:
                mask = np.all(point_cloud[:, :3] > min_bound, axis=1)
                point_cloud = point_cloud[mask]

                mask = np.all(point_cloud[:, :3] < max_bound, axis=1)
                point_cloud = point_cloud[mask]
            point_cloud = point_cloud_sampling(point_cloud, num_points, 'fps') # (num_points, 3)

            # # debugging
            # visualizer.visualize_pointcloud(point_cloud)
            # import pdb; pdb.set_trace()

            GRIPPER_OPEN_CONTINUOUS_VALUE = 0.085
            GRIPPER_CLOSE_CONTINUOUS_VALUE = 0.
    
            obs_dict = {
                "image": obs.front_rgb,
                "agent_pos": obs.gripper_pose[0:3],
                "point_cloud": point_cloud,
                "depth": obs.front_depth,
                # NOTE: change the gripper state to continuous value.
                "grip": GRIPPER_OPEN_CONTINUOUS_VALUE if bool(obs.gripper_open) else GRIPPER_CLOSE_CONTINUOUS_VALUE,
                "quat": obs.gripper_pose[3:],
                "euler": np.deg2rad(quaternion_to_discrete_euler(obs.gripper_pose[3:], resolution=ROTATION_RESOLUTION))
            }

            total_count_sub += 1

            obs_img = obs_dict['image']
            obs_point_cloud = obs_dict['point_cloud']
            obs_depth = obs_dict['depth']
    
            # NOTE: change the agent_pos to agent state: xyz, euler angle, and gripper state.
            if args.rot_representation == "quat":
                obs_robot_state = np.concatenate([obs.gripper_pose, np.array([obs_dict["grip"]])]) # xyz(3) + quaternion(4) + gripper state(1)
            elif args.rot_representation == "euler":
                obs_robot_state = np.concatenate([obs_dict["agent_pos"], obs_dict["euler"], np.array([obs_dict["grip"]])]) # xyz(3) + euler(3) + gripper state(1) 

            # NOTE: action is the delta value between two robot states.
            THRESHOLD = 0.001
            delta_action = obs_robot_state - prev_robot_state
            delta_action[np.abs(delta_action) < THRESHOLD] = 0
            prev_robot_state = obs_robot_state
            
            img_arrays_sub.append(obs_img)
            point_cloud_arrays_sub.append(obs_point_cloud)
            depth_arrays_sub.append(obs_depth)
            
            action_arrays_sub.append(delta_action)
            state_arrays_sub.append(obs_robot_state)


            terminal = idx == len(demo) - 1
            reward = float(terminal) * 1.0 if terminal else 0
            ep_reward += reward


            ep_success = terminal
            ep_success_times += ep_success

        if not ep_success:
            cprint(f'Episode: {episode_idx} failed with reward {ep_reward} and success times {ep_success_times}', 'red')
            continue            
        else:
            # NOTE: the delta action is the action at the current observation want to take, which is 3d diffusion policy
            # will predict. That means we need to shift 1 index.
            # The code `action_arrays_sub` appears to be a placeholder or a comment in the Python
            # code. It does not perform any specific action or operation in Python.
            action_arrays_sub = action_arrays_sub[1:] 
            if args.rot_representation == "quat":
                action_arrays_sub.append(np.zeros(8))
            elif args.rot_representation == "euler":
                action_arrays_sub.append(np.zeros(7))
            
            total_count += total_count_sub
            episode_ends_arrays.append(copy.deepcopy(total_count)) # the index of the last step of the episode    
            img_arrays.extend(copy.deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(copy.deepcopy(point_cloud_arrays_sub))
            depth_arrays.extend(copy.deepcopy(depth_arrays_sub))
            state_arrays.extend(copy.deepcopy(state_arrays_sub))
            action_arrays.extend(copy.deepcopy(action_arrays_sub))
            cprint('Episode: {}, Reward: {}, Success Times: {}'.format(episode_idx, ep_reward, ep_success_times), 'green')
            episode_idx += 1

    # save data
     ###############################
    # save data
    ###############################
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)


    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')

    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    # del e, pi

if __name__ == "__main__":
    
    arg = ARGConfig()
    arg.add_arg("env_name", "coffee-button", "Environment name")
    arg.add_arg("device", "0", "Computing device")
    arg.add_arg("policy", "Gaussian", "Policy Type: Gaussian | Deterministic (default: Gaussian)")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("algo", "BAC", "choose algorithm (BAC, SAC, TD3, TD3-BEE)")
    arg.add_arg("start_steps", 10000, "Number of start steps")
    arg.add_arg("automatic_entropy_tuning", True, "Automaically adjust α (default: True)")
    arg.add_arg("quantile", 0.7, "the quantile regression for value function (default: 0.9)")
    arg.add_arg("seed", 0, "experiment seed")
    arg.add_arg("lambda", "fixed_0.5", "method to calculated lambda, fixed_x, ada, min, max")
    arg.add_arg("des", "", "short description for the experiment")
    arg.add_arg("num_steps", 1000001, "total number of steps")
    arg.add_arg("save_checkpoint", False, "save checkpoint or not")
    arg.add_arg("replay_size", 1000000, "size of replay buffer")


    arg.add_arg("num_episodes", 10, "number of total collected episodes")
    arg.add_arg("root_dir", "/home/nil/manipulation/RVT2/rvt/data/rlbench_data/128x128/", "directory to load rlbench raw data")
 
    arg.add_arg('use_point_crop', True, "use point crop or not")
    arg.add_arg('num_points', 512, "number of points in point cloud")
    arg.add_arg('use_script_policy', False, "use script policy or not")

    # RLBench
    arg.add_arg('save_dir', '/home/nil/manipulation/3D-Diffusion-Policy/3D-Diffusion-Policy/data', 'directory to save data')
    arg.add_arg('rot_representation', "quat", "rotation action: quaternion or euler")
    arg.parser()

    config = dmc_config  
    config.update(arg)
    main(config)