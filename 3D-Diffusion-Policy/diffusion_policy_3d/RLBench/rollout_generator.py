# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling RVT or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
#
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from multiprocessing import Value

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling
import visualizer
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.RLBench.gen_demonstration_rlbench import discrete_euler_to_quaternion

class RolloutGenerator(object):

    def __init__(self, 
                env_device = 'cuda:0', 
                use_point_crop = True, 
                rotation_euler = False,
                task_bound = None,
                num_points = 512,
                rotation_resolution=None):
        self._env_device = env_device
        self.use_point_crop = use_point_crop
        self.rotation_euler = rotation_euler
        self.task_bound = task_bound
        self.num_points = num_points
        self.rotation_resolution = rotation_resolution

        if self.use_point_crop:
            x_min, y_min, z_min, x_max, y_max, z_max = task_bound['default']
            self.min_bound = [x_min, y_min, z_min]
            self.max_bound = [x_max, y_max, z_max]
    
    def rlbench_obs2diffusion_policy_obs(self, obs):
        point_cloud = obs["front_point_cloud"].transpose((1, 2, 0)).reshape(-1, 3) # (H, W, 3) -> (H*W, 3)
        # NOTE: crop background.
        if self.use_point_crop:
            mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
            point_cloud = point_cloud[mask]

            mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
            point_cloud = point_cloud[mask]   

            point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps') # (num_points, 3)

            # # debugging
            # import pdb; pdb.set_trace()
            # visualizer.visualize_pointcloud(point_cloud)
        obs["gripper_open"] = np.expand_dims(obs["gripper_open"], 0) # shape (1,)
        agent_pos = np.concatenate([obs["gripper_pose"][0:3], obs["gripper_open"]]) # xyz + gripper open
        
        output_obs_history = {
            "agent_pos": agent_pos,
            "point_cloud": point_cloud
        }            

        return output_obs_history        

    def diffusion_policy_action2rlbench_action(self, diffusion_policy_action):
        RLBENCH_ACTION_DIM = 9
        WYPT_DIM = 3
        EULER_DIM = 3
        QUAT_DIM = 4
        GRIPPER_OPEN_DIM = 1
        
        # TODO: fix select only one action at a time
        action = diffusion_policy_action[0]
        
        if self.rotation_euler:
            rlbench_action = np.zeros(RLBENCH_ACTION_DIM) # xyz, quat, gripper open, ignore collision
            
            rlbench_action[:3] = action[:3] #xyz
            pred_rot_quat = discrete_euler_to_quaternion(
                action[3:3+3], self.rotation_resolution
            )
            rlbench_action[3:3+4] = pred_rot_quat # quat
            rlbench_action[7:7+1] = action[6]   # gripper open
            rlbench_action[8] = 0.0              # ignore collision
            
            rlbench_action = ActResult(rlbench_action)
            return rlbench_action
        else:
            raise NotImplementedError

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False,
                  replay_ground_truth: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
            # get ground-truth action sequence
            if replay_ground_truth:
                actions = env.get_ground_truth_action(eval_demo_seed)
        else:
            obs = env.reset()
        agent.reset()
        
        obs = self.rlbench_obs2diffusion_policy_obs(obs)
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        for step in range(episode_length):
            prepped_data = {k:torch.tensor(np.array([v]), device=self._env_device) for k, v in obs_history.items()}
            if not replay_ground_truth:
                # act_result = agent.act(step_signal.value, prepped_data,
                #                     deterministic=eval)
                with torch.no_grad():
                    action_dict = agent.predict_action(prepped_data)
                    # device_transfer
                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action'].squeeze(0) # [n_action_steps, action_dim]
                    act_result = self.diffusion_policy_action2rlbench_action(action)                    
            else:
                if step >= len(actions):
                    return
                act_result = ActResult(actions[step])

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            transition.observation = self.rlbench_obs2diffusion_policy_obs(transition.observation)
            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)

            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
