import wandb
import numpy as np
import torch
import collections
import tqdm
import os
import csv

# from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.env import RLBenchEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper
from scipy.spatial.transform import Rotation as R

from diffusion_policy_3d.policy.base_pointcloud_policy import BasePointcloudPolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

# LOGGING
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.utils.transition import ReplayTransition
from yarr.agents.agent import ActResult
from yarr.agents.agent import VideoSummary

class RLBenchPointcloudRunner(BasePointcloudRunner):
    def __init__(self,
                 # RLBench config START
                 eval_datafolder,
                 episode_length,
                 headless,
                 record_every_n,
                 cameras,
                 task_name,
                 eval_episodes,
                 # RLBench config END
                 output_dir,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                #  fps=10,
                #  crf=22,
                #  render_size=84,
                 tqdm_interval_sec=5.0,
                #  n_envs=None,

                #  n_train=None,
                #  n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
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
        
        #TODO: add SimpleVideoRecordingWrapper
        def env_fn(task_name):
            return RLBenchEnv(
                tasks=[task_name],
                eval_datafolder=eval_datafolder,
                episode_length =episode_length,
                headless=headless,
                eval_episodes=eval_episodes,
                record_every_n=record_every_n,
                cameras=cameras,
                use_point_crop=use_point_crop,
                num_points=num_points
            )
        self.num_points = num_points
        self.episode_length = episode_length

        self.eval_episodes = eval_episodes

        # RLBench Env
        self.env = env_fn(self.task_name)
        self.env.eval = True

        # self.fps = fps
        # self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        # TODO: implement logger for RLBench Env.
        # self.logger_util_test = logger_util.LargestKRecorder(K=3)
        # self.logger_util_test10 = logger_util.LargestKRecorder(K=5)


    def run_evaluation(self, policy: BasePointcloudPolicy, replay_ground_truth: bool = True):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        env = self.env

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            # obs = env.reset()
            eval_demo_seed = episode_idx
            obs = env.reset_to_demo(eval_demo_seed)

            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            while not done:
                # # create obs dict
                # np_obs_dict = dict(obs)
                np_obs_dict = obs

                # NOTE: fake prev obs for testing RLBench Env
                H, W, C = np_obs_dict["image"].shape
                OBS_SPACE = np_obs_dict["agent_pos"].shape[0]

                np_obs_dict["image"] = np.tile(np_obs_dict["image"].reshape(1, H, W, C), (self.n_obs_steps, 1, 1, 1))
                np_obs_dict["agent_pos"] = np.tile(np_obs_dict["agent_pos"].reshape(1, OBS_SPACE), (self.n_obs_steps, 1))
                np_obs_dict["point_cloud"] = np.tile(np_obs_dict["point_cloud"].reshape(1, self.num_points, 3), (self.n_obs_steps, 1, 1))
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # NOTE: select one only
                # normalized quaternion
                action = action[0]
                r = R.from_quat(action[3:3+4])
                action[3:3+4] = r.as_quat()
                import pdb;pdb.set_trace()

                # step env
                # obs, reward, done, info = env.step(action)
                transition = env.step(action)

                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)
            

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]  # select first frame
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        # clear out video buffer
        _ = env.reset()
        # clear memory
        videos = None

        return log_data

    # TODO: fix logging, log_dir, save_video, and replay_ground_truth params.
    def run(self, policy: BasePointcloudPolicy, 
                log_dir: str = "/home/nil/manipulation/debug",
                save_video: bool = True,
                replay_ground_truth: bool = False
            ):

        stats_accumulator = SimpleAccumulator(eval_video_fps=30)

        eval_env = self.env
        eval_env.launch()

        scores = []
        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            episode_rollout = []
            generator = self.generator(
                policy=policy,
                eval_demo_seed=episode_idx,
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
                # current_task_id = transition.info["active_task_id"]
                # assert current_task_id == task_id

            reward = episode_rollout[-1].reward
            # task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            print(
                f"Evaluating {self.task_name} | Episode {episode_idx} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
            )

        # REPORT SUMMARIES
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = self.task_name

        for s in summaries:
            if "eval" in s.name:
                s.name = "%s/%s" % (s.name, task_name)
        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"

        print(f"[Evaluation] Finished {task_name} | Final Score: {task_score}\n")
        scores.append(task_score)

        if save_video:
            import cv2
            import shutil

            video_image_folder = "./tmp"
            record_fps = 25
            record_folder = os.path.join(log_dir, "videos")
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

        eval_env.shutdown()

        import pdb;pdb.set_trace()
        return scores

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, policy, eval_demo_seed, replay_ground_truth, record_enabled=False):
        env = self.env
        obs = env.reset_to_demo(eval_demo_seed)
        device = policy.device
        agent = policy 
        episode_length = self.episode_length

        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * self.n_obs_steps for k, v in obs.items()}
        
        # TODO: check!
        for step in range(episode_length):

            # device transfer
            obs_dict = {k:torch.tensor(np.array([v]), device=device) for k, v in obs_history.items()}
            
            # device transfer
            # obs_dict = dict_apply(obs_history,
            #                         lambda x: torch.from_numpy(x).to(
            #                             device=device))

            # run policy
            with torch.no_grad():
                obs_dict_input = {}  # flush unused keys
                # obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                # obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                obs_dict_input['point_cloud'] = obs_dict['point_cloud']
                obs_dict_input['agent_pos'] = obs_dict['agent_pos']
                action_dict = policy.predict_action(obs_dict_input)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                                        lambda x: x.detach().to('cpu').numpy())
                                            
            # TODO: fix only select the first action in the trajectory
            action = np_action_dict['action'].squeeze(0)

            # NOTE: select one only
            # normalized quaternion
            action = action[0]
            r = R.from_quat(action[3:3+4])
            action[3:3+4] = r.as_quat()
            act_result = ActResult(action)
            
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
                    # act_result = agent.act(step_signal.value, prepped_data,
                    #                     deterministic=eval)
                    # TODO 

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