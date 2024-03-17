import wandb
import numpy as np
import torch
import collections
import tqdm
# from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.env import RLBenchEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_pointcloud_policy import BasePointcloudPolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

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

        self.eval_episodes = eval_episodes

        # RLBench Env
        self.env = env_fn(self.task_name)
        self.env.eval = True
        self.env.launch()

        # self.fps = fps
        # self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        # TODO: implement logger for RLBench Env.
        # self.logger_util_test = logger_util.LargestKRecorder(K=3)
        # self.logger_util_test10 = logger_util.LargestKRecorder(K=5)


    def run(self, policy: BasePointcloudPolicy, replay_ground_truth: bool = True):
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
                import pdb; pdb.set_trace()
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

                import pdb; pdb.set_trace()
                # step env
                obs, reward, done, info = env.step(action)


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
