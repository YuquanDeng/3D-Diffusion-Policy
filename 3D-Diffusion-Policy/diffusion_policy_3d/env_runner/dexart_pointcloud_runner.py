import wandb
import numpy as np
import torch
import collections
import tqdm
from termcolor import cprint
from diffusion_policy_3d.env import DexArtEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_pointcloud_policy import BasePointcloudPolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
import diffusion_policy_3d.common.logger_util as logger_util


class DexArtPointcloudRunner(BasePointcloudRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 train_start_seed=0,
                 n_test=22,
                 test_start_seed=10000,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train
        # test env uses different instances
        self.env_test = env_fn(is_test=True)
        self.episode_test = n_test

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)
        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)
        
    def run(self, policy: BasePointcloudPolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train
        env_test = self.env_test

        all_returns_train = []
        all_success_rates_train = []
        all_returns_test = []
        all_success_rates_test = []

        

        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",
                                    leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()

            policy.reset()

            done = False
            reward_sum = 0.
            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns_train.append(reward_sum)
            all_success_rates_train.append(env_train.is_success())

        ##############################
        # test env loop
        for episode_id in tqdm.tqdm(range(self.episode_test), desc=f"DexArt {self.task_name} Train Env",
                                    leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_test.reset()

            policy.reset()

            done = False
            reward_sum = 0.

            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    action_dict = policy.predict_action(obs_dict_input)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env_test.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns_test.append(reward_sum)
            all_success_rates_test.append(env_test.is_success())

        SR_mean_train = np.mean(all_success_rates_train)
        SR_mean_test = np.mean(all_success_rates_test)
        returns_mean_train = np.mean(all_returns_train)
        returns_mean_test = np.mean(all_returns_test)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_success_rates_test'] = SR_mean_test
        log_data['mean_returns_train'] = returns_mean_train
        log_data['mean_returns_test'] = returns_mean_test

        log_data['test_mean_score'] = (SR_mean_test + SR_mean_train) / 2

        self.logger_util_train.record(SR_mean_train)
        self.logger_util_test.record(SR_mean_test)
        self.logger_util_train10.record(SR_mean_train)
        self.logger_util_test10.record(SR_mean_test)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()
        

        cprint( f"Mean SR train: {SR_mean_train:.3f}, Mean SR test: {SR_mean_test:.3f}", 'green')

        # visualize sim
        videos_test = env_test.env.get_video()
        videos_train = env_train.env.get_video()

        if len(videos_test.shape) == 5:
            videos_test = videos_test[:, 0]  # select first frame
        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train
        sim_video_test = wandb.Video(videos_test, fps=self.fps, format="mp4")
        log_data[f'sim_video_test'] = sim_video_test

        # clear out video buffer
        _ = env_train.reset()
        _ = env_test.reset()
        videos_test = None
        videos_train = None
        del env_train
        del env_test

        return log_data
