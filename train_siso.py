import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import wandb
from wandb.integration.sb3 import WandbCallback

from environments import EnvLoadRL, EnvLoadRLConst

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# set up matplotlib
# plt.ion()
plt.show()

env_const = False           # Constant reference environment
train, test = (False, True) # Training and/or Testing

sys_params_dict = {"dt": 1 / 10e3, "r": 1, "l": 1e-2, "vdc": 500}
if env_const:
    max_episode_steps = 500
    max_episodes = 200
    env = EnvLoadRLConst(sys_params=sys_params_dict)

else:
    max_episode_steps = 750
    max_episodes = 1000
    env = EnvLoadRL(sys_params=sys_params_dict)

env = gym.wrappers.TimeLimit(env, max_episode_steps)
env = gym.wrappers.RecordEpisodeStatistics(env)

# Wandb
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": max_episodes*max_episode_steps,
    "env_name": "RL Constant Reference" if env_const else "RL Variable Reference",
}
# run = wandb.init(
#     project="sb3",
#     config=config,
#     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#     save_code=True,  # optional
# )

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, tensorboard_log=f"runs/ddpg")
vec_env = model.get_env()

if train:
    model.learn(total_timesteps=config["total_timesteps"], log_interval=10, progress_bar=True)
    model.save("ddpg_EnvLoadRLConst" if env_const else "ddpg_EnvLoadRL")
if test:
    model = DDPG.load("ddpg_EnvLoadRLConst" if env_const else "ddpg_EnvLoadRL")

obs = vec_env.reset()

test_max_episodes = 10
for episode in range(test_max_episodes):
    action_list = []
    reward_list = []
    state_list  = [obs[0]]

    # plt.figure(episode, figsize=(30, 5))
    plt.figure(episode)
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        if not done:
            action_list.append(action[0][0])
            state_list.append(obs[0])
            reward_list.append(rewards[0])

    plt.clf()
    # Plot State
    plt.subplot(131)
    plt.title("State vs step")
    plt.plot(state_list)
    # Plot action
    plt.subplot(132)
    plt.title("Action vs step")
    plt.plot(action_list)
    # Plot reward
    plt.subplot(133)
    plt.title("Reward vs step")
    plt.plot(reward_list)

    plt.pause(0.001)  # pause a bit so that plots are updated
