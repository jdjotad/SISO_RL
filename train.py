import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from environments import EnvLoadRL, EnvLoadRLConst

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# set up matplotlib
# plt.ion()
plt.show()

sys_params_dict = {"dt": 1 / 10e3, "r": 1, "l": 1e-2, "vdc": 500}
env_const = True
if env_const:
    max_episode_steps = 500
    max_episodes = 200
    env = EnvLoadRLConst(sys_params=sys_params_dict)
else:
    max_episode_steps = 1000
    max_episodes = 1000
    env = EnvLoadRL(sys_params=sys_params_dict)

env = gym.wrappers.TimeLimit(env, max_episode_steps)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, device="cpu")
vec_env = model.get_env()

train, test = (True, True)
if train:
    model.learn(total_timesteps=max_episodes*max_episode_steps, log_interval=10, progress_bar=True)
    model.save("ddpg_EnvLoadRL")
if test:
    model = DDPG.load("ddpg_EnvLoadRL")

obs = vec_env.reset()

test_max_episodes = 10
for episode in range(test_max_episodes):
    action_list = []
    state_list  = [obs[0][0]]

    # plt.figure(episode, figsize=(30, 5))
    plt.figure(episode)
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        if not done:
            action_list.append(action[0][0])
            state_list.append(obs[0][0])

    plt.clf()
    # Plot State
    plt.subplot(121)
    plt.title("State vs step")
    plt.plot(state_list)
    # Plot action
    plt.subplot(122)
    plt.title("Action vs step")
    plt.plot(action_list)

    plt.pause(0.001)  # pause a bit so that plots are updated
