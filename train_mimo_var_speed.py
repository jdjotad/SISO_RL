import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import wandb
import argparse

from environments import (EnvLoad3RLConstRefConstSpeed,
                          EnvLoad3RLConstRef,
                          EnvLoad3RLConstRefDeltaVdq,
                          EnvLoad3RLConstSpeed,
                          EnvLoad3RLConstSpeedDeltaVdq,
                          EnvLoad3RLDeltaVdq,)

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# CLI Input
parser = argparse.ArgumentParser()
parser.add_argument("env_number", nargs='?', type=int, const=0, default=0)
env_number = parser.parse_args().env_number

# Training and/or Testing
train, test = (True, True)

# set up matplotlib
# plt.ion()
plt.show()

sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                   "r": 1,              # Resistance [Ohm]
                   "l": 1e-2,           # Inductance [H]
                   "vdc": 500,          # DC bus voltage [V]
                   "we_nom": 400*np.pi, # Nominal speed
                   "we_norm_const": 0,  # Constant speed value [rad/s] ; [-0.9, 0.9]
                   "id_ref_norm_const": 0,  # Constant id value [A] ; [-0.9, 0.9]
                   "iq_ref_norm_const": 0,  # Constant iq value [A] ; [-0.9, 0.9]
                   }

environments = {0: {"env": EnvLoad3RLConstRefConstSpeed,
                    "name": "3-Phase RL Constant Reference Constant Speed",
                    "max_episode_steps": 500,
                    "max_episodes": 500,
                    "tolerance": 1e-4,
                    "model_name": "ddpg_EnvLoad3RLConstRefConstSpeed"
                    },
                1: {"env": EnvLoad3RLConstRef,
                    "name": "3-Phase RL Constant Reference",
                    "max_episode_steps": 500,
                    "max_episodes": 500,
                    "tolerance": 1e-4,
                    "model_name": "ddpg_EnvLoad3RLConstRef"
                    },
                2: {"env": EnvLoad3RLConstRefDeltaVdq,
                    "name": "3-Phase RL Constant Reference / Delta Vdq penalty",
                    "max_episode_steps": 500,
                    "max_episodes": 500,
                    "tolerance": 1e-5,
                    "model_name": "ddpg_EnvLoad3RLConstRefDeltaVdq"
                    },
                3: {"env": EnvLoad3RLConstSpeed,
                    "name": "3-Phase RL Constant Speed",
                    "max_episode_steps": 500,
                    "max_episodes": 1000,
                    "tolerance": 1e-4,
                    "model_name": "ddpg_EnvLoad3RLConstSpeed"
                    },
                4: {"env": EnvLoad3RLConstSpeedDeltaVdq,
                    "name": "3-Phase RL Constant Speed / Delta Vdq penalty",
                    "max_episode_steps": 500,
                    "max_episodes": 1000,
                    "tolerance": 1e-4,
                    "model_name": "ddpg_EnvLoad3RLConstSpeedDeltaVdq"
                    },
                5: {"env": EnvLoad3RLDeltaVdq,
                    "name": "3-Phase RL / Delta Vdq penalty",
                    "max_episode_steps": 750,
                    "max_episodes": 10000,
                    "tolerance": 1e-5,
                    "model_name": "ddpg_EnvLoad3RLDeltaVdq"
                    },
                }

if env_number not in environments.keys():
    print(f"Environment number not recognized, choose between {list(environments.keys())[0]} and {list(environments.keys())[-1]}")
    exit(-1)
env_sel = environments[env_number]   # Choose Environment to test
print(f"Running: {env_sel['name']}")
print(f"Model: {env_sel['model_name']}")

sys_params_dict["tolerance"] = env_sel["tolerance"]     # Store tolerance value in sys_params

env = env_sel["env"](sys_params=sys_params_dict)
env = gym.wrappers.TimeLimit(env, env_sel["max_episode_steps"])
env = gym.wrappers.RecordEpisodeStatistics(env)

# Wandb
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": env_sel["max_episodes"]*env_sel["max_episode_steps"],
    "env_name": env_sel["name"],
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, tensorboard_log=f"runs/ddpg")
vec_env = model.get_env()

if train:
    model.learn(total_timesteps=config["total_timesteps"], log_interval=10, progress_bar=True)
    model.save(env_sel["model_name"])
if test:
    model = DDPG.load(env_sel["model_name"])

test_max_episodes = 10
for episode in range(test_max_episodes):
    obs = vec_env.reset()

    action_list = []
    reward_list = []
    state_list  = [obs[0][0:4] if len(obs[0]) > 4 else obs[0]]

    # plt.figure(episode, figsize=(30, 5))
    plt.figure(episode)
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        if not done:
            action_list.append(action[0])
            state_list.append(obs[0][0:4] if len(obs[0]) > 4 else obs[0])
            reward_list.append(rewards[0])

    plt.clf()
    if len(obs[0]) > 4:
        plt.suptitle(f"Speed = {sys_params_dict['we_nom']*obs[0][4]} [rad/s]",)
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
