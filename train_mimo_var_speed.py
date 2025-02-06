import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

import wandb
import argparse

from environments import (EnvLoad3RLConstRefConstSpeed,
                          EnvLoad3RLConstRef,
                          EnvLoad3RLConstRefDeltaVdq,
                          EnvLoad3RLConstSpeed,
                          EnvLoad3RLConstSpeedDeltaVdq,
                          EnvLoad3RLDeltaVdq,
                          EnvPMSM)

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# CLI Input
parser = argparse.ArgumentParser()
parser.add_argument("env_number", nargs='?', type=int, const=0, default=0)
parser.add_argument("job_id", nargs='?', type=str, const="", default="")
env_number = parser.parse_args().env_number
job_id = parser.parse_args().job_id

# Training and/or Testing
train, test = (False, True)

# set up matplotlib
# plt.ion()
plt.show()

if env_number <= 5:
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 1,              # Resistance [Ohm]
                       "l": 1e-2,           # Inductance [H]
                       "vdc": 500,          # DC bus voltage [V]
                       "we_nom": 200*2*np.pi, # Nominal speed [rad/s]
                       "we_norm_const": 0,  # Constant speed value [rad/s] ; [-0.9, 0.9]
                       "id_ref_norm_const": 0,  # Constant id value [A] ; [-0.9, 0.9]
                       "iq_ref_norm_const": 0,  # Constant iq value [A] ; [-0.9, 0.9]
                       }
else:
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 29.0808e-3,     # Resistance [Ohm]
                       "ld": 0.91e-3,       # Inductance d-frame [H]
                       "lq": 1.17e-3,       # Inductance q-frame [H]
                       "lambda_PM": 0.172312604, # Flux-linkage due to permanent magnets [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 300,            # Maximum current [A]
                       "we_norm_const": 0,      # Constant speed value [rad/s] ; [-0.9, 0.9]
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
                    "max_episode_steps": 500,
                    "max_episodes": 1000,
                    "tolerance": 1e-4,
                    "model_name": "ddpg_EnvLoad3RLDeltaVdq"
                    },
                6: {"env": EnvPMSM,
                    "name": "PMSM / Delta Vdq penalty / Reward squared",
                    "max_episode_steps": 500,   # 750
                    "max_episodes": 300,       # 2500
                    "tolerance": 1e-4,
                    "reward": "squared",
                    "model_name": "ddpg_EnvPMSM_squaredreward"
                    },
                7: {"env": EnvPMSM,
                    "name": "PMSM / Delta Vdq penalty / Reward abs",
                    "max_episode_steps": 500,   # 750
                    "max_episodes": 300,       # 2500
                    "tolerance": 1e-4,
                    "reward": "abs",
                    "model_name": "ddpg_EnvPMSM_absreward"
                    },
                8: {"env": EnvPMSM,
                    "name": "PMSM / Delta Vdq penalty / Reward sqrt",
                    "max_episode_steps": 500,   # 750
                    "max_episodes": 300,       # 2500
                    "tolerance": 1e-4,
                    "reward": "sqrt",
                    "model_name": "ddpg_EnvPMSM_sqrtreward"
                    },
                }

if env_number not in environments.keys():
    print(f"Environment number not recognized, choose between {list(environments.keys())[0]} and {list(environments.keys())[-1]}")
    exit(-1)
env_sel = environments[env_number]   # Choose Environment to test

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
if train:
    run = wandb.init(
        project="sb3",
        name=env_sel["model_name"] + "_" + job_id,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

print(f"Running: {env_sel['name']}")
print(f"Model: {env_sel['model_name']}")

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, tensorboard_log=f"runs/ddpg")
vec_env = model.get_env()

if train:
    model.learn(total_timesteps=config["total_timesteps"], log_interval=10, progress_bar=True)
    model.save(env_sel["model_name"])
if test:
    model = DDPG.load(os.path.join('weights/',env_sel["model_name"]))

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
            plt.suptitle(f"Reward: {env_sel["reward"]}\nSpeed = {sys_params_dict['we_nom']*obs[0][4]} [rad/s]",)
        # Plot State
        ax = plt.subplot(131)
        ax.set_title("State vs step")
        ax.plot(state_list,label=['Id', 'Iq', 'Idref', 'Iqref'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
              ncol=2, fancybox=True, shadow=True)
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(action_list,label=['Vd', 'Vq'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward_list)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        plt.pause(0.001)  # pause a bit so that plots are updated
