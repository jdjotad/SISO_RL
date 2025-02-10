import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import wandb
import argparse

from environments import *

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# CLI Input
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", nargs='?', type=str, default="LoadRL",
                    choices=['LoadRL', 'Load3RL', 'PMSM'], help='Environment name')
parser.add_argument("--reward_function", nargs='?', type=str, default="quadratic",
                    choices=['absolute', 'quadratic', 'quadratic_2', 'square_root', 'square_root_2',
                             'quartic_root', 'quartic_root_2'], help='Reward function type')
parser.add_argument("--job_id", nargs='?', type=str, default="")
parser.add_argument("--train", action=argparse.BooleanOptionalAction)
parser.add_argument("--test", action=argparse.BooleanOptionalAction)


env_name        = parser.parse_args().env_name
reward_function = parser.parse_args().reward_function
job_id          = parser.parse_args().job_id
train           = parser.parse_args().train
test            = parser.parse_args().test

# set up matplotlib
# plt.ion()
if env_name == "LoadRL":
    if reward_function in ["quadratic_2", "square_root_2", "quartic_root_2"]:
        sys.exit("This reward function has not been implemented for this environment")
    sys_params_dict = {"dt": 1 / 10e3,  # Sampling time [s]
                       "r": 1,          # Resistance [Ohm]
                       "l": 1e-2,       # Inductance [H]
                       "vdc": 500,      # DC bus voltage [V]
                       "i_max": 100,    # Maximum current [A]
                       }
elif env_name == "Load3RL":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 1,              # Resistance [Ohm]
                       "l": 1e-2,           # Inductance [H]
                       "vdc": 500,          # DC bus voltage [V]
                       "we_nom": 200*2*np.pi, # Nominal speed [rad/s]
                       }
    idq_max_norm = lambda vdq_max,we,r,l: vdq_max / np.sqrt(np.power(r, 2) + np.power(we * l, 2))
    # Maximum current [A]
    sys_params_dict["i_max"] = idq_max_norm(sys_params_dict["vdc"]/2, sys_params_dict["we_nom"],
                                            sys_params_dict["r"], sys_params_dict["l"])
elif env_name == "PMSM":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 29.0808e-3,     # Resistance [Ohm]
                       "ld": 0.91e-3,       # Inductance d-frame [H]
                       "lq": 1.17e-3,       # Inductance q-frame [H]
                       "lambda_PM": 0.172312604, # Flux-linkage due to permanent magnets [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 300,            # Maximum current [A]
                       }
else:
    raise NotImplementedError
    # sys.exit("Environment name not existant")

environments = {"LoadRL": {"env": EnvLoadRL,
                    "name": f"Single Phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 200,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvLoadRL_{reward_function}"
                    },
                "Load3RL": {"env": EnvLoad3RL,
                    "name": f"Three-phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvLoad3RL_{reward_function}"
                    },
                "PMSM": {"env": EnvPMSM,
                    "name": f"PMSM / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 500,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvPMSM_{reward_function}"
                    },
                }

env_sel = environments[env_name]                # Choose Environment

sys_params_dict["reward"] = env_sel["reward"]   # Store reward function type in sys_params

env = env_sel["env"](sys_params=sys_params_dict)
env = gym.wrappers.TimeLimit(env, env_sel["max_episode_steps"])
env = gym.wrappers.RecordEpisodeStatistics(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, tensorboard_log=f"runs/ddpg")
vec_env = model.get_env()

if train:
    # Wandb
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": env_sel["max_episodes"] * env_sel["max_episode_steps"],
        "env_name": env_sel["name"],
    }
    run = wandb.init(
        project="sb3",
        name=env_sel["model_name"] + "_" + job_id,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )
    print(f"Training: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")

    model.learn(total_timesteps=config["total_timesteps"], log_interval=10, progress_bar=True)
    model.save(os.path.join('weights/',env_sel["model_name"]))
if test:
    plot = PlotTest()

    print(f"Testing: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")
    model = DDPG.load(os.path.join('weights/',env_sel["model_name"]))

    test_max_episodes = 10
    for episode in range(test_max_episodes):
        obs = vec_env.reset()

        action_list = []
        reward_list = []
        state_list  = [obs[0][0:2] if env_name == "LoadRL" else obs[0][0:4]]

        # plt.figure(episode, figsize=(30, 5))
        plt.figure(episode, figsize=(10, 6))
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            if not done:
                action_list.append(action[0])
                state_list.append(obs[0][0:2] if env_name == "LoadRL" else obs[0][0:4]) # Don't save prev_V
                reward_list.append(rewards[0])
        if env_name == "LoadRL":
            plot.plot_single_phase(episode, state_list, action_list, reward_list,
                                  env_sel['model_name'], env_sel['reward'])
        else:
            plot.plot_three_phase(episode, state_list, action_list, reward_list,
                                  env_sel['model_name'], env_sel['reward'], sys_params_dict['we_nom'] * obs[0][4])