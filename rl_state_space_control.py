import gymnasium as gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
import csv

import wandb
import argparse

from environments import *
from utils import Metrics, RewardLoggingCallback

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

# CLI Input
choices = {"env_name": ['LoadRL', 'Load3RL', 'PMSM', 'PMSMTC', 'PMSMDataBased'],
           "reward_function": ['absolute', 'quadratic', 'quadratic_2', 'square_root', 'square_root_2',
                             'quartic_root', 'quartic_root_2'],
            "error_type": ['absolute', 'relative', 'relative_imax']}
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", nargs='?', type=str, default="LoadRL",
                    choices=choices["env_name"], help='Environment name')
parser.add_argument("--reward_function", nargs='?', type=str, default="quadratic",
                    choices=choices["reward_function"], help='Reward function type')
parser.add_argument("--job_id", nargs='?', type=str, default="")
parser.add_argument("--train", action=argparse.BooleanOptionalAction)
parser.add_argument("--test_1", action=argparse.BooleanOptionalAction)
parser.add_argument("--test_2", action=argparse.BooleanOptionalAction)
parser.add_argument("--test_3", action=argparse.BooleanOptionalAction)
parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
parser.add_argument("--error_type", nargs='?', type=str, default="absolute",
                    choices=choices["error_type"], help='Error type for Test 2')
parser.add_argument("--plot", action=argparse.BooleanOptionalAction)


env_name        = parser.parse_args().env_name
reward_function = parser.parse_args().reward_function
job_id          = parser.parse_args().job_id
train           = parser.parse_args().train
test_1          = parser.parse_args().test_1
test_2          = parser.parse_args().test_2
test_3          = parser.parse_args().test_3
error_type      = parser.parse_args().error_type
plot_figs       = parser.parse_args().plot
wandb_init      = parser.parse_args().wandb 

# set up matplotlib
# plt.ion()
if env_name == "LoadRL":
    if reward_function not in ["absolute", "quadratic", "square_root", "quartic_root"]:
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
elif env_name == "PMSM" or "PMSMTC":
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "p": 4,              # Pair of poles
                       "r": 29.0808e-3,     # Resistance [Ohm]
                       "ld": 0.91e-3,       # Inductance d-frame [H]
                       "lq": 1.17e-3,       # Inductance q-frame [H]
                       "lambda_PM": 0.172312604, # Flux-linkage due to permanent magnets [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 300,            # Maximum current [A]
                       "te_max": 200,            # Maximum torque [Nm]
                       }
elif env_name == "PMSMDataBased":
    # Rows = Id / Columns = Iq
    pmsm_data = scp.io.loadmat("look_up_table_based_pmsm_prius_motor_data.mat", spmatrix=False)
    ldd = pmsm_data['Lmidd']
    ldq = pmsm_data['Lmidq']
    lqq = pmsm_data['Lmiqq']
    psid = pmsm_data['Psid']
    psiq = pmsm_data['Psiq']
    id = pmsm_data['imd'].flatten()
    iq = pmsm_data['imq'].flatten()
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 0.015,     # Resistance [Ohm]
                       "id": id,       # Current vector d-frame [A]
                       "iq": iq,       # Cirrene vector d-frame [A]
                       "ldd": ldd,       # Self-inductance matrix d-frame [H]
                       "ldq": ldq,       # Cross-coupling inductance matrix dq-frame [H]
                       "lqq": lqq,       # Self-inductance matrix q-frame [H]
                       "lss": 0.0001,       # Leakage inductance [H]
                       "psid": psid,       # Flux-linkage matrix d-frame [Wb]
                       "psiq": psiq,       # Flux-linkage matrix q-frame [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 150,            # Maximum current [A]
                       }
else:
    raise NotImplementedError
    # sys.exit("Environment name not existant")

environments = {"LoadRL": {"env": EnvLoadRL,
                    "name": f"Single Phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 200,
                    "max_episodes": 200,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvLoadRL_{reward_function}"
                    },
                "Load3RL": {"env": EnvLoad3RL,
                    "name": f"Three-phase RL system / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 200,
                    "max_episodes": 300,
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvLoad3RL_{reward_function}"
                    },
                "PMSM": {"env": EnvPMSM,
                    "name": f"PMSM / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 200,
                    "max_episodes": 1_000, # 3_000 10_000
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvPMSM_{reward_function}"
                    },
                "PMSMTC": {"env": EnvPMSMTC,
                    "name": f"PMSM torque control / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 200,
                    "max_episodes": 3_000, # 10_000
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvPMSMTC_{reward_function}"
                    },
                "PMSMDataBased": {"env": EnvPMSMDataBased,
                    "name": f"PMSM data based / Delta Vdq penalty / Reward {reward_function}",
                    "max_episode_steps": 200,
                    "max_episodes": 500, # 10_000 
                    "reward": reward_function,
                    "model_name": f"ddpg_EnvPMSMDataBased_{reward_function}"
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

# Function to store the reward in a csv file
reward_callback = RewardLoggingCallback(csv_file=f"train_{env_name}_{reward_function}.csv", log_interval=25)

model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, tensorboard_log=f"runs/ddpg")
vec_env = model.get_env()

if train:
    # Wandb
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": env_sel["max_episodes"] * env_sel["max_episode_steps"],
        "env_name": env_sel["name"],
    }
    if wandb_init:
        run = wandb.init(
            project="sb3",
            name=env_sel["model_name"] + "_" + job_id,
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
        )
    print(f"Training: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")

    model.learn(total_timesteps=config["total_timesteps"], log_interval=25, callback=reward_callback) # log_interval=10, progress_bar=True
    reward_callback._save_to_csv()
    model.save(os.path.join('weights/',env_sel["model_name"]))

if test_1:
    metrics = Metrics(dt=sys_params_dict["dt"])
    plot = PlotTest()

    test_max_episodes = 5   # 300

    settling_steps_arr = np.zeros(test_max_episodes)
    settling_time_arr  = np.zeros(test_max_episodes)
    overshoot_id_arr   = np.zeros(test_max_episodes)
    overshoot_iq_arr   = np.zeros(test_max_episodes)
    undershoot_id_arr  = np.zeros(test_max_episodes)
    undershoot_iq_arr  = np.zeros(test_max_episodes)
    ss_error_id_arr    = np.zeros(test_max_episodes)
    ss_error_iq_arr    = np.zeros(test_max_episodes)
    ss_error_id_arr_imax = np.zeros(test_max_episodes)
    ss_error_iq_arr_imax = np.zeros(test_max_episodes)

    # Test a single reward_function
    print(f"Test 1 in environment: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")

    model = DDPG.load(os.path.join('weights/',env_sel["model_name"]), print_system_info=False)

    high_error_episodes = []
    for episode in range(test_max_episodes):
        vec_env.seed(seed=episode) # Just once out of the for loop
        vec_env.set_options({"id_norm": 0.8563502430915833, 
                             "iq_norm": -0.342145174741745,})
        obs = vec_env.reset()

        action_list = []
        reward_list = []
        state_list  = [obs[0][0:2] if env_name == "LoadRL" else obs[0][0:4]]

        # plt.figure(episode, figsize=(10, 6))
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            if not done:
                action_list.append(action[0])
                state_list.append(obs[0][0:2] if env_name == "LoadRL" else obs[0][0:4]) # Don't save prev_V
                reward_list.append(rewards[0])

        # Transform lists to numpy arrays
        action_list = np.array(action_list)
        state_list  = np.array(state_list)
        reward_list = np.array(reward_list)

        # Save metrics
        id_data = state_list[:,0]
        iq_data = state_list[:,1]
        id_ref  = state_list[0,2]
        iq_ref  = state_list[0,3]
        settling_steps_id, ss_val_id = metrics.settling_time(id_data)
        settling_steps_iq, ss_val_iq = metrics.settling_time(iq_data)
        settling_steps_arr[episode] = np.max([settling_steps_id, settling_steps_iq])
        settling_time_arr[episode]  = sys_params_dict["dt"]*settling_steps_arr[episode]
        overshoot_id_arr[episode]   = metrics.overshoot(id_data, ss_val_id)
        overshoot_iq_arr[episode]   = metrics.overshoot(iq_data, ss_val_iq)
        undershoot_id_arr[episode]  = metrics.undershoot(id_data, ss_val_id)
        undershoot_iq_arr[episode]  = metrics.undershoot(iq_data, ss_val_iq)
        error_id = metrics.error(id_ref, ss_val_id)
        error_iq = metrics.error(id_ref, ss_val_id)
        ss_error_id_arr[episode]    = error_id / np.abs(id_ref)
        ss_error_iq_arr[episode]    = error_iq / np.abs(iq_ref)
        ss_error_id_arr_imax[episode] = error_id / sys_params_dict["i_max"]
        ss_error_iq_arr_imax[episode] = error_iq / sys_params_dict["i_max"]

        if ss_error_id_arr[episode] > 0.1 or ss_error_iq_arr[episode] > 0.1:
            high_error_episodes.append(f"Error in episode {episode}: Id = {100*ss_error_id_arr[episode]:.2f} % / Iq = {100*ss_error_iq_arr[episode]:.2f} %")
        # print(f"Overshoot id in episode {episode}: {100*overshoot_id_arr[episode]:.2f} %")
        # print(f"Overshoot iq in episode {episode}: {100*overshoot_iq_arr[episode]:.2f} %")
        # print(f"Undershoot id in episode {episode}: {100*undershoot_id_arr[episode]:.2f} %")
        # print(f"Undershoot iq in episode {episode}: {100*undershoot_iq_arr[episode]:.2f} %")
        # print(f"Settling time in episode {episode:d}: {settling_steps_arr[episode]:.0f} [steps] * {1e3*sys_params_dict["dt"]:.2f} [ms/step] = " +
        #       f"{1e3*settling_time_arr[episode]:.2f} [ms]")

        # Plot and save figures
        if plot_figs:
            if env_name == "LoadRL":
                plot.plot_single_phase(episode, state_list, action_list, reward_list,
                                    env_name, env_sel['model_name'], env_sel['reward'], show=True)
            else:
                plot.plot_three_phase(episode, state_list, action_list, reward_list,
                                    env_name, env_sel['model_name'], env_sel['reward'], sys_params_dict['we_nom'] * obs[0][4], save=True, show=True)
    
    # Average metrics
    with open(f"{env_name}_metrics.txt", 'a') as f:
        print(f"Model: {env_name} - Reward function: {reward_function} - Episodes: {test_max_episodes}", file=f)
        print(f"Average settling steps: {np.mean(settling_steps_arr):.2f} [steps]", file=f)
        print(f"Average settling time: {1e3*np.mean(settling_time_arr):.2f} [ms]", file=f)
        print(f"Average overshoot id: {100*np.mean(overshoot_id_arr):.2f} %", file=f)
        print(f"Average overshoot iq: {100*np.mean(overshoot_iq_arr):.2f} %", file=f)
        print(f"Average undershoot id: {100*np.mean(undershoot_id_arr):.2f} %", file=f)
        print(f"Average undershoot iq: {100*np.mean(undershoot_iq_arr):.2f} %", file=f)
        print(f"Average steady-state error id / reference: {100*np.mean(ss_error_id_arr):.2f} %", file=f)
        print(f"Average steady-state error iq / reference: {100*np.mean(ss_error_iq_arr):.2f} %", file=f)
        print(f"Average steady-state error id / Imax: {100*np.mean(ss_error_id_arr_imax):.2f} %", file=f)
        print(f"Average steady-state error iq / Imax: {100*np.mean(ss_error_iq_arr_imax):.2f} %", file=f)
        [print(high_error_episode, file=f) for high_error_episode in high_error_episodes]

if test_2:
    # Wandb
    if wandb_init:
        run = wandb.init(
            project="sb3",
            name=f"Test 2: {env_name} {reward_function} / {job_id}",
            save_code=True,  # optional
        )

    # Test a single reward_function
    print(f"Test 2 in environment: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")

    model = DDPG.load(os.path.join('weights/',env_sel["model_name"]), print_system_info=False)

    current_options = { "id_norm": 0, 
                        "iq_norm": 0,
                        "id_ref_norm": 0,
                        "iq_ref_norm": 0}
    voltage_options = {"prev_vd_norm": 0, 
                       "prev_vq_norm": 0}

    seed = 0
    vec_env.seed(seed=seed)

    sim_steps = 100
    speed_steps = 10
    current_steps = 20

    speed_norm_array  = np.linspace(0, 0.9, num = speed_steps)
    id_norm_array     = np.linspace(-0.8, 0.9, num = current_steps)
    iq_norm_array     = np.linspace(-0.8, 0.9, num = current_steps)
    
    # Idref test
    id_ref_norm_array = np.linspace(-0.5, 0.5, num = current_steps)
    
    error_data_id_ref = np.zeros((current_steps, speed_steps*(current_steps**2)))
    for id_idx, current_options["id_ref_norm"] in enumerate(tqdm(id_ref_norm_array, total=len(id_ref_norm_array), desc="Idref", leave=True)):
        max_iq = np.sqrt(1 - current_options["id_ref_norm"]**2)
        iq_ref_norm_array = np.linspace(-max_iq, max_iq, num = current_steps)
        inner_loop_combinations = [(id, iq_ref, iq_ref, speed)  for id in id_norm_array 
                                                                for iq_ref in iq_ref_norm_array
                                                                for speed  in speed_norm_array]
        error_id_ref = []
        for current_options["id_norm"], current_options["iq_norm"], current_options["iq_ref_norm"], speed_norm in tqdm(inner_loop_combinations, total=len(inner_loop_combinations), desc="Combinations", leave=False):
                # If the reference point is outside of the circle, pass
                if ((current_options["id_ref_norm"]**2 + current_options["iq_ref_norm"]**2 >= 1) or 
                    (current_options["id_norm"]**2     + current_options["iq_norm"]**2     >= 1)    ):
                    error_id_ref.append(-10)
                    continue
                options = current_options | {"we_norm": speed_norm} | voltage_options
                vec_env.set_options(options)
                obs = vec_env.reset()

                id = []
                done = False
                for i in np.arange(sim_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = vec_env.step(action)
                    if not done:
                        id.append(obs.flatten()[0])

                id_ss, id_ref = (np.mean(id[-10:]), obs.flatten()[2])
                # error_id_ref.append((id_ref - id_ss) / id_ref)
                error_id_ref.append(id_ref - id_ss)
                if np.abs(error_id_ref[-1]) >= 0.1:
                    print(f"Id0: {current_options['id_norm']} / " +
                          f"Iq0: {current_options['iq_norm']} / " +
                          f"Idref: {current_options['id_ref_norm']} / " +
                          f"Iqref: {current_options['iq_ref_norm']} / " +
                          f"we: {speed_norm} / " +
                          f"error: {error_id_ref[-1]}")

            
        error_data_id_ref[id_idx] = error_id_ref
    
    # Save values
    with open(f'test_{env_name}_{reward_function}_Id_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Id_ref = {idref}" for idref in id_ref_norm_array])
        writer.writerows(error_data_id_ref.transpose())

     # Iqref test
    iq_ref_norm_array = np.linspace(-0.5, 0.5, num = current_steps)

    error_data_iq_ref = np.zeros((current_steps, speed_steps*(current_steps**2)))
    for iq_idx, current_options["iq_ref_norm"] in enumerate(tqdm(iq_ref_norm_array, total=len(iq_ref_norm_array), desc="Iqref", leave=True)):
        max_id = np.sqrt(1 - current_options["iq_ref_norm"]**2)
        id_ref_norm_array = np.linspace(-max_id, max_id, num = current_steps)
        inner_loop_combinations = [(idref, iq, idref, speed)  for idref in id_ref_norm_array 
                                                              for iq    in iq_norm_array
                                                              for speed  in speed_norm_array]
        error_iq_ref = []
        for current_options["id_norm"], current_options["iq_norm"], current_options["id_ref_norm"], speed_norm in tqdm(inner_loop_combinations, total=len(inner_loop_combinations), desc="Combinations", leave=False):
                if ((current_options["id_ref_norm"]**2 + current_options["iq_ref_norm"]**2 >= 1) or 
                    (current_options["id_norm"]**2     + current_options["iq_norm"]**2     >= 1)    ):
                    error_iq_ref.append(-10)
                    print(f"Id_ref = {current_options["id_ref_norm"]} / Iq_ref = {current_options["iq_ref_norm"]}")
                    print(f"Id = {current_options["id_norm"]} / Iq = {current_options["iq_norm"]}")
                    continue
                options = current_options | {"we_norm": speed_norm} | voltage_options
                vec_env.set_options(options)
                obs = vec_env.reset()

                iq = []
                done = False
                for i in np.arange(sim_steps):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = vec_env.step(action)

                    iq.append(obs.flatten()[1])

                iq_ss, iq_ref = (np.mean(iq[-10:]), obs.flatten()[3])
                error_iq_ref.append(iq_ref - iq_ss)

                if np.abs(error_iq_ref[-1]) >= 0.1:
                    print(f"Id0: {current_options['id_norm']} / " +
                          f"Iq0: {current_options['iq_norm']} / " +
                          f"Idref: {current_options['id_ref_norm']} / " +
                          f"Iqref: {current_options['iq_ref_norm']} / " +
                          f"we: {speed_norm} / " +
                          f"error: {error_iq_ref[-1]}")
                    
            
        error_data_iq_ref[iq_idx] = error_iq_ref
    
    # Save values
    with open(f'test_data/test_{env_name}_{reward_function}_Iq_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Iq_ref = {iqref}" for iqref in iq_ref_norm_array])
        writer.writerows(error_data_iq_ref.transpose())


    if plot_figs:
        plot = PlotTest()

        mean = np.mean(error_data_id_ref, axis=1)
        median = np.median(error_data_id_ref, axis=1)
        std = np.std(error_data_id_ref, axis=1)
        min = np.min(error_data_id_ref, axis=1)
        max = np.max(error_data_id_ref, axis=1)
        first_quartile = np.percentile(error_data_id_ref, 25, axis=1)
        third_quartile = np.percentile(error_data_id_ref, 75, axis=1)

        # labels = [f"[{id_ref_norm_array[idx]:.2f}, {id_ref_norm_array[idx + 1]:.2f}[" for idx in np.arange(len(id_ref_norm_array[:-1]))]
        labels = [f"{sys_params_dict['i_max']*idref:.2f}" for idref in id_ref_norm_array]

        fig, ax = plt.subplots()
        ax.set_ylabel('Tracking error [%]')
        ax.set_xlabel('Id_ref [A]')
        bplot = ax.boxplot(error_data_id_ref.transpose(),
                        tick_labels=labels,# will be used to label x-ticks
                        meanline=True,
                        showmeans=True) 
        plt.show()

if test_3:
    ## REWRITE TEST 3 AND 2 FOR TORQUE CONTROL
    plot = PlotTest()
    # Wandb
    if wandb_init:
        run = wandb.init(
            project="sb3",
            name=f"Test 3: {env_name} {reward_function} / {job_id}",
            save_code=True,  # optional
        )

    # Test a single reward_function
    print(f"Test 3 in environment: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")

    model = DDPG.load(os.path.join('weights/',env_sel["model_name"]), print_system_info=False)

    current_options = { "id_norm": 0, 
                        "iq_norm": 0,
                        "id_ref_norm": 0,
                        "iq_ref_norm": 0}
    voltage_options = {"prev_vd_norm": 0, 
                       "prev_vq_norm": 0}


    show = False

    episodes = 10_000 # 10_000
    if env_name in ["PMSM, PMSMDataBased"]:
        # Observation
        # [id, iq, idref, iqref]
        error_data_id_ref = np.zeros((episodes,1))
        error_data_iq_ref = np.zeros((episodes,1))
        id0_norm_array = np.zeros((episodes,1))
        iq0_norm_array = np.zeros((episodes,1))
        id_ref_norm_array = np.zeros((episodes,1))
        iq_ref_norm_array = np.zeros((episodes,1))
        for episode in tqdm(range(episodes)):
            vec_env.seed(seed=episode)
            obs = vec_env.reset()
            id0, iq0 = (obs.flatten()[0], obs.flatten()[1])
            id_ref, iq_ref = (obs.flatten()[2], obs.flatten()[3])

            action_list = []
            state_list = []
            reward_list = []

            id = []
            iq = []
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = vec_env.step(action)
                if not done:
                    action_list.append(action.flatten())
                    state_list.append(obs.flatten()[0:4]) # Don't save prev_V
                    reward_list.append(rewards.flatten())
                    id.append(obs.flatten()[0])
                    iq.append(obs.flatten()[1])
            
            id_ss,  iq_ss  = (np.mean(id[-20:]), np.mean(iq[-20:]))
            
            if show:
                plot.plot_three_phase(episode, state_list, action_list, reward_list,
                                            env_name, env_sel['model_name'], env_sel['reward'], sys_params_dict['we_nom'] * obs[0][4], show=True)
            
            id0_norm_array[episode]    = id0
            iq0_norm_array[episode]    = iq0
            id_ref_norm_array[episode] = id_ref
            iq_ref_norm_array[episode] = iq_ref
            error_data_id_ref[episode] = id_ref - id_ss
            error_data_iq_ref[episode] = iq_ref - iq_ss
        
        data = np.hstack((id_ref_norm_array, iq_ref_norm_array, 
                        id0_norm_array, iq0_norm_array,
                        error_data_id_ref, error_data_iq_ref))

        with open(f'test_data/test_{env_name}_{reward_function}_Idq_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Id_ref", "Iq_ref", "Id0", "Iq0", "Id_error", "Iq_error"])
            writer.writerows(data)
    elif env_name in ["PMSMTC"]:
        # Observation
        # [te, te_ref, id, iq]
        error_data_te = np.zeros((episodes,1))
        id0_norm_array = np.zeros((episodes,1))
        iq0_norm_array = np.zeros((episodes,1))
        te_0_norm_array = np.zeros((episodes,1))
        te_ref_norm_array = np.zeros((episodes,1))
        for episode in tqdm(range(episodes)):
            vec_env.seed(seed=episode)
            obs = vec_env.reset()
            te0, te_ref = (obs.flatten()[0], obs.flatten()[1])
            id0, iq0 = (obs.flatten()[2], obs.flatten()[3])

            action_list = []
            state_list = []
            reward_list = []

            te = []
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = vec_env.step(action)
                if not done:
                    action_list.append(action.flatten())
                    state_list.append(obs.flatten()[0:4]) # Don't save prev_V
                    reward_list.append(rewards.flatten())
                    te.append(obs.flatten()[0])
            
            te_ss = np.mean(te[-20:])
            
            if show:
                plot.plot_three_phase(episode, state_list, action_list, reward_list,
                                            env_name, env_sel['model_name'], env_sel['reward'], sys_params_dict['we_nom'] * obs[0][4], show=True)
            
            id0_norm_array[episode]    = id0
            iq0_norm_array[episode]    = iq0
            te_0_norm_array[episode]   = te0
            te_ref_norm_array[episode] = te_ref
            error_data_te[episode]     = te_ref - te_ss
        
        data = np.hstack((te_0_norm_array, te_ref_norm_array, 
                        id0_norm_array, iq0_norm_array,
                        error_data_te))

        with open(f'test_data/test_{env_name}_{reward_function}_Te_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Te0", "Teref", "Id0", "Iq0", "Te_error"])
            writer.writerows(data)
    if plot_figs:
        plot = PlotTest()

        mean = np.mean(error_data_id_ref, axis=1)
        median = np.median(error_data_id_ref, axis=1)
        std = np.std(error_data_id_ref, axis=1)
        min = np.min(error_data_id_ref, axis=1)
        max = np.max(error_data_id_ref, axis=1)
        first_quartile = np.percentile(error_data_id_ref, 25, axis=1)
        third_quartile = np.percentile(error_data_id_ref, 75, axis=1)

        # labels = [f"[{id_ref_norm_array[idx]:.2f}, {id_ref_norm_array[idx + 1]:.2f}[" for idx in np.arange(len(id_ref_norm_array[:-1]))]
        labels = [f"{sys_params_dict['i_max']*idref:.2f}" for idref in id_ref_norm_array]

        fig, ax = plt.subplots()
        ax.set_ylabel('Tracking error [%]')
        ax.set_xlabel('Id_ref [A]')
        bplot = ax.boxplot(error_data_id_ref.transpose(),
                        tick_labels=labels,# will be used to label x-ticks
                        meanline=True,
                        showmeans=True) 
        plt.show()