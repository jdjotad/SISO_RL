import os
import csv
import numpy as np
import gymnasium as gym
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

# Import your environment modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments import EnvLoadRL, EnvLoad3RL, EnvPMSM, EnvPMSMTC, EnvPMSMDataBased, EnvPMSMTCABC
from utils_local import PerformanceMetrics, RewardLoggingCallback, PlotUtility
from algorithms import setup_environment

def run_test_1(env_config, model, vec_env, sys_params_dict, env_name, plot_figs=False):
    """Run Test 1: Basic model performance evaluation."""
    metrics = PerformanceMetrics(dt=sys_params_dict["dt"])
    plot = PlotUtility()
    
    print(f"Test 1 in environment: {env_config['name']}")
    print(f"Model: {env_config['model_name']}")
    
    # Load the trained model
    model = DDPG.load(os.path.join('weights', env_config["model_name"]), print_system_info=False)
    
    # Get the base environment (unwrapping the vectorized environment)
    base_env = vec_env.unwrapped.envs[0] if hasattr(vec_env, 'unwrapped') else vec_env

    test_max_episodes = 5  # Can be increased for more thorough testing
    
    # Initialize arrays for metrics
    ss_error_te_array = np.zeros(test_max_episodes)
    ss_error_mtpa_array = np.zeros((test_max_episodes, 2))
    
    high_error_episodes = []

    # Set seed for reproducibility
    seed = 0
    vec_env.seed(seed=seed)

    # Run test episodes
    for episode in range(test_max_episodes):
        obs = vec_env.reset()
        
        mtpa_id, mtpa_iq = base_env.unwrapped.mtpa()
        mtpa_id_norm = mtpa_id / sys_params_dict["i_max"]
        mtpa_iq_norm = mtpa_iq / sys_params_dict["i_max"]

        action_list = []
        reward_list = []
        state_list = [obs.flatten()[0:4]] # te, te_ref, id, iq
        
        te0, te_ref0, id0, iq0, we0, vd0, vq0 = obs.flatten()
        # Run episode
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            if not done:
                action_list.append(action[0])
                state_list.append(obs.flatten()[0:4]) # te, te_ref, id, iq
                reward_list.append(rewards[0])
        
        # Transform lists to numpy arrays
        action_list = np.array(action_list)
        state_list = np.array(state_list)
        reward_list = np.array(reward_list)
        
        # Calculate metrics
        te_ss_val = np.mean(state_list[-20:,0])
        id_ss_val = np.mean(state_list[-20:,2])
        iq_ss_val = np.mean(state_list[-20:,3])
        te_ref = state_list[-1,1]
        
        ss_error_te_array[episode] = np.abs(te_ss_val - te_ref) / te_ref
        ss_error_mtpa_array[episode] = [np.abs(id_ss_val - mtpa_id_norm) / mtpa_id_norm, np.abs(iq_ss_val - mtpa_iq_norm) / mtpa_iq_norm]
        
        print(f"Episode {episode}: Te_error = {100*ss_error_te_array[episode]:.2f}%, "
              f"Id_error = {100*ss_error_mtpa_array[episode][0]:.2f}%, "
              f"Iq_error = {100*ss_error_mtpa_array[episode][1]:.2f}%")
        # Plot results if requested
        if plot_figs:
            plot.plot_three_phase(
                episode, state_list, action_list, reward_list,
                env_name, env_config['model_name'], env_config['reward'], 
                sys_params_dict['we_nom'] * obs[0][4], save=True, show=True, mtpa=[mtpa_id_norm, mtpa_iq_norm])
            

if __name__ == "__main__":
    env_config = {
            "name": f"PMSM torque control / Delta Vdq penalty / Reward absolute",
            "max_episode_steps": 500, # 200
            "max_episodes": 2_000, # 1000
            "reward": "absolute",
            "model_name": f"ddpg_EnvPMSMTC_widq_0_1_wvdq_0_1_absolute"
    }
    # Setup the environment and model
    sys_params_dict = {
            "dt": 1 / 10e3,      # Sampling time [s]
            "p": 4,              # Pair of poles
            "r": 29.0808e-3,     # Resistance [Ohm]
            "ld": 0.91e-3,       # Inductance d-frame [H]
            "lq": 1.17e-3,       # Inductance q-frame [H]
            "lambda_PM": 0.172312604, # Flux-linkage due to permanent magnets [Wb]
            "vdc": 1200,             # DC bus voltage [V]
            "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
            "i_max": 300,            # Maximum current [A]
            "te_max": 200,           # Maximum torque [Nm]
        }
    sys_params_dict["reward"] = env_config["reward"]   # Store reward function type in sys_params

    env, model, vec_env = setup_environment("PMSMTC", env_config, sys_params_dict, 0.1, 0.1)

    run_test_1(env_config, model, vec_env, sys_params_dict, "PMSMTC", plot_figs=True)