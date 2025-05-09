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
from environments import EnvLoadRL, EnvLoad3RL, EnvPMSM, EnvPMSMTC, EnvPMSMDataBased, EnvPMSMTCABC
from utils import PerformanceMetrics, RewardLoggingCallback, PlotUtility

def setup_environment(env_name, env_config, sys_params_dict):
    """Setup the environment and model for training or testing."""
    # Create environment based on environment name
    env_classes = {
        "LoadRL": EnvLoadRL,
        "Load3RL": EnvLoad3RL,
        "PMSM": EnvPMSM,
        "PMSMTC": EnvPMSMTC,
        "PMSMDataBased": EnvPMSMDataBased,
        "PMSMTCABC": EnvPMSMTCABC
    }
    
    # Create environment
    env = env_classes[env_name](sys_params=sys_params_dict)
    env = gym.wrappers.TimeLimit(env, env_config["max_episode_steps"])
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # Setup action noise for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    # Create DDPG model
    model = DDPG("MlpPolicy", env=env, action_noise=action_noise, verbose=1, 
                 tensorboard_log=f"runs/ddpg")
    
    # Get vectorized environment
    vec_env = model.get_env()
    
    return env, model, vec_env

def train_model(env_config, model, env_name, reward_function, job_id, use_wandb=False):
    """Train the DDPG model."""
    print(f"Training: {env_config['name']}")
    print(f"Model: {env_config['model_name']}")
    
    # Setup WandB if enabled
    if use_wandb:
        config = {
            "policy_type": "MlpPolicy",
            "total_timesteps": env_config["max_episodes"] * env_config["max_episode_steps"],
            "env_name": env_config["name"],
        }
        wandb.init(
            project="sb3",
            name=env_config["model_name"] + "_" + job_id,
            config=config,
            sync_tensorboard=True,
            save_code=True,
        )
    
    # Setup reward callback to log rewards
    reward_callback = RewardLoggingCallback(
        csv_file=f"train_{env_name}_{reward_function}.csv", 
        log_interval=25
    )
    
    # Train the model
    total_timesteps = env_config["max_episodes"] * env_config["max_episode_steps"]
    model.learn(total_timesteps=total_timesteps, log_interval=25, callback=reward_callback)
    
    # Save reward log and model weights
    reward_callback._save_to_csv()
    
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    model.save(os.path.join('weights', env_config["model_name"]))

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
    settling_steps_arr = np.zeros(test_max_episodes)
    settling_time_arr = np.zeros(test_max_episodes)
    overshoot_id_arr = np.zeros(test_max_episodes)
    overshoot_iq_arr = np.zeros(test_max_episodes)
    undershoot_id_arr = np.zeros(test_max_episodes)
    undershoot_iq_arr = np.zeros(test_max_episodes)
    ss_error_id_arr = np.zeros(test_max_episodes)
    ss_error_iq_arr = np.zeros(test_max_episodes)
    ss_error_id_arr_imax = np.zeros(test_max_episodes)
    ss_error_iq_arr_imax = np.zeros(test_max_episodes)
    
    high_error_episodes = []
    
    # Run test episodes
    for episode in range(test_max_episodes):
        vec_env.seed(seed=episode)
        # vec_env.set_options({
        #     "id_norm": 0.8563502430915833,
        #     "iq_norm": -0.342145174741745,
        # })
        obs = vec_env.reset()
        
        mtpa_id, mtpa_iq = base_env.unwrapped.mtpa()
        mtpa_id_norm = mtpa_id / sys_params_dict["i_max"]
        mtpa_iq_norm = mtpa_iq / sys_params_dict["i_max"]

        action_list = []
        reward_list = []
        state_list = [obs.flatten()[0:2] if env_name == "LoadRL" else (obs.flatten()[0:5] if env_name == "PMSMTCABC" else obs.flatten()[0:4])]
        
        # Run episode
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = vec_env.step(action)
            if not done:
                action_list.append(action[0])
                state_list.append(obs.flatten()[0:2] if env_name == "LoadRL" else (obs.flatten()[0:5] if env_name == "PMSMTCABC" else obs.flatten()[0:4]))
                reward_list.append(rewards[0])
        
        # Transform lists to numpy arrays
        action_list = np.array(action_list)
        state_list = np.array(state_list)
        reward_list = np.array(reward_list)
        
        # Calculate metrics
        id_data = state_list[:,0]
        iq_data = state_list[:,1]
        id_ref = state_list[0,2]
        iq_ref = state_list[0,3]
        
        settling_steps_id, ss_val_id = metrics.settling_time(id_data)
        settling_steps_iq, ss_val_iq = metrics.settling_time(iq_data)
        
        settling_steps_arr[episode] = np.max([settling_steps_id, settling_steps_iq])
        settling_time_arr[episode] = sys_params_dict["dt"] * settling_steps_arr[episode]
        overshoot_id_arr[episode] = metrics.overshoot(id_data, ss_val_id)
        overshoot_iq_arr[episode] = metrics.overshoot(iq_data, ss_val_iq)
        undershoot_id_arr[episode] = metrics.undershoot(id_data, ss_val_id)
        undershoot_iq_arr[episode] = metrics.undershoot(iq_data, ss_val_iq)
        
        error_id = metrics.error(id_ref, ss_val_id)
        error_iq = metrics.error(id_ref, ss_val_id)
        
        ss_error_id_arr[episode] = error_id / np.abs(id_ref)
        ss_error_iq_arr[episode] = error_iq / np.abs(iq_ref)
        ss_error_id_arr_imax[episode] = error_id / sys_params_dict["i_max"]
        ss_error_iq_arr_imax[episode] = error_iq / sys_params_dict["i_max"]
        
        # Check for high error episodes
        if ss_error_id_arr[episode] > 0.1 or ss_error_iq_arr[episode] > 0.1:
            high_error_episodes.append(
                f"Error in episode {episode}: "
                f"Id = {100*ss_error_id_arr[episode]:.2f} % / "
                f"Iq = {100*ss_error_iq_arr[episode]:.2f} %"
            )
        
        # Plot results if requested
        if plot_figs:
            if env_name == "LoadRL":
                plot.plot_single_phase(
                    episode, state_list, action_list, reward_list,
                    env_name, env_config['model_name'], env_config['reward'], show=True
                )
            else:
                plot.plot_three_phase(
                    episode, state_list, action_list, reward_list,
                    env_name, env_config['model_name'], env_config['reward'], 
                    sys_params_dict['we_nom'] * obs[0][4], save=True, show=True, mtpa=[mtpa_id_norm, mtpa_iq_norm]
                )
    
    # Save average metrics
    os.makedirs('test_data', exist_ok=True)
    with open(f"test_data/{env_name}_metrics.txt", 'a') as f:
        print(f"Model: {env_name} - Reward function: {env_config['reward']} - Episodes: {test_max_episodes}", file=f)
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
        for high_error_episode in high_error_episodes:
            print(high_error_episode, file=f)

def run_test_2(env_config, model, vec_env, sys_params_dict, env_name, reward_function, 
              job_id, error_type, use_wandb=False, plot_figs=False):
    """Run Test 2: Systematic evaluation across parameter space."""
    # Setup WandB if enabled
    if use_wandb:
        wandb.init(
            project="sb3",
            name=f"Test 2: {env_name} {reward_function} / {job_id}",
            save_code=True,
        )
    
    print(f"Test 2 in environment: {env_config['name']}")
    print(f"Model: {env_config['model_name']}")
    
    # Load the trained model
    model = DDPG.load(os.path.join('weights', env_config["model_name"]), print_system_info=False)
    
    # Setup parameters for test
    current_options = {
        "id_norm": 0,
        "iq_norm": 0,
        "id_ref_norm": 0,
        "iq_ref_norm": 0
    }
    voltage_options = {
        "prev_vd_norm": 0,
        "prev_vq_norm": 0
    }
    
    # Set seed for reproducibility
    seed = 0
    vec_env.seed(seed=seed)
    
    # Define parameter space for testing
    sim_steps = 100
    speed_steps = 10
    current_steps = 20
    
    speed_norm_array = np.linspace(0, 0.9, num=speed_steps)
    id_norm_array = np.linspace(-0.8, 0.9, num=current_steps)
    iq_norm_array = np.linspace(-0.8, 0.9, num=current_steps)
    
    # Test for Id reference variations
    id_ref_norm_array = np.linspace(-0.5, 0.5, num=current_steps)
    error_data_id_ref = np.zeros((current_steps, speed_steps*(current_steps**2)))
    
    # Run Id reference test
    for id_idx, id_ref_norm in enumerate(tqdm(id_ref_norm_array, total=len(id_ref_norm_array), desc="Idref", leave=True)):
        current_options["id_ref_norm"] = id_ref_norm
        max_iq = np.sqrt(1 - id_ref_norm**2)
        iq_ref_norm_array = np.linspace(-max_iq, max_iq, num=current_steps)
        
        inner_loop_combinations = [
            (id_norm, iq_ref, iq_ref, speed) 
            for id_norm in id_norm_array
            for iq_ref in iq_ref_norm_array
            for speed in speed_norm_array
        ]
        
        error_id_ref = []
        for id_norm, iq_norm, iq_ref_norm, speed_norm in tqdm(inner_loop_combinations, total=len(inner_loop_combinations), desc="Combinations", leave=False):
            current_options["id_norm"] = id_norm
            current_options["iq_norm"] = iq_norm
            current_options["iq_ref_norm"] = iq_ref_norm
            
            # Set options and run simulation
            options = {**current_options, "we_norm": speed_norm, **voltage_options}
            vec_env.set_options(options)
            obs = vec_env.reset()
            
            id_values = []
            done = False
            for _ in range(sim_steps):
                action, _states = model.predict(obs)
                obs, rewards, done, info = vec_env.step(action)
                if not done:
                    id_values.append(obs.flatten()[0])
            
            # Calculate error
            id_ss, id_ref = (np.mean(id_values[-10:]), obs.flatten()[2])
            error_id_ref.append(id_ref - id_ss)
            
            # Log high error cases
            if np.abs(error_id_ref[-1]) >= 0.1:
                print(f"Id0: {current_options['id_norm']} / "
                      f"Iq0: {current_options['iq_norm']} / "
                      f"Idref: {current_options['id_ref_norm']} / "
                      f"Iqref: {current_options['iq_ref_norm']} / "
                      f"we: {speed_norm} / "
                      f"error: {error_id_ref[-1]}")
        
        error_data_id_ref[id_idx] = error_id_ref
    
    # Save Id error data
    os.makedirs('test_data', exist_ok=True)
    with open(f'test_data/test_{env_name}_{reward_function}_Id_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Id_ref = {idref}" for idref in id_ref_norm_array])
        writer.writerows(error_data_id_ref.transpose())
    
    # Test for Iq reference variations
    iq_ref_norm_array = np.linspace(-0.5, 0.5, num=current_steps)
    error_data_iq_ref = np.zeros((current_steps, speed_steps*(current_steps**2)))
    
    # Run Iq reference test
    for iq_idx, iq_ref_norm in enumerate(tqdm(iq_ref_norm_array, total=len(iq_ref_norm_array), desc="Iqref", leave=True)):
        current_options["iq_ref_norm"] = iq_ref_norm
        max_id = np.sqrt(1 - iq_ref_norm**2)
        id_ref_norm_array = np.linspace(-max_id, max_id, num=current_steps)
        
        inner_loop_combinations = [
            (id_ref, iq, id_ref, speed)
            for id_ref in id_ref_norm_array
            for iq in iq_norm_array
            for speed in speed_norm_array
        ]
        
        error_iq_ref = []
        for id_norm, iq_norm, id_ref_norm, speed_norm in tqdm(inner_loop_combinations, total=len(inner_loop_combinations), desc="Combinations", leave=False):
            current_options["id_norm"] = id_norm
            current_options["iq_norm"] = iq_norm
            current_options["id_ref_norm"] = id_ref_norm
            
            # Set options and run simulation
            options = {**current_options, "we_norm": speed_norm, **voltage_options}
            vec_env.set_options(options)
            obs = vec_env.reset()
            
            iq_values = []
            done = False
            for _ in range(sim_steps):
                action, _states = model.predict(obs)
                obs, rewards, done, info = vec_env.step(action)
                iq_values.append(obs.flatten()[1])
            
            # Calculate error
            iq_ss, iq_ref = (np.mean(iq_values[-10:]), obs.flatten()[3])
            error_iq_ref.append(iq_ref - iq_ss)
            
            # Log high error cases
            if np.abs(error_iq_ref[-1]) >= 0.1:
                print(f"Id0: {current_options['id_norm']} / "
                      f"Iq0: {current_options['iq_norm']} / "
                      f"Idref: {current_options['id_ref_norm']} / "
                      f"Iqref: {current_options['iq_ref_norm']} / "
                      f"we: {speed_norm} / "
                      f"error: {error_iq_ref[-1]}")
        
        error_data_iq_ref[iq_idx] = error_iq_ref
    
    # Save Iq error data
    with open(f'test_data/test_{env_name}_{reward_function}_Iq_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f"Iq_ref = {iqref}" for iqref in iq_ref_norm_array])
        writer.writerows(error_data_iq_ref.transpose())

def run_test_3(env_name, env_sel, reward_function, model, vec_env, sys_params_dict, 
             wandb_init=False, job_id="", plot_figs=False):
    """
    Run Test 3 on the trained model across multiple episodes to evaluate performance.
    
    Args:
        env_name: Name of the environment to test
        env_sel: Dictionary containing environment configuration
        reward_function: Type of reward function used
        model: Trained RL model
        vec_env: Vectorized environment
        sys_params_dict: Dictionary of system parameters
        wandb_init: Whether to initialize wandb logging
        job_id: Job identifier
        plot_figs: Whether to generate plots
        
    Returns:
        None (saves results to CSV files)
    """
    
    # Import wandb conditionally
    if wandb_init:
        import wandb
        run = wandb.init(
            project="sb3",
            name=f"Test 3: {env_name} {reward_function} / {job_id}",
            save_code=True
        )
    
    # Initialize plotting if needed
    if plot_figs:
        from utils import PlotUtility
        plot = PlotUtility()
    else:
        plot = None
    
    print(f"Test 3 in environment: {env_sel['name']}")
    print(f"Model: {env_sel['model_name']}")
    
    # Testing options
    current_options = {
        "id_norm": 0, 
        "iq_norm": 0,
        "id_ref_norm": 0,
        "iq_ref_norm": 0
    }
    voltage_options = {
        "prev_vd_norm": 0, 
        "prev_vq_norm": 0
    }
    
    show = False
    episodes = 10_000  # Number of episodes to test
    
    # Ensure test_data directory exists
    os.makedirs('test_data', exist_ok=True)
    
    if env_name in ["PMSM", "PMSMDataBased"]:
        # Initialize arrays for storing results
        error_data_id_ref = np.zeros((episodes, 1))
        error_data_iq_ref = np.zeros((episodes, 1))
        id0_norm_array = np.zeros((episodes, 1))
        iq0_norm_array = np.zeros((episodes, 1))
        id_ref_norm_array = np.zeros((episodes, 1))
        iq_ref_norm_array = np.zeros((episodes, 1))
        
        # Run episodes
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
                    state_list.append(obs.flatten()[0:4])  # Don't save prev_V
                    reward_list.append(rewards.flatten())
                    id.append(obs.flatten()[0])
                    iq.append(obs.flatten()[1])
            
            # Calculate steady-state values
            id_ss, iq_ss = (np.mean(id[-20:]), np.mean(iq[-20:]))
            
            # Generate plots if requested
            if show and plot:
                plot.plot_three_phase(
                    episode, state_list, action_list, reward_list,
                    env_name, env_sel['model_name'], env_sel['reward'], 
                    sys_params_dict['we_nom'] * obs[0][4], show=True
                )
            
            # Store results
            id0_norm_array[episode] = id0
            iq0_norm_array[episode] = iq0
            id_ref_norm_array[episode] = id_ref
            iq_ref_norm_array[episode] = iq_ref
            error_data_id_ref[episode] = id_ref - id_ss
            error_data_iq_ref[episode] = iq_ref - iq_ss
        
        # Combine data for saving
        data = np.hstack((
            id_ref_norm_array, iq_ref_norm_array,
            id0_norm_array, iq0_norm_array,
            error_data_id_ref, error_data_iq_ref
        ))
        
        # Save results to CSV
        with open(f'test_data/test_{env_name}_{reward_function}_Idq_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Id_ref", "Iq_ref", "Id0", "Iq0", "Id_error", "Iq_error"])
            writer.writerows(data)

    elif env_name in ["PMSMTC"]:
        # Initialize arrays for torque control mode
        error_data_te = np.zeros((episodes, 1))
        id0_norm_array = np.zeros((episodes, 1))
        iq0_norm_array = np.zeros((episodes, 1))
        te_0_norm_array = np.zeros((episodes, 1))
        te_ref_norm_array = np.zeros((episodes, 1))
        
        # Run episodes
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
                    state_list.append(obs.flatten()[0:4])  # Don't save prev_V
                    reward_list.append(rewards.flatten())
                    te.append(obs.flatten()[0])
            
            # Calculate steady-state values
            te_ss = np.mean(te[-20:])
            
            # Generate plots if requested
            if show and plot:
                plot.plot_three_phase(
                    episode, state_list, action_list, reward_list,
                    env_name, env_sel['model_name'], env_sel['reward'], 
                    sys_params_dict['we_nom'] * obs[0][4], show=True
                )
            
            # Store results
            id0_norm_array[episode] = id0
            iq0_norm_array[episode] = iq0
            te_0_norm_array[episode] = te0
            te_ref_norm_array[episode] = te_ref
            error_data_te[episode] = te_ref - te_ss
        
        # Combine data for saving
        data = np.hstack((
            te_0_norm_array, te_ref_norm_array,
            id0_norm_array, iq0_norm_array,
            error_data_te
        ))
        
        # Save results to CSV
        with open(f'test_data/test_{env_name}_{reward_function}_Te_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Te0", "Teref", "Id0", "Iq0", "Te_error"])
            writer.writerows(data)