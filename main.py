import os
import argparse
import numpy as np
from algorithms import (
    setup_environment,
    train_model,
    run_test_1,
    run_test_2,
    run_test_3
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_arguments():
    """Parse command line arguments."""
    # Define valid choices for arguments
    choices = {
        "env_name": ['LoadRL', 'Load3RL', 'PMSM', 'PMSMTC', 'PMSMDataBased', 'PMSMTCABC'],
        "reward_function": [
            'absolute', 'quadratic', 'quadratic_2', 'square_root', 
            'square_root_2', 'quartic_root', 'quartic_root_2'
        ],
        "error_type": ['absolute', 'relative', 'relative_imax']
    }
    
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
    
    return parser.parse_args()

def configure_system_parameters(env_name, reward_function):
    """Configure system parameters based on environment name."""
    sys_params_dict = {}
    
    if env_name == "LoadRL":
        if reward_function not in ["absolute", "quadratic", "square_root", "quartic_root"]:
            raise ValueError("This reward function has not been implemented for this environment")
        sys_params_dict = {
            "dt": 1 / 10e3,  # Sampling time [s]
            "r": 1,          # Resistance [Ohm]
            "l": 1e-2,       # Inductance [H]
            "vdc": 500,      # DC bus voltage [V]
            "i_max": 100,    # Maximum current [A]
        }
    elif env_name == "Load3RL":
        sys_params_dict = {
            "dt": 1 / 10e3,      # Sampling time [s]
            "r": 1,              # Resistance [Ohm]
            "l": 1e-2,           # Inductance [H]
            "vdc": 500,          # DC bus voltage [V]
            "we_nom": 200*2*np.pi, # Nominal speed [rad/s]
        }
        # Maximum current [A]
        idq_max_norm = lambda vdq_max, we, r, l: vdq_max / np.sqrt(np.power(r, 2) + np.power(we * l, 2))
        sys_params_dict["i_max"] = idq_max_norm(
            sys_params_dict["vdc"]/2, 
            sys_params_dict["we_nom"],
            sys_params_dict["r"], 
            sys_params_dict["l"]
        )
    elif env_name in ["PMSM", "PMSMTC", "PMSMTCABC"]:
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
    elif env_name == "PMSMDataBased":
        # Load PMSM data - this would need to be adapted to how your data is stored
        import scipy.io as scp
        pmsm_data = scp.io.loadmat("look_up_table_based_pmsm_prius_motor_data.mat", spmatrix=False)
        
        sys_params_dict = {
            "dt": 1 / 10e3,      # Sampling time [s]
            "r": 0.015,          # Resistance [Ohm]
            "id": pmsm_data['imd'].flatten(),       # Current vector d-frame [A]
            "iq": pmsm_data['imq'].flatten(),       # Current vector d-frame [A]
            "ldd": pmsm_data['Lmidd'],              # Self-inductance matrix d-frame [H]
            "ldq": pmsm_data['Lmidq'],              # Cross-coupling inductance matrix dq-frame [H]
            "lqq": pmsm_data['Lmiqq'],              # Self-inductance matrix q-frame [H]
            "lss": 0.0001,                          # Leakage inductance [H]
            "psid": pmsm_data['Psid'],              # Flux-linkage matrix d-frame [Wb]
            "psiq": pmsm_data['Psiq'],              # Flux-linkage matrix q-frame [Wb]
            "vdc": 1200,                            # DC bus voltage [V]
            "we_nom": 200*2*np.pi,                  # Nominal speed [rad/s]
            "i_max": 150,                           # Maximum current [A]
        }
    else:
        raise NotImplementedError(f"Environment {env_name} not implemented")
    
    # Add reward function to system parameters
    sys_params_dict["reward"] = reward_function
    
    return sys_params_dict

def main():
    """Main function to coordinate the RL training and testing."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure system parameters
    sys_params_dict = configure_system_parameters(args.env_name, args.reward_function)
    
    # Define environment configuration
    env_config = {
        "LoadRL": {
            "name": f"Single Phase RL system / Delta Vdq penalty / Reward {args.reward_function}",
            "max_episode_steps": 200,
            "max_episodes": 200,
            "reward": args.reward_function,
            "model_name": f"ddpg_EnvLoadRL_{args.reward_function}"
        },
        "Load3RL": {
            "name": f"Three-phase RL system / Delta Vdq penalty / Reward {args.reward_function}",
            "max_episode_steps": 200,
            "max_episodes": 300,
            "reward": args.reward_function,
            "model_name": f"ddpg_EnvLoad3RL_{args.reward_function}"
        },
        "PMSM": {
            "name": f"PMSM / Delta Vdq penalty / Reward {args.reward_function}",
            "max_episode_steps": 200,
            "max_episodes": 1_000,
            "reward": args.reward_function,
            "model_name": f"ddpg_EnvPMSM_{args.reward_function}"
        },
        "PMSMTC": {
            "name": f"PMSM torque control / Delta Vdq penalty / Reward {args.reward_function}",
            "max_episode_steps": 200,
            "max_episodes": 1_000,
            "reward": args.reward_function,
            "model_name": f"ddpg_EnvPMSMTC_{args.reward_function}"
        },
        "PMSMDataBased": {
            "name": f"PMSM data based / Delta Vdq penalty / Reward {args.reward_function}",
            "max_episode_steps": 200,
            "max_episodes": 500,
            "reward": args.reward_function,
            "model_name": f"ddpg_EnvPMSMDataBased_{args.reward_function}"
        },
        "PMSMTCABC": {
            "name": f"PMSM torque control with ABC / Delta Vdq penalty / Reward {args.reward_function}",
            "max_episode_steps": 500,
            "max_episodes": 3_000,
            "reward": args.reward_function,
            "model_name": f"ddpg_EnvPMSMTCABC_{args.reward_function}"
        }
    }
    
    # Get the environment configuration
    env_sel = env_config[args.env_name]
    
    # Setup the environment and model
    env, model, vec_env = setup_environment(args.env_name, env_sel, sys_params_dict)
    
    # Run train or tests based on arguments
    if args.train:
        train_model(
            env_sel, 
            model, 
            args.env_name, 
            args.reward_function, 
            args.job_id, 
            args.wandb
        )
    
    if args.test_1:
        run_test_1(
            env_sel, 
            model, 
            vec_env, 
            sys_params_dict, 
            args.env_name, 
            args.plot
        )
    
    if args.test_2:
        run_test_2(
            env_sel, 
            model, 
            vec_env, 
            sys_params_dict, 
            args.env_name, 
            args.reward_function, 
            args.job_id, 
            args.error_type, 
            args.wandb, 
            args.plot
        )
    
    if args.test_3:
        run_test_3(
            env_sel, 
            model, 
            vec_env, 
            sys_params_dict, 
            args.env_name, 
            args.reward_function, 
            args.job_id, 
            args.wandb, 
            args.plot
        )

if __name__ == "__main__":
    main()
