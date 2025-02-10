# SISO_RL

## Install 

Installation instructions.

```bash
conda create -n siso python=3.12
conda activate siso
pip install -r requirements.txt
```

## Run

How to train an RL agent considering a single-phase resistive inductive load, three-phase resistive inductive load, or a permanent magnet synchronous machine:

```bash
python rl_state_space_control.py --env_name X --reward_function Y --train
```

To test the resulting network:

```bash
python rl_state_space_control.py --env_name X --reward_function Y --test
```

where *X* is the environment name {*LoadRL*, *Load3RL*, *PMSM*}, and Y is the reward function type {*absolute*, *quadratic*, *quadratic_2*, *square_root*, *square_root_2*, *quartic_root*, *quartic_root_2*}
