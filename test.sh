declare -a reward_functions=(	"absolute" 
			    	"quadratic" 
				"quadratic_2"
				"square_root"
				"square_root_2"
				"quartic_root"
				"quartic_root_2")
source ~/.venv/RL/bin/activate
for reward_function in "${reward_functions[@]}"
do
    python rl_state_space_control.py --env_name LoadRL --reward_function $reward_function --test
    python rl_state_space_control.py --env_name Load3RL --reward_function $reward_function --test
    python rl_state_space_control.py --env_name PMSM --reward_function $reward_function --test
done
