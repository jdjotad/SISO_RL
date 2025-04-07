declare -a reward_functions=(	"absolute" 
			    	"quadratic" 
				"quadratic_2"
				"square_root"
				"square_root_2"
				"quartic_root"
				"quartic_root_2")
declare environments=("PMSM")
source ~/.venv/RL/bin/activate
for reward_function in "${reward_functions[@]}"
do
	for environment in "${environments[@]}"
	do
		python rl_state_space_control.py --env_name $environment --reward_function $reward_function --test --plot
	done
done
