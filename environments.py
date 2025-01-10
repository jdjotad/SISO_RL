import numpy as np
import scipy.signal as signal
import gymnasium as gym
from gymnasium import spaces
    
# Resistive Inductive Load Environment with constant reference
class EnvLoadRLConst(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.vdc    = sys_params["vdc"]     # DC bus voltage
        self.i_max  = self.vdc / self.r     # Maximum current [A]

        a = np.array([[-self.r / self.l]])
        b = np.array([[1 / self.l]])
        c = np.array([[1]])
        d = np.array([[0.]])

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are parameters, s(t) the state, and a(t) the action.
        # s(t) = current
        # a(t) = voltage
        self.ad = ad[0][0]
        self.bd = bd[0][0]

        # Limitations for the system
        self.min_action  = -1.0
        self.max_action  =  1.0
        self.min_state   = -1.0
        self.max_state   =  1.0

        self.low_state = np.array(
            [self.min_state], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_state], dtype=np.float32
        )

        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action: np.ndarray):
        state = self.state

        action_input = self.vdc * action[0]     # Denormalize action
        next_state   = self.ad * state + self.bd * action_input
        next_state_norm = np.clip(next_state/self.i_max, self.min_state, self.max_state)

        terminated = False

        reward = -np.power((state - self.reference)/self.i_max, 2)

        # Update state
        self.state = next_state

        return np.array(next_state_norm), reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-1, 1]
        state_norm      = self.np_random.uniform(low=low, high=high)
        self.state      = self.i_max * state_norm
        self.reference  = 0

        return np.array(state_norm, dtype=np.float32), {}

# Resistive Inductive Load Environment varying reference between episodes
class EnvLoadRL(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.vdc    = sys_params["vdc"]     # DC bus voltage
        self.i_max  = self.vdc / self.r     # Maximum current [A]

        a = np.array([[-self.r / self.l]])
        b = np.array([[1 / self.l]])
        c = np.array([[1]])
        d = np.array([[0.]])

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are parameters, s(t) the state, and a(t) the action.
        # s(t) = current
        # a(t) = voltage
        self.ad = ad[0][0]
        self.bd = bd[0][0]

        # Limitations for the system
        self.min_action  = -1.0
        self.max_action  =  1.0
        self.min_current = -1.0
        self.max_current =  1.0
        self.min_reference = -1.0
        self.max_reference =  1.0

        self.low_state = np.array(
            [self.min_current, self.min_reference ], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_current, self.max_reference], dtype=np.float32
        )

        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_input = self.vdc * action[0]  # Denormalize action

        current             = self.current
        next_current        = self.ad * current + self.bd * action_input
        next_current_norm   = np.clip(next_current/self.i_max, self.min_current, self.max_current)

        reference_norm = self.reference / self.i_max

        terminated = False

        reward = -np.power(current - self.reference, 2)

        # Observation: [current, reference]
        obs = np.array([next_current_norm, reference_norm], dtype=np.float32)

        # Update state
        self.current = next_current

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-1, 1]
        current_norm    = self.np_random.uniform(low=low, high=high)
        low, high = [-0.1, 0.1]
        reference_norm  = self.np_random.uniform(low=low, high=high)
        self.current    = self.i_max * current_norm
        self.reference  = self.i_max * reference_norm

        obs = np.array([current_norm, reference_norm], dtype=np.float32)
        return obs, {}

if __name__ == "__main__":
    # Environment
    sys_params_dict = {"dt":1/10e3, "r": 1, "l": 1e-2, "vdc": 500}
    env_test = EnvLoadRL(sys_params=sys_params_dict)
    env_test.reset()
    env_test.step(action=env_test.action_space.sample())