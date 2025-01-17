import numpy as np
import scipy.signal as signal
import gymnasium as gym
from gymnasium import spaces

from utils import *

# Resistive Inductive Load Environment with constant reference
class EnvLoadRLConst(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.vdc    = sys_params["vdc"]     # DC bus voltage [V]
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

        if action[0] > 1 or action[0] < -1:
            breakpoint()
        action_input = self.vdc * action[0]     # Denormalize action
        next_state   = np.clip(self.ad * state + self.bd * action_input, -self.i_max, self.i_max)
        next_state_norm = next_state/self.i_max

        terminated = False

        reward = -np.power((state - self.reference)/self.i_max, 2)

        # Update state
        self.state = next_state

        return np.array(next_state_norm), reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-0.9, 0.9]
        state_norm      = np.round(self.np_random.uniform(low=low, high=high),6)
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
        self.vdc    = sys_params["vdc"]     # DC bus voltage [V]
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

        # Calculate current in next step
        next_current        = np.clip(self.ad * self.current + self.bd * action_input, -self.i_max, self.i_max)

        # Normalize observation space
        next_current_norm   =   next_current / self.i_max
        reference_norm      = self.reference / self.i_max
        # Observation: [current, reference]
        obs = np.array([next_current_norm, reference_norm], dtype=np.float32)

        terminated = False

        # Reward function
        current_norm = self.current / self.i_max
        tracking_error = np.power(current_norm - reference_norm, 2)
        if tracking_error <= 1e-6:
            reward = 1
        else:
            reward = -tracking_error

        # Update state value
        self.current = next_current

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-0.9, 0.9]
        current_norm    = np.round(self.np_random.uniform(low=low, high=high),5)
        reference_norm  = np.round(self.np_random.uniform(low=low, high=high),5)
        # current_norm    = self.np_random.uniform(low=low, high=high)
        # reference_norm  = self.np_random.uniform(low=low, high=high)
        self.current    = self.i_max * current_norm
        self.reference  = self.i_max * reference_norm

        obs = np.array([current_norm, reference_norm], dtype=np.float32)
        return obs, {}

    def _get_obs(self):
        current_norm, reference_norm = np.array([self.current, self.reference]) / self.i_max
        return np.array([current_norm, reference_norm], dtype=np.float32)

# 3-Phase Resistive Inductive Load Environment with constant reference
class EnvLoad3RLConst(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.vdc    = sys_params["vdc"]     # DC bus voltage [V]
        self.we     = sys_params["we"]      # Electrical speed [rad/s]
        self.i_max  = (self.vdc/2)/self.r # Maximum current [A]

        self.abc_dq0 = ClarkePark.abc_to_dq0_d
        self.dq0_abc = ClarkePark.dq0_to_abc_d

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l ,            self.we],
                      [        -self.we ,   -self.r / self.l]])
        b = np.array([[1 / self.l   ,       0     ],
                      [ 0           ,   1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        self.ad = ad
        self.bd = bd

        # Steady-state
        # s_ss = ad * s_ss + bd * a_ss
        # s_ss = (I - ad) ^ -1 * bd * a_ss
        s_ss1 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([self.vdc / 2, self.vdc / 2])
        s_ss2 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([-self.vdc / 2, self.vdc / 2])
        s_ss3 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([self.vdc / 2, -self.vdc / 2])
        s_ss4 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([-self.vdc / 2, -self.vdc / 2])

        # Limitations for the system
        # Actions
        self.min_vd, self.max_vd = [-1.0, 1.0]
        self.min_vq, self.max_vq = [-1.0, 1.0]

        self.low_actions = np.array(
            [self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_actions = np.array(
            [self.max_vd, self.max_vq], dtype=np.float32
        )

        # Observations
        self.min_id,     self.max_id     = [-1.0, 1.0]
        self.min_iq,     self.max_iq     = [-1.0, 1.0]
        self.min_ref_id, self.max_ref_id = [-1.0, 1.0]
        self.min_ref_iq, self.max_ref_iq = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id,  self.min_ref_iq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq], dtype=np.float32
        )

        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(4,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = (self.vdc/2) * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        id_next = np.clip(id_next, -self.i_max, self.i_max)
        iq_next = np.clip(iq_next, -self.i_max, self.i_max)

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        # Observation: [id, iq, id_ref, iq_ref]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        if tracking_error_id <= 1e-5 and tracking_error_iq <= 1e-5:
            reward = 1
        else:
            reward = -(tracking_error_id + tracking_error_iq)

        # Update states
        self.id = id_next
        self.iq = iq_next

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-0.9, 0.9]
        # Initialization
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(1 - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        # id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        # iq_ref_lim  = np.sqrt(1 - np.power(id_ref_norm, 2))
        # iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)
        id_ref_norm = 0
        iq_ref_norm = 0

        # Store idq and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm

        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm], dtype=np.float32)
        return obs, {}

# 3-Phase Resistive Inductive Load Environment with variable reference between episodes
class EnvLoad3RL(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.vdc    = sys_params["vdc"]     # DC bus voltage [V]
        self.we     = sys_params["we"]      # Electrical speed [rad/s]
        self.i_max  = (self.vdc/2)/self.r # Maximum current [A]

        self.abc_dq0 = ClarkePark.abc_to_dq0_d
        self.dq0_abc = ClarkePark.dq0_to_abc_d

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l ,            self.we],
                      [        -self.we ,   -self.r / self.l]])
        b = np.array([[1 / self.l   ,       0     ],
                      [ 0           ,   1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        self.ad = ad
        self.bd = bd

        # Steady-state
        # s_ss = ad * s_ss + bd * a_ss
        # s_ss = (I - ad) ^ -1 * bd * a_ss
        s_ss1 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([self.vdc/2, self.vdc/2])
        s_ss2 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([-self.vdc/2, self.vdc/2])
        s_ss3 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([self.vdc/2, -self.vdc/2])
        s_ss4 = np.linalg.inv(np.eye(2) - self.ad) @ self.bd @ np.array([-self.vdc/2, -self.vdc/2])

        # Limitations for the system
        # Actions
        self.min_vd, self.max_vd = [-1.0, 1.0]
        self.min_vq, self.max_vq = [-1.0, 1.0]

        self.low_actions = np.array(
            [self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_actions = np.array(
            [self.max_vd, self.max_vq], dtype=np.float32
        )

        # Observations
        self.min_id,     self.max_id     = [-1.0, 1.0]
        self.min_iq,     self.max_iq     = [-1.0, 1.0]
        self.min_ref_id, self.max_ref_id = [-1.0, 1.0]
        self.min_ref_iq, self.max_ref_iq = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id,  self.min_ref_iq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq], dtype=np.float32
        )

        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(4,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = (self.vdc/2) * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        id_next = np.clip(id_next, -self.i_max, self.i_max)
        iq_next = np.clip(iq_next, -self.i_max, self.i_max)

        # Steady-state
        # s_ss = ad * s_ss + bd * a_ss
        # a_ss = bd^-1 * s_ss * (I - ad)
        a_ss = np.linalg.inv(self.bd) @ np.array([self.id_ref,self.iq_ref]) @ (np.eye(2) - self.ad)

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        # Observation: [id, iq, id_ref, iq_ref]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        if tracking_error_id <= 1e-4 and tracking_error_iq <= 1e-4:
            reward = 1
        else:
            reward = -(tracking_error_id + tracking_error_iq)

        # Update states
        self.id = id_next
        self.iq = iq_next

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-0.9, 0.9]
        # Initialization
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_lim  = np.sqrt(1 - np.power(id_norm, 2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim), 5)
        # [id_ref, iq_ref]
        id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_ref_lim  = np.sqrt(1 - np.power(id_ref_norm, 2))
        iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)

        # Store idq and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm

        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm], dtype=np.float32)
        return obs, {}

if __name__ == "__main__":
    # Environment
    sys_params_dict = {"dt":1/10e3, "r": 1, "l": 1e-2, "vdc": 500, "we": 100}
    env_test = EnvLoad3RLConst(sys_params=sys_params_dict)
    obs, _ = env_test.reset()
    env_test.step(action=env_test.action_space.sample())
