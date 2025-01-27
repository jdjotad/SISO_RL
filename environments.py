import numpy as np
import scipy.signal as signal
import gymnasium as gym
from gymnasium import spaces

from typing import Any, SupportsFloat

from utils import *

# Resistive Inductive Load Environment with constant reference
class EnvLoadRLConstRef(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.vdc    = sys_params["vdc"]     # DC bus voltage [V]
        self.i_max  = self.vdc / self.r     # Maximum current [A]

        self.reference  = 0

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

        self.low_state = np.array(
            [self.min_current], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_current], dtype=np.float32
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
        next_current = np.clip(self.ad * self.current + self.bd * action_input, -self.i_max, self.i_max)

        # Normalize observation space
        next_current_norm = next_current / self.i_max
        reference_norm = self.reference / self.i_max
        # Observation: [current, reference]
        obs = np.array([next_current_norm], dtype=np.float32)

        terminated = False

        # Reward function
        current_norm = self.current / self.i_max
        tracking_error = np.power(current_norm - reference_norm, 2)
        additional_reward = 0
        if tracking_error <= 1e-6:
            additional_reward = 1
        reward = -tracking_error + additional_reward

        # Update state
        self.current = next_current

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-0.9, 0.9]
        current_norm    = np.round(self.np_random.uniform(low=low, high=high),6)
        self.current    = self.i_max * current_norm

        return np.array(current_norm, dtype=np.float32), {}

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
        additional_reward = 0
        if tracking_error <= 1e-6:
            additional_reward = 1
        reward = -tracking_error + additional_reward

        # Update state value
        self.current = next_current

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = [-0.9, 0.9]
        current_norm    = np.round(self.np_random.uniform(low=low, high=high),5)
        reference_norm  = np.round(self.np_random.uniform(low=low, high=high),5)
        self.current    = self.i_max * current_norm
        self.reference  = self.i_max * reference_norm

        obs = np.array([current_norm, reference_norm], dtype=np.float32)
        return obs, {}

    def _get_obs(self):
        current_norm, reference_norm = np.array([self.current, self.reference]) / self.i_max
        return np.array([current_norm, reference_norm], dtype=np.float32)

# 3-Phase Resistive Inductive Load Environment with constant reference current and constant speed
class EnvLoad3RLConstRefConstSpeed(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.we     = sys_params["we_norm_const"] * sys_params["we_nom"]  # Electrical speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Maximum voltage [V]
        self.vdq_max = np.sqrt(2) * vdc / 4

        # Maximum current [A]
        self.i_max = (vdc / 2) / np.sqrt(np.power(self.r, 2) + np.power(self.we * self.l, 2))

        #  ----- Maximum current analysis -----
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l ,       self.we   ],
                      [-self.we         , -self.r / self.l]])
        b = np.array([[1 / self.l   ,      0    ],
                      [0            , 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss1 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss2 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss3 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss4 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        #  ----- End of maximum current analysis -----

        # Band tolerance for additional reward
        self.tol = sys_params["tolerance"]

        # Store idq_ref
        self.id_ref = self.i_max * sys_params["id_ref_norm_const"]          # Constant Id reference value [A]
        self.iq_ref = self.i_max * sys_params["iq_ref_norm_const"]          # Constant Iq reference value [A]

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

        self.low_observations = np.array(
            [self.min_id, self.min_iq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(2,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        norm_idq_next = np.sqrt(np.power(id_next,2) + np.power(iq_next,2))
        factor_idq = self.i_max / norm_idq_next
        if factor_idq < 1:
            id_next = factor_idq * id_next
            iq_next = factor_idq * iq_next

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        # Observation: [id, iq]
        obs = np.array([id_next_norm, iq_next_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        additional_reward = 0
        if tracking_error_id <= self.tol and tracking_error_iq <= self.tol:
            additional_reward = 1
        reward = -(tracking_error_id + tracking_error_iq) + additional_reward

        # Update states
        self.id = id_next
        self.iq = iq_next

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2) - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)

        # Store idq
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm

        obs = np.array([id_norm, iq_norm], dtype=np.float32)
        return obs, {}

# 3-Phase Resistive Inductive Load Environment with constant reference
class EnvLoad3RLConstRef(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Maximum voltage [V]
        self.vdq_max = np.sqrt(2) * vdc / 4

        # Maximum current [A]
        self.i_max = self.vdq_max / self.r

        # Maximum initial and reference current (speed dependant)
        self.idq_max_norm = lambda we: self.vdq_max / np.sqrt(
            np.power(self.r, 2) + np.power(we * self.l, 2)) / self.i_max


        #  ----- Maximum current analysis -----
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l, self.we_nom],
                      [-self.we_nom, -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss1 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss2 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss3 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss4 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        #  ----- End of maximum current analysis -----

        # Band tolerance for additional reward
        self.tol = sys_params["tolerance"]

        # Store idq_ref
        self.id_ref = self.i_max * sys_params["id_ref_norm_const"]          # Constant Id reference value [A]
        self.iq_ref = self.i_max * sys_params["iq_ref_norm_const"]          # Constant Iq reference value [A]

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
        self.min_we,     self.max_we     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id, self.min_ref_iq, self.min_we], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq, self.max_we], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(5,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        if factor_idq < 1:
            id_next = factor_idq * id_next
            iq_next = factor_idq * iq_next

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        we_norm      = self.we / self.we_nom
        # Observation: [id, iq, id_ref, iq_ref, we]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm, we_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        additional_reward = 0
        if tracking_error_id <= self.tol and tracking_error_iq <= self.tol:
            additional_reward = 1
        reward = -(tracking_error_id + tracking_error_iq) + additional_reward

        # Update states
        self.id = id_next
        self.iq = iq_next

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [we]
        we_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        # Redefine maximum and minimum current values depending on speed
        we = we_norm * self.we_nom
        current_limit = self.idq_max_norm(we)
        low, high = 0.9 * np.array([-current_limit, current_limit])
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2) - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        id_ref_norm = self.id_ref / self.i_max
        iq_ref_norm = self.iq_ref / self.i_max

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l ,         we      ],
                      [-we              , -self.r / self.l]])
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

        # Store idq, and we
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.we     = self.we_nom * we_norm
        # Observation: [id, iq, id_ref, iq_ref, we]
        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm, we_norm], dtype=np.float32)
        return obs, {}

class EnvLoad3RLConstRefDeltaVdq(gym.Env):
    """
    ## Description

    This three-phase resistive inductive load environment with constant reference and delta Vdq in reward function
    considers a problem represented in state space as:

        [did/dt] = [-r/l       we][id]  +  [1/l      0 ][vd]
        [diq/dt]   [-we      -r/l][iq]     [ 0      1/l][vq]

    - `r`: phase stator resistance [Ohm]
    - `l`: phase inductance [H]
    - `id`: current in d-axis [A]
    - `iq`: current in q-axis [A]
    - `we`: electrical speed [rad/s]
    - `vd`: voltage in d-axis [V]
    - `vq`: voltage in q-axis [V]

    ## Action Space

    The action is a `ndarray` with shape `(2,)` representing the voltage applied in dq frame.

    | Num | Action         | Min  | Max |
    |-----|----------------|------|-----|
    | 0   | D-axis voltage | -1.0 | 1.0 |
    | 1   | Q-axis voltage | -1.0 | 1.0 |

    ## Observation Space

    The observation is a `ndarray` with shape `(7,)` representing the electrical current in dq, reference current in dq,
    electrical speed, and previous voltage values in dq

    | Num | Observation                     | Min  | Max |
    |-----|---------------------------------|------|-----|
    | 0   | Current in d axis               | -1.0 | 1.0 |
    | 1   | Current in q axis               | -1.0 | 1.0 |
    | 2   | D-axis reference current        | -1.0 | 1.0 |
    | 3   | Q-axis reference current        | -1.0 | 1.0 |
    | 4   | Electrical speed                | -1.0 | 1.0 |
    | 5   | Previous d-axis voltage applied | -1.0 | 1.0 |
    | 6   | Previous q-axis voltage applied | -1.0 | 1.0 |

    ## Rewards

    The reward function is defined as:

    *r = -((id_norm-id_ref_norm)<sup>2</sup> + (id_norm-id_ref_norm)<sup>2</sup> +
                                0.1 * ((prev_vd_norm-vd_norm)<sup>2</sup> + (prev_vq_norm-vq_norm)<sup>2</sup>))
                                    + additional_reward*

    where `id_norm` is the system's current in d-axis normalized between *[-1, 1]*, `iq_norm` is the system's current
    in q-axis normalized between *[-1, 1]*, `id_ref_norm` is the system's reference current in d-axis normalized between
    *[-1, 1]*, `iq_ref_norm` is the system's reference current in q-axis normalized between *[-1, 1]*, `prev_vd_norm`
    is the previously applied voltage in d-axis normalized between *[-1, 1]*, prev_vq_norm` is the previously applied
    voltage in q-axis normalized between *[-1, 1]*, `vd_norm` is the voltage applied in d-axis normalized between
    *[-1, 1]*, `vq_norm` is the voltage applied in q-axis normalized between *[-1, 1]*, and `additional_reward` is 1
    only when `idq_norm` and `idq_ref_norm` are sufficiently close (based no predefined tolerance value) or 0 otherwise.

    Based on the above equation, the minimum reward that can be obtained is
    *-(2<sup>2</sup> + 2<sup>2</sup>  + 0.1 * (2<sup>2</sup> + 2<sup>2</sup>)) + 0 = -8.8*,
    while the maximum reward that can be obtained is
    *-(0<sup>2</sup> + 0<sup>2</sup>  + 0.1 * (0<sup>2</sup> + 0<sup>2</sup>)) + 1 = 1*,

    ## Starting State

    The starting state is a random current in *[-1, 1]* considering a limitation based on a circle of radius 1. Also,
    the electrical speed is randomized in *[-1, 1]*.

    ## Episode Truncation

    The episode truncates at 1000 time steps.
    """

    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Maximum voltage [V]
        self.vdq_max = np.sqrt(2) * vdc / 4

        # Maximum current [A]
        self.i_max = self.vdq_max / self.r

        # Maximum initial and reference current (speed dependant)
        self.idq_max_norm = lambda we: self.vdq_max / np.sqrt(
            np.power(self.r, 2) + np.power(we * self.l, 2)) / self.i_max

        #  ----- Maximum current analysis -----
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l, self.we_nom],
                      [-self.we_nom, -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss1 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss2 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss3 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss4 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        #  ----- End of maximum current analysis -----

        # Band tolerance for additional reward
        self.tol = sys_params["tolerance"]

        # Store idq_ref
        self.id_ref = self.i_max * sys_params["id_ref_norm_const"]          # Constant Id reference value [A]
        self.iq_ref = self.i_max * sys_params["iq_ref_norm_const"]          # Constant Iq reference value [A]

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
        self.min_we,     self.max_we     = [-1.0, 1.0]
        self.min_vd,     self.max_vd     = [-1.0, 1.0]
        self.min_vq,     self.max_vq     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id, self.min_ref_iq, self.min_we, self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq, self.max_we, self.max_vd, self.max_vq], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(7,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        if factor_idq < 1:
            id_next = factor_idq * id_next
            iq_next = factor_idq * iq_next

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        we_norm      = self.we / self.we_nom
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm, we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        delta_vd = np.power(action[0] - prev_vd_norm, 2)
        delta_vq = np.power(action[1] - prev_vq_norm, 2)
        additional_reward = 0
        if tracking_error_id <= self.tol and tracking_error_iq <= self.tol:
            additional_reward = 1
        reward = -(tracking_error_id + tracking_error_iq + 0.1*(delta_vd + delta_vq)) + additional_reward

        # Update states
        self.id = id_next
        self.iq = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [we]
        we_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        # Redefine maximum and minimum current values depending on speed
        we = we_norm * self.we_nom
        current_limit = self.idq_max_norm(we)
        low, high = 0.9 * np.array([-current_limit, current_limit])
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2) - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        id_ref_norm = self.id_ref / self.i_max
        iq_ref_norm = self.iq_ref / self.i_max

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l ,           we    ],
                      [-we              , -self.r / self.l]])
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

        # Store idq, and we
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.we     = self.we_nom * we_norm

        # Additional steps to store previous actions
        n = 1
        self.prev_vd = 0
        self.prev_vq = 0
        for _ in range(n):
            self.step(action=self.action_space.sample())

        # Store prev_vd and prev_vq
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm, we_norm, prev_vd_norm, prev_vq_norm],
                       dtype=np.float32)
        return obs, {}

# 3-Phase Resistive Inductive Load Environment with constant speed
class EnvLoad3RLConstSpeed(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Electrical speed [rad/s]
        self.we = sys_params["we_norm_const"] * sys_params["we_nom"]

        # Maximum voltage [V]
        self.vdq_max = np.sqrt(2) * vdc / 4

        # Maximum current [A]
        self.i_max  = self.vdq_max / np.sqrt(np.power(self.r,2) + np.power(self.we*self.l,2))

        #  ----- Maximum current analysis -----
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l, self.we],
                      [-self.we, -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss1 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss2 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss3 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss4 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        #  ----- End of maximum current analysis -----

        # Band tolerance for additional reward
        self.tol = sys_params["tolerance"]

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l,        self.we],
                      [-self.we,        -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        self.ad = ad
        self.bd = bd

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
        self.min_we,     self.max_we     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id, self.min_ref_iq, self.min_we], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq, self.max_we], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(5,), dtype=np.float32
        )

        self.counter = 0

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        if factor_idq < 1:
            id_next = factor_idq * id_next
            iq_next = factor_idq * iq_next

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        we_norm      = self.we /self.we_nom
        # Observation: [id, iq, id_ref, iq_ref, we]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm, we_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        additional_reward = 0
        if tracking_error_id <= self.tol and tracking_error_iq <= self.tol:
            additional_reward = 1
        reward = -(tracking_error_id + tracking_error_iq) + additional_reward

        # Update states
        self.id = id_next
        self.iq = iq_next

        self.counter += 1

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_ref_lim = np.sqrt(np.power(high,2)  - np.power(id_ref_norm, 2))
        iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)
        # [we]
        we_norm = self.we / self.we_nom

        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm

        # Observation: [id, iq, id_ref, iq_ref, we]
        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm, we_norm],
                       dtype=np.float32)
        return obs, {}

# 3-Phase Resistive Inductive Load Environment with constant speed and delta vdq in reward function
class EnvLoad3RLConstSpeedDeltaVdq(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Electrical speed [rad/s]
        self.we = sys_params["we_norm_const"] * sys_params["we_nom"]

        # Maximum voltage [V]
        self.vdq_max = np.sqrt(2) * vdc / 4

        # Maximum current [A]
        self.i_max  = self.vdq_max / np.sqrt(np.power(self.r,2) + np.power(self.we*self.l,2))

        #  ----- Maximum current analysis -----
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l, self.we],
                      [-self.we, -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss1 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss2 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss3 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss4 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        #  ----- End of maximum current analysis -----

        # Band tolerance for additional reward
        self.tol = sys_params["tolerance"]

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l,        self.we],
                      [-self.we,        -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        self.ad = ad
        self.bd = bd

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
        self.min_vd,     self.max_vd     = [-1.0, 1.0]
        self.min_vq,     self.max_vq     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id, self.min_ref_iq, self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq, self.max_vd, self.max_vq], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(6,), dtype=np.float32
        )

        self.counter = 0

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        if factor_idq < 1:
            id_next = factor_idq * id_next
            iq_next = factor_idq * iq_next

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, prev_vd, prev_vq]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        delta_vd = np.power(action[0] - prev_vd_norm, 2)
        delta_vq = np.power(action[1] - prev_vq_norm, 2)
        additional_reward = 0
        if tracking_error_id <= self.tol and tracking_error_iq <= self.tol:
            additional_reward = 1
        reward = -((tracking_error_id + tracking_error_iq) + 0.1*(delta_vd + delta_vq)) + additional_reward

        # Update states
        self.id = id_next
        self.iq = iq_next

        self.counter += 1

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_ref_lim = np.sqrt(np.power(high,2)  - np.power(id_ref_norm, 2))
        iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)

        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm

        # Additional steps to store previous actions
        n = 1
        self.prev_vd = 0
        self.prev_vq = 0
        for _ in range(n):
            self.step(action=self.action_space.sample())

        # Store prev_vd and prev_vq
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, prev_vd, prev_vq]
        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm, prev_vd_norm, prev_vq_norm],
                       dtype=np.float32)
        return obs, {}

class EnvLoad3RLDeltaVdq(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.l      = sys_params["l"]       # Phase Inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Maximum voltage [V]
        self.vdq_max = np.sqrt(2) * vdc / 4

        # Maximum current [A]
        self.i_max  = self.vdq_max / self.r

        # Maximum initial and reference current (speed dependant)
        self.idq_max_norm = lambda we: self.vdq_max / np.sqrt(np.power(self.r,2) + np.power(we*self.l,2))/self.i_max

        #  ----- Maximum current analysis -----
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l, self.we_nom],
                      [-self.we_nom, -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss = lambda vdq: -np.linalg.inv(a) @ b @ vdq

        s_ss1 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss2 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss3 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss4 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])

        a = np.array([[-self.r / self.l, 0],
                      [0, -self.r / self.l]])
        # Steady-state
        # 0 = a * s_ss + b * a_ss
        # s_ss = - a^-1* b * a_ss
        s_ss5 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, self.vdq_max])
        s_ss6 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        s_ss7 = -np.linalg.inv(a) @ b @ np.array([self.vdq_max, -self.vdq_max])
        s_ss8 = -np.linalg.inv(a) @ b @ np.array([-self.vdq_max, self.vdq_max])
        #  ----- End of maximum current analysis -----

        # Band tolerance for additional reward
        self.tol = sys_params["tolerance"]

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
        self.min_we,     self.max_we     = [-1.0, 1.0]
        self.min_vd,     self.max_vd     = [-1.0, 1.0]
        self.min_vq,     self.max_vq     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_id, self.min_iq, self.min_ref_id, self.min_ref_iq, self.min_we, self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_id, self.max_iq, self.max_ref_id, self.max_ref_iq, self.max_we, self.max_vd, self.max_vq], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(7,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_vdq = self.vdq_max * action  # Denormalize action

        s_t = np.array([self.id,
                        self.iq])
        a_t = action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        if factor_idq < 1:
            id_next = factor_idq * id_next
            iq_next = factor_idq * iq_next

        # Normalize observation
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        id_ref_norm  = self.id_ref / self.i_max
        iq_ref_norm  = self.iq_ref / self.i_max
        we_norm      = self.we / self.we_nom
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
        obs = np.array([id_next_norm, iq_next_norm,  id_ref_norm, iq_ref_norm, we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        terminated = False

        # Reward function
        id_norm = self.id / self.i_max
        iq_norm = self.iq / self.i_max
        tracking_error_id = np.power(id_norm - id_ref_norm, 2)
        tracking_error_iq = np.power(iq_norm - iq_ref_norm, 2)
        delta_vd = np.power(action[0] - prev_vd_norm, 2)
        delta_vq = np.power(action[1] - prev_vq_norm, 2)
        additional_reward = 0
        if tracking_error_id <= self.tol and tracking_error_iq <= self.tol:
            additional_reward = 1
        reward = -((tracking_error_id + tracking_error_iq) + 0.5*(delta_vd + delta_vq)) + additional_reward

        # Update states
        self.id = id_next
        self.iq = iq_next

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [we]
        we_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        # Redefine maximum and minimum current values depending on speed
        we = we_norm * self.we_nom
        current_limit = self.idq_max_norm(we)
        low, high = 0.9 * np.array([-current_limit, current_limit])
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_ref_lim = np.sqrt(np.power(high,2)  - np.power(id_ref_norm, 2))
        iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L       we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we      -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l,            we],
                      [     -we,        -self.r / self.l]])
        b = np.array([[1 / self.l,          0],
                      [0,           1 / self.l]])
        c = np.eye(2)
        d = np.zeros(2)

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        self.ad = ad
        self.bd = bd

        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm
        self.we     = self.we_nom * we_norm

        # Additional steps to store previous actions
        n = 1
        self.prev_vd = 0
        self.prev_vq = 0
        for _ in range(n):
            self.step(action=self.action_space.sample())

        # Store prev_vd and prev_vq
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
        obs = np.array([id_norm, iq_norm, id_ref_norm, iq_ref_norm, we_norm, prev_vd_norm, prev_vq_norm],
                       dtype=np.float32)
        return obs, {}

if __name__ == "__main__":
    # Environment
    sys_params_dict = {"dt": 1 / 10e3,  # Sampling time [s]
                       "r": 1,  # Resistance [Ohm]
                       "l": 1e-2,  # Inductance [H]
                       "vdc": 500,  # DC bus voltage [V]
                       "we_nom": 400 * np.pi,  # Nominal speed
                       "we_norm_const": 0,  # Constant speed value [rad/s] ; [-0.9, 0.9]
                       "id_ref_norm_const": 0,  # Constant id value [A] ; [-0.9, 0.9]
                       "iq_ref_norm_const": 0,  # Constant iq value [A] ; [-0.9, 0.9]
                       }
    env_test = EnvLoad3RLConstRefConstSpeed(sys_params=sys_params_dict)
    obs_test, _ = env_test.reset()
    env_test.step(action=env_test.action_space.sample())
