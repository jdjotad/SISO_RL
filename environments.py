import numpy as np
import scipy.signal as signal
import gymnasium as gym
from scipy.optimize import minimize

from gymnasium import spaces

from typing import Any, SupportsFloat

from numpy import ndarray
import matplotlib.pyplot as plt
from utils import *

class EnvLoadRL(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Resistance [Ohm]
        self.l      = sys_params["l"]       # Inductance [H]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Reward function type
        self.reward_function = sys_params["reward"]

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Steady-state analysis functions
        self.ss_analysis = SteadyStateAnalysis()

        # State-space system representation
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
        # Action
        self.min_v, self.max_v = [-1.0, 1.0]

        self.low_actions = np.array(
            [self.min_v], dtype=np.float32
        )
        self.high_actions = np.array(
            [self.max_v], dtype=np.float32
        )

        # Observations
        self.min_i,     self.max_i     = [-1.0, 1.0]
        self.min_ref_i, self.max_ref_i = [-1.0, 1.0]
        self.min_v,     self.max_v     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_i, self.min_ref_i, self.min_v], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_i, self.max_ref_i, self.max_v], dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Define action and observation space within a Box property
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, dtype=np.float32
        )

    def step(self, action: np.ndarray):
        action_clip = np.clip(action[0], -1,1)
        input_voltage = self.vdq_max * action_clip  # Denormalize action

        s_t = self.i
        a_t = input_voltage

        # s(t+1) = ad * s(t) + bd * a(t)
        i_next = np.clip(self.ad * s_t + self.bd * a_t, -self.i_max, self.i_max)

        # Normalize observation space
        i_next_norm = i_next / self.i_max
        i_ref_norm  = self.i_ref / self.i_max
        prev_v_norm = self.prev_v / self.vdq_max

        # Observation: [current, reference, prev_v]
        obs = np.array([i_next_norm, i_ref_norm, prev_v_norm], dtype=np.float32)

        terminated = False

        # Reward function
        i_norm = self.i / self.i_max
        e_i = np.abs(i_norm - i_ref_norm)
        delta_v = np.abs(action_clip - prev_v_norm)

        if self.reward_function == "absolute":
            reward = -(e_i + 0.1 * delta_v)
        elif self.reward_function == "quadratic":
            reward = -(np.power(e_i, 2) + 0.1 * np.power(delta_v, 2))
        elif self.reward_function == "square_root":
            reward = -(np.power(e_i, 1/2) + 0.1 * np.power(delta_v, 1/2))
        elif self.reward_function == "quartic_root":
            reward = -(np.power(e_i, 1/4) + 0.1 * np.power(delta_v, 1/4))
        else:
            raise NotImplementedError

        # Update states
        self.i = i_next
        self.prev_v = input_voltage

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        low, high = 0.9 * np.array([-1, 1])
        # Initialization
        # [i]
        i_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        # [i]
        i_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, plot_current=True)     # Continuous
        # self.ss_analysis.discrete(ad, bd, plot_current=True)     # Discrete

        # Store idq, and idq_ref
        self.i     = self.i_max * i_norm
        self.i_ref = self.i_max * i_ref_norm

        # Additional steps to store previous actions
        n = 2
        self.prev_v = 0
        for _ in range(n):
            obs, _, _, _, _ = self.step(action=self.action_space.sample())
        return obs, {}

class EnvLoad3RL(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Resistance [Ohm]
        self.l      = sys_params["l"]       # Inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Reward function type
        self.reward_function = sys_params["reward"]

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Steady-state analysis functions
        self.ss_analysis = SteadyStateAnalysis()

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

        # Calculate if that the module of Vdq is bigger than 1
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[1], 2))
        # factor_vdq = self.vdq_max / norm_vdq
        # factor_vdq = factor_vdq if factor_vdq < 1 else 1
        factor_vdq = 1

        s_t = np.array([self.id,
                        self.iq])
        a_t = factor_vdq * action_vdq

        # s(t+1) = ad * s(t) + bd * a(t)
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

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
        e_id = np.abs(id_norm - id_ref_norm)
        e_iq = np.abs(iq_norm - iq_ref_norm)
        delta_vd = np.abs(action[0] - prev_vd_norm)
        delta_vq = np.abs(action[1] - prev_vq_norm)

        if self.reward_function == "absolute":
            reward = -(e_id + e_iq + 0.1 * (delta_vd + delta_vq))
        elif self.reward_function == "quadratic":
            reward = -((np.power(e_id, 2) + np.power(e_iq, 2)) +
                            0.1 * (np.power(delta_vd, 2) + np.power(delta_vq, 2)))
        elif self.reward_function == "quadratic_2":
            reward = -((np.power(e_id + e_iq, 2)) + 0.1 * (np.power(delta_vd + delta_vq, 2)))
        elif self.reward_function == "square_root":
            reward = -((np.power(e_id, 1/2) + np.power(e_iq, 1/2)) +
                            0.1 * (np.power(delta_vd, 1/2) + np.power(delta_vq, 1/2)))
        elif self.reward_function == "square_root_2":
            reward = -((np.power(e_id + e_iq, 1/2)) + 0.1 * (np.power(delta_vd + delta_vq, 1/2)))
        elif self.reward_function == "quartic_root":
            reward = -((np.power(e_id, 1/4) + np.power(e_iq, 1/4)) +
                            0.1 * (np.power(delta_vd, 1/4) + np.power(delta_vq, 1/4)))
        elif self.reward_function == "quartic_root_2":
            reward = -((np.power(e_id + e_iq, 1/4)) + 0.1 * (np.power(delta_vd + delta_vq, 1/4)))
        else:
            raise NotImplementedError

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
        we_norm = np.round(self.np_random.uniform(low=0, high=high), 5)
        # Define denormalized speed value
        we = we_norm * self.we_nom
        # we_norm = 0.1
        # [id,iq]
        id_norm = np.round(self.np_random.uniform(low=low, high=high),5)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = np.round(self.np_random.uniform(low=-iq_lim, high=iq_lim),5)
        # [id_ref, iq_ref]
        id_ref_norm = np.round(self.np_random.uniform(low=low, high=high), 5)
        iq_ref_lim = np.sqrt(np.power(high,2)  - np.power(id_ref_norm, 2))
        iq_ref_norm = np.round(self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim), 5)

        ## Testing points
        # we = 909.89321869 # [rad/s]
        # we_norm = 909.89321869/self.we_nom
        # id_norm = 0.1
        # iq_norm = -0.6
        # id_ref_norm = -0.6
        # iq_ref_norm = 0.33

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/L      we][Id]  +  [1/L      0 ][Vd]
        # [dIq/dt]   [-we     -R/L][Iq]     [ 0      1/L][Vq]
        a = np.array([[-self.r / self.l,               we     ],
                      [         -we,          -self.r / self.l]])
        b = np.array([[1 / self.l, 0],
                      [0, 1 / self.l]])
        c = np.eye(2)
        d = np.zeros((2,2))

        (ad, bd, _, _, _) = signal.cont2discrete((a, b, c, d), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t)
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        self.ad = ad
        self.bd = bd

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, plot_current=True)     # Continuous
        # self.ss_analysis.discrete(ad, bd, plot_current=True)     # Discrete

        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm
        self.we     = self.we_nom * we_norm

        # Additional steps to store previous actions
        n = 2
        self.prev_vd = 0
        self.prev_vq = 0
        for _ in range(n):
            obs, _, _, _, _ = self.step(action=self.action_space.sample())
        return obs, {}

class EnvPMSM(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.ld     = sys_params["ld"]      # D-axis Inductance [H]
        self.lq     = sys_params["lq"]      # Q-axis Inductance [H]
        self.lambda_PM = sys_params["lambda_PM"]  # Flux-linkage due to permanent magnets [Wb]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Reward function type
        self.reward_function = sys_params["reward"]

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Steady-state analysis functions
        self.ss_analysis = SteadyStateAnalysis()

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

        # Calculate if that the module of Vdq is bigger than 1
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[1], 2))
        # factor_vdq = self.vdq_max / norm_vdq
        # factor_vdq = factor_vdq if factor_vdq < 1 else 1
        factor_vdq = 1

        s_t = np.array([self.id,
                        self.iq])
        a_t = factor_vdq * action_vdq

        # s(t+1) = ad * s(t) + bd * a(t) + w
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t + self.wd
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

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
        e_id = np.abs(id_norm - id_ref_norm)
        e_iq = np.abs(iq_norm - iq_ref_norm)
        delta_vd = np.abs(action[0] - prev_vd_norm)
        delta_vq = np.abs(action[1] - prev_vq_norm)

        if self.reward_function == "absolute":
            reward = -(e_id + e_iq + 0.1 * (delta_vd + delta_vq))
        elif self.reward_function == "quadratic":
            reward = -((np.power(e_id, 2) + np.power(e_iq, 2)) +
                       0.1 * (np.power(delta_vd, 2) + np.power(delta_vq, 2)))
        elif self.reward_function == "quadratic_2":
            reward = -((np.power(e_id + e_iq, 2)) + 0.1 * (np.power(delta_vd + delta_vq, 2)))
        elif self.reward_function == "square_root":
            reward = -((np.power(e_id, 1/2) + np.power(e_iq, 1/2)) +
                       0.1 * (np.power(delta_vd, 1/2) + np.power(delta_vq, 1/2)))
        elif self.reward_function == "square_root_2":
            reward = -((np.power(e_id + e_iq, 1/2)) + 0.1 * (np.power(delta_vd + delta_vq, 1/2)))
        elif self.reward_function == "quartic_root":
            reward = -((np.power(e_id, 1/4) + np.power(e_iq, 1/4)) +
                       0.1 * (np.power(delta_vd, 1/4) + np.power(delta_vq, 1/4)))
        elif self.reward_function == "quartic_root_2":
            reward = -((np.power(e_id + e_iq, 1/4)) + 0.1 * (np.power(delta_vd + delta_vq, 1/4)))

        # Update states
        self.id = id_next
        self.iq = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        # Boundary for initialization values
        low, high = 0.9 * np.array([-1, 1])

        # Initialization speed
        # [we]
        we_norm = self.np_random.uniform(low=0, high=high)

        # Overwrite predefined speed from options
        if options:
            we_norm = np.float32(options.get("we_norm")) if options.get("we_norm") else we_norm

        # Define denormalized speed value
        we = we_norm * self.we_nom

        ## Testing points
        # we = 909.89321869 # [rad/s]
        # we_norm = 909.89321869/self.we_nom
        # id_norm = 0.1
        # iq_norm = -0.6
        # id_ref_norm = -0.6
        # iq_ref_norm = 0.33

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/Ld      we*Lq/Ld][Id]  +  [1/Ld      0 ][Vd] + [      0      ]
        # [dIq/dt]   [-we*Ld/Lq     -R/Lq][Iq]     [ 0      1/Lq][Vq]   [-we*lambda_PM]
        a = np.array([[-self.r / self.ld,           we * self.lq / self.ld],
                      [-we * self.ld / self.lq,     -self.r / self.lq]])
        b = np.array([[1 / self.ld, 0],
                      [0, 1 / self.lq]])
        w = np.array([[0], [-we * self.lambda_PM]])
        c = np.eye(2)
        d = np.zeros((2,2))

        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2,1))))
        (ad, bdw, _, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t) + w
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        # w = disturbance due to flux-linkage from permanent magnets
        self.ad = ad
        self.bd = bdw[:,:b.shape[1]]
        self.wd = bdw[:,b.shape[1]:].squeeze()

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, w.squeeze(), plot_current=True)        # Continuous
        # self.ss_analysis.discrete(ad, self.bd, self.wd, plot_current=True)     # Discrete

        # Initialization currents
        # [id,iq]
        id_norm = self.np_random.uniform(low=low, high=high)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = self.np_random.uniform(low=-iq_lim, high=iq_lim)
        # [id_ref, iq_ref]
        id_ref_norm = self.np_random.uniform(low=low, high=high)
        iq_ref_lim = np.sqrt(np.power(high,2)  - np.power(id_ref_norm, 2))
        iq_ref_norm = self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim)

        # Overwrite predefined current values from options
        if options:
            id_norm      = np.float32(options.get("id_norm"))      if options.get("id_norm")      is not None else id_norm
            iq_norm      = np.float32(options.get("iq_norm"))      if options.get("iq_norm")      is not None else iq_norm
            id_ref_norm  = np.float32(options.get("id_ref_norm"))  if options.get("id_ref_norm")  is not None else id_ref_norm
            iq_ref_norm  = np.float32(options.get("iq_ref_norm"))  if options.get("iq_ref_norm")  is not None else iq_ref_norm
            prev_vd_norm = np.float32(options.get("prev_vd_norm")) if options.get("prev_vd_norm") is not None else None
            prev_vq_norm = np.float32(options.get("prev_vq_norm")) if options.get("prev_vq_norm") is not None else None

            self.prev_vd = prev_vd_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            self.prev_vq = prev_vq_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            
            # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
            obs = np.array([id_norm, iq_norm,  id_ref_norm, iq_ref_norm, we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        else:
            self.prev_vd = None
            self.prev_vq = None

        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm
        self.we     = self.we_nom * we_norm

        # Additional steps to store previous actions
        if self.prev_vd is None or self.prev_vq is None:
            n = 2
            self.prev_vd = 0
            self.prev_vq = 0
            for _ in range(n):
                obs, _, _, _, _ = self.step(action=self.action_space.sample())
        return obs, {}

class EnvPMSMTC(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.ld     = sys_params["ld"]      # D-axis Inductance [H]
        self.lq     = sys_params["lq"]      # Q-axis Inductance [H]
        self.lambda_PM = sys_params["lambda_PM"]  # Flux-linkage due to permanent magnets [Wb]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        self.p      = sys_params["p"]       # Pair of poles 
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Reward function type
        self.reward_function = sys_params["reward"]

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Maximum torque [Nm]
        self.te_max  = sys_params["te_max"]

        # Torque estimation
        self.psi_d = lambda id: self.ld*id + self.lambda_PM
        self.psi_q = lambda iq: self.lq*iq
        self.te_calculation = lambda id, iq: 3/2*sys_params["p"]*(self.psi_d(id)*iq - self.psi_q(iq)*id)

        # Steady-state analysis functions
        self.ss_analysis = SteadyStateAnalysis(self.vdq_max, self.i_max)

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
        self.min_te,     self.max_te     = [-1.0, 1.0]
        self.min_ref_te, self.max_ref_te = [-1.0, 1.0]
        self.min_id,     self.max_id     = [-1.0, 1.0]
        self.min_iq,     self.max_iq     = [-1.0, 1.0]
        self.min_we,     self.max_we     = [-1.0, 1.0]
        self.min_vd,     self.max_vd     = [-1.0, 1.0]
        self.min_vq,     self.max_vq     = [-1.0, 1.0]

        self.low_observations = np.array(
            [self.min_te, self.min_ref_te, self.min_id, self.min_iq, self.min_we, self.min_vd, self.min_vq], dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_te, self.max_ref_te, self.max_id, self.max_iq, self.max_we, self.max_vd, self.max_vq], dtype=np.float32
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

        # Calculate if that the module of Vdq is bigger than 1
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[1], 2))
        # factor_vdq = self.vdq_max / norm_vdq
        # factor_vdq = factor_vdq if factor_vdq < 1 else 1
        factor_vdq = 1

        s_t = np.array([self.id,
                        self.iq])
        a_t = factor_vdq * action_vdq

        # s(t+1) = ad * s(t) + bd * a(t) + w
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t + self.wd
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

        # Estimate new electric torque
        te_next = self.te_calculation(id_next, iq_next)

        # Normalize observation
        te_next_norm = te_next / self.te_max
        te_ref_norm  = self.te_ref / self.te_max
        id_next_norm = id_next / self.i_max
        iq_next_norm = iq_next / self.i_max
        we_norm      = self.we / self.we_nom
        prev_vd_norm = self.prev_vd / self.vdq_max
        prev_vq_norm = self.prev_vq / self.vdq_max
        # Observation: [te, te_ref, id, iq, we, prev_vd, prev_vq]
        obs = np.array([te_next_norm, te_ref_norm, id_next_norm, iq_next_norm,  we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        terminated = False

        # Reward function
        te_norm = self.te_calculation(self.id, self.iq) / self.te_max
        i_dq_mag = np.sqrt(np.power(self.id, 2) + np.power(self.iq, 2)) / self.i_max
        e_te = np.abs(te_norm - te_ref_norm)
        delta_vd = np.abs(action[0] - prev_vd_norm)
        delta_vq = np.abs(action[1] - prev_vq_norm)

        # Scaling factor for objectives
        w_idq = 0.1
        w_vdq = 0.1
        if self.reward_function == "absolute":
            reward = -(e_te + w_idq * i_dq_mag + w_vdq * (delta_vd + delta_vq))
        elif self.reward_function == "quadratic":
            reward = -(np.power(e_te, 2) + w_idq * np.power(i_dq_mag, 2) + w_vdq * (np.power(delta_vd, 2) + np.power(delta_vq, 2)))
        elif self.reward_function == "quadratic_2":
            reward = -(np.power(e_te, 2) + w_idq * np.power(i_dq_mag, 2) + w_vdq * (np.power(delta_vd + delta_vq, 2)))
        elif self.reward_function == "square_root":
            reward = -(np.power(e_te, 1/2) + w_idq * np.power(i_dq_mag, 1/2) + w_vdq * (np.power(delta_vd, 1/2) + np.power(delta_vq, 1/2)))
        elif self.reward_function == "square_root_2":
            reward = -(np.power(e_te, 1/2) + w_idq * np.power(i_dq_mag, 1/2) + w_vdq * (np.power(delta_vd + delta_vq, 1/2)))
        elif self.reward_function == "quartic_root":
            reward = -(np.power(e_te, 1/4) + w_idq * np.power(i_dq_mag, 1/4) + w_vdq * (np.power(delta_vd, 1/4) + np.power(delta_vq, 1/4)))
        elif self.reward_function == "quartic_root_2":
            reward = -(np.power(e_te, 1/4) + w_idq * np.power(i_dq_mag, 1/4) + w_vdq * (np.power(delta_vd + delta_vq, 1/4)))

        # Update states
        self.id = id_next
        self.iq = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        # Boundary for initialization values
        low, high = 0.9 * np.array([-1, 1])

        # Initialization speed
        # [we]
        we_norm = self.np_random.uniform(low=0, high=high)

        # Overwrite predefined speed from options
        if options:
            we_norm = np.float32(options.get("we_norm")) if options.get("we_norm") else we_norm

        # Define denormalized speed value
        we = we_norm * self.we_nom

        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/Ld      we*Lq/Ld][Id]  +  [1/Ld      0 ][Vd] + [      0      ]
        # [dIq/dt]   [-we*Ld/Lq     -R/Lq][Iq]     [ 0      1/Lq][Vq]   [-we*lambda_PM]
        a = np.array([[-self.r / self.ld,           we * self.lq / self.ld],
                      [-we * self.ld / self.lq,     -self.r / self.lq]])
        b = np.array([[1 / self.ld, 0],
                      [0, 1 / self.lq]])
        w = np.array([[0], [-we * self.lambda_PM]])
        c = np.eye(2)
        d = np.zeros((2,2))

        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2,1))))
        (ad, bdw, _, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method='zoh')

        # s_(t+1) = ad * s(t) + bd * a(t) + w
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        # w = disturbance due to flux-linkage from permanent magnets
        self.ad = ad
        self.bd = bdw[:,:b.shape[1]]
        self.wd = bdw[:,b.shape[1]:].squeeze()

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, w.squeeze(), plot_current=True)        # Continuous
        # self.ss_analysis.discrete(ad, self.bd, self.wd, plot_current=True)     # Discrete

        # Initialization currents
        # [id,iq]
        id_norm = self.np_random.uniform(low=low, high=high)
        iq_lim  = np.sqrt(np.power(high, 2) - np.power(id_norm, 2))
        iq_norm = self.np_random.uniform(low=-iq_lim, high=iq_lim)
        # [te_ref]
        te_ref_norm = self.np_random.uniform(low=low, high=high)
        
        # Overwrite values if provided in options
        if options:
            if "id_norm" in options:
                id_norm = np.float32(options.get("id_norm"))
            if "iq_norm" in options:
                iq_norm = np.float32(options.get("iq_norm"))
            if "te_ref_norm" in options:
                te_ref_norm = np.float32(options.get("te_ref_norm"))
            
            # Initialize previous voltages if provided
            if "prev_vd_norm" in options and "prev_vq_norm" in options:
                prev_vd_norm = np.float32(options.get("prev_vd_norm"))
                prev_vq_norm = np.float32(options.get("prev_vq_norm"))
                
                self.prev_vd = prev_vd_norm * self.vdq_max
                self.prev_vq = prev_vq_norm * self.vdq_max
            else:
                self.prev_vd = self.prev_vq = None
        else:
            self.prev_vd = self.prev_vq = None
        
        # Store dq currents and torque reference
        self.id = self.i_max * id_norm
        self.iq = self.i_max * iq_norm
        self.te_ref = self.te_max * te_ref_norm
        self.we = we

        # Calculate torque
        te_norm = self.te_calculation(self.id, self.iq) / self.te_max
        
        # Initialize previous voltages if not set
        if self.prev_vd is None or self.prev_vq is None:
            n_init_steps = 2
            self.prev_vd = self.prev_vq = 0
            for _ in range(n_init_steps):
                obs, _, _, _, _ = self.step(action=self.action_space.sample())
            return obs, {}
        else:
            # Observation: [te, te_ref, id, iq, we, prev_vd, prev_vq]
            obs = np.array([te_norm, te_ref_norm, id_norm, iq_norm,  we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)
        
        info = {}
        info['mtpa_id'], info['mtpa_iq'] = self.mtpa()
        return obs, info

    def mtpa(self):
        current_constraint = lambda idq: self.i_max**2 - (idq[0]**2 + idq[1]**2)
        cost_function = lambda idq: idq[0]**2 + idq[1]**2

        # Torque calculation
        te_ref = self.te_calculation(self.id, self.iq)

        # Initial guess
        Id0 = 0
        Iq0 = self.te_ref / (3/2*self.p**self.lambda_PM)

        # Define constraint
        current_constraint = {'type': 'ineq', 'fun': lambda idq: self.i_max**2 - (idq[0]**2 + idq[1]**2)}
        torque_constraint = {'type': 'eq', 'fun': lambda idq: self.te_ref - self.te_calculation(idq[0], idq[1])}

        # Solve MTPA optimization
        result = minimize(
            cost_function, 
            [Id0, Iq0], 
            method='SLSQP',
            constraints=[current_constraint, torque_constraint],
            bounds=[(-self.i_max, self.i_max), (-self.i_max, self.i_max)]  # id typically negative, iq positive for MTPA
        )

        optimal_currents = result.x
        # print(f"Optimal currents: id = {optimal_currents[0]:.3f}, iq = {optimal_currents[1]:.3f}")
        mtpa_id_ref = optimal_currents[0]
        mtpa_iq_ref = optimal_currents[1]

        return mtpa_id_ref, mtpa_iq_ref
    
class EnvPMSMDataBased(gym.Env):
    def __init__(self, sys_params, render_mode = None):
        # System parameters
        self.dt     = sys_params["dt"]      # Simulation step time [s]
        self.r      = sys_params["r"]       # Phase Stator Resistance [Ohm]
        self.lss    = sys_params["lss"]     # Leakage inductance [H]
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Current-dependant parameters
        id = sys_params["id"]
        iq = sys_params["iq"]
        self.ldd  = DataBasedParameter(id, iq, sys_params["ldd"])      # Self-inductance matrix d-frame [H]
        self.ldq  = DataBasedParameter(id, iq, sys_params["ldq"])      # Cross-coupling inductance matrix dq-frame [H]
        self.lqq  = DataBasedParameter(id, iq, sys_params["lqq"])      # Self-inductance matrix q-frame [H]
        self.psid = DataBasedParameter(id, iq, sys_params["psid"])   # Flux-linkage matrix d-frame [Wb]
        self.psiq = DataBasedParameter(id, iq, sys_params["psiq"])   # Flux-linkage matrix q-frame [Wb]

        # Reward function type
        self.reward_function = sys_params["reward"]

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Steady-state analysis functions
        self.ss_analysis = SteadyStateAnalysis()

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

        # Calculate if that the module of Vdq is bigger than 1
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[1], 2))
        # factor_vdq = self.vdq_max / norm_vdq
        # factor_vdq = factor_vdq if factor_vdq < 1 else 1
        factor_vdq = 1

        s_t = np.array([self.id,
                        self.iq])
        a_t = factor_vdq * action_vdq

        # s(t+1) = ad * s(t) + bd * a(t) + w
        id_next, iq_next = self.ad @ s_t + self.bd @ a_t + self.wd
        # Rescale the current states to limit it within the boundaries if needed
        norm_idq_next = np.sqrt(np.power(id_next, 2) + np.power(iq_next, 2))
        factor_idq = self.i_max / norm_idq_next
        factor_idq = factor_idq if factor_idq < 1 else 1
        id_next, iq_next = factor_idq * np.array([id_next, iq_next])

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
        e_id = np.abs(id_norm - id_ref_norm)
        e_iq = np.abs(iq_norm - iq_ref_norm)
        delta_vd = np.abs(action[0] - prev_vd_norm)
        delta_vq = np.abs(action[1] - prev_vq_norm)

        if self.reward_function == "absolute":
            reward = -(e_id + e_iq + 0.1 * (delta_vd + delta_vq))
        elif self.reward_function == "quadratic":
            reward = -((np.power(e_id, 2) + np.power(e_iq, 2)) +
                       0.1 * (np.power(delta_vd, 2) + np.power(delta_vq, 2)))
        elif self.reward_function == "quadratic_2":
            reward = -((np.power(e_id + e_iq, 2)) + 0.1 * (np.power(delta_vd + delta_vq, 2)))
        elif self.reward_function == "square_root":
            reward = -((np.power(e_id, 1/2) + np.power(e_iq, 1/2)) +
                       0.1 * (np.power(delta_vd, 1/2) + np.power(delta_vq, 1/2)))
        elif self.reward_function == "square_root_2":
            reward = -((np.power(e_id + e_iq, 1/2)) + 0.1 * (np.power(delta_vd + delta_vq, 1/2)))
        elif self.reward_function == "quartic_root":
            reward = -((np.power(e_id, 1/4) + np.power(e_iq, 1/4)) +
                       0.1 * (np.power(delta_vd, 1/4) + np.power(delta_vq, 1/4)))
        elif self.reward_function == "quartic_root_2":
            reward = -((np.power(e_id + e_iq, 1/4)) + 0.1 * (np.power(delta_vd + delta_vq, 1/4)))

        # Update states
        self.id = id_next
        self.iq = iq_next
        self.prev_vd = action_vdq[0]
        self.prev_vq = action_vdq[1]

        # Update state-space matrices
        ad, bdw = self.state_space_discrete(self.id, self.iq)
        self.ad = ad
        self.bd = bdw[:,:2]
        self.wd = bdw[:,2:].squeeze()

        return obs, reward, terminated, False, {}

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)

        # Boundary for initialization values
        low, high = 0.9 * np.array([-1, 1])

        # Initialization speed
        # [we]
        we_norm = self.np_random.uniform(low=0, high=high)

        # Overwrite predefined speed from options
        if options:
            we_norm = np.float32(options.get("we_norm")) if options.get("we_norm") is not None else we_norm

        # Define denormalized speed value
        self.we = self.we_nom * we_norm        

        # Steady-state analysis
        # self.ss_analysis.continuous(a, b, w.squeeze(), plot_current=True)        # Continuous
        # self.ss_analysis.discrete(ad, self.bd, self.wd, plot_current=True)     # Discrete

        # Initialization currents
        # [id,iq]
        id_norm = self.np_random.uniform(low=low, high=high)
        iq_lim  = np.sqrt(np.power(high,2)  - np.power(id_norm,2))
        iq_norm = self.np_random.uniform(low=-iq_lim, high=iq_lim)
        # [id_ref, iq_ref]
        id_ref_norm = self.np_random.uniform(low=low, high=high)
        iq_ref_lim = np.sqrt(np.power(high,2)  - np.power(id_ref_norm, 2))
        iq_ref_norm = self.np_random.uniform(low=-iq_ref_lim, high=iq_ref_lim)

        # Overwrite predefined current values from options
        if options:
            id_norm      = np.float32(options.get("id_norm"))      if options.get("id_norm")      is not None else id_norm
            iq_norm      = np.float32(options.get("iq_norm"))      if options.get("iq_norm")      is not None else iq_norm
            id_ref_norm  = np.float32(options.get("id_ref_norm"))  if options.get("id_ref_norm")  is not None else id_ref_norm
            iq_ref_norm  = np.float32(options.get("iq_ref_norm"))  if options.get("iq_ref_norm")  is not None else iq_ref_norm
            prev_vd_norm = np.float32(options.get("prev_vd_norm")) if options.get("prev_vd_norm") is not None else None
            prev_vq_norm = np.float32(options.get("prev_vq_norm")) if options.get("prev_vq_norm") is not None else None

            self.prev_vd = prev_vd_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            self.prev_vq = prev_vq_norm * self.vdq_max if prev_vd_norm is not None and prev_vq_norm is not None else None
            
            # Observation: [id, iq, id_ref, iq_ref, we, prev_vd, prev_vq]
            obs = np.array([id_norm, iq_norm,  id_ref_norm, iq_ref_norm, we_norm, prev_vd_norm, prev_vq_norm], dtype=np.float32)

        else:
            self.prev_vd = None
            self.prev_vq = None

        # Store idq, and idq_ref
        self.id     = self.i_max * id_norm
        self.iq     = self.i_max * iq_norm
        self.id_ref = self.i_max * id_ref_norm
        self.iq_ref = self.i_max * iq_ref_norm

        # s_(t+1) = ad * s(t) + bd * a(t) + w
        # where ad and bd are 2x2 matrices, s(t) the state [Id, Iq], and a(t) the actions [Vd, Vq].
        # s(t) = dq currents
        # a(t) = dq voltages
        # w = disturbance due to flux-linkage from permanent magnets
        ad, bdw = self.state_space_discrete(self.id, self.iq)
        self.ad = ad
        self.bd = bdw[:,:2]
        self.wd = bdw[:,2:].squeeze()

        # Additional steps to store previous actions
        if self.prev_vd is None or self.prev_vq is None:
            n = 2
            self.prev_vd = 0
            self.prev_vq = 0
            for _ in range(n):
                obs, _, _, _, _ = self.step(action=self.action_space.sample())

        info = {}
        return obs, info
    
    def state_space_discrete(self, id, iq):
        ldd = self.ldd.interp2d(id,iq)
        ldq = self.ldq.interp2d(id,iq)
        lqq = self.lqq.interp2d(id,iq)
        psid = self.psid.interp2d(id,iq)
        psiq = self.psiq.interp2d(id,iq)

        l_dq = np.array([[ldd + self.lss, ldq],[ldq, lqq + self.lss]])
        l_dq_inv = np.linalg.inv(l_dq)
        # dq-frame continuous state-space
        # dx/dt = a*x + b*u
        # [dId/dt] = [-R/Ld      we*Lq/Ld][Id]  +  [1/Ld      0 ][Vd] + [      0      ]
        # [dIq/dt]   [-we*Ld/Lq     -R/Lq][Iq]     [ 0      1/Lq][Vq]   [-we*lambda_PM]
        a = l_dq_inv @ np.array([[-self.r ,             self.we*self.lss],
                                 [-self.we*self.lss,             -self.r]])
        b = l_dq_inv @ np.array([[1,       0],
                                 [0,       1 ]])
        w = l_dq_inv @ np.array([[self.we * psiq], [-self.we * psid]])
        c = np.eye(2)
        d = np.zeros((2,2))

        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2,1))))
        (ad, bdw, _, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method='zoh')

        return ad, bdw

class EnvPMSMTCABC(gym.Env):
    """
    Reinforcement Learning environment for Permanent Magnet Synchronous Motor (PMSM)
    Torque Control in three-phase (abc) frame.
    
    This environment simulates a PMSM with:
    - Three-phase stator windings (abc-frame)
    - Torque control capabilities
    - Clarke-Park transformations between abc and dq frames
    - Accurate motor dynamics based on physical parameters
    
    The RL agent provides voltage commands and learns to control the motor torque
    while minimizing current usage and voltage variations.
    """
    
    def __init__(self, sys_params, render_mode=None):
        """
        Initialize the abc-frame PMSM Torque Control environment.
        
        Args:
            sys_params (dict): Dictionary containing system parameters
            render_mode (str, optional): Rendering mode
        """
        # Initialize system parameters
        self._init_system_parameters(sys_params)
        
        # Initialize transformation and analysis tools
        self._init_transformation_tools()
        
        # Define action and observation spaces
        self._init_action_space()
        self._init_observation_space()
        
        # Set render mode
        self.render_mode = render_mode

    def _init_system_parameters(self, sys_params):
        """Initialize all system parameters from the provided dictionary."""
        # Motor electrical parameters
        self.dt = sys_params["dt"]  # Simulation step time [s]
        self.r = sys_params["r"]  # Phase Stator Resistance [Ohm]
        self.ld = sys_params["ld"]  # D-axis Inductance [H]
        self.lq = sys_params["lq"]  # Q-axis Inductance [H]
        self.lambda_PM = sys_params["lambda_PM"]  # Flux-linkage due to permanent magnets [Wb]
        self.p = sys_params["p"]  # Pair of poles 
        
        # Operational parameters
        self.we_nom = sys_params["we_nom"]  # Nominal speed [rad/s]
        self.vdc = sys_params["vdc"]  # DC bus voltage [V]
        self.i_max = sys_params["i_max"]  # Maximum current [A]
        self.te_max = sys_params["te_max"]  # Maximum torque [Nm]
        
        # Voltage limits derived from DC bus
        self.vabc_max = self.vdc/2
        self.vdq_max = self.vdc/2
        
        # Reward function configuration
        self.reward_function = sys_params["reward"]
        
        # State variables initialization
        self.theta_e = 0.0  # Electrical angle [rad]
        self.id = self.iq = 0.0  # dq-frame currents
        self.ia = self.ib = self.ic = 0.0  # abc-frame currents
        self.we = 0.0  # Electrical angular velocity [rad/s]
        self.te_ref = 0.0  # Reference torque [Nm]
        self.prev_va = self.prev_vb = self.prev_vc = 0.0  # Previous voltages

    def _init_transformation_tools(self):
        """Initialize transformation tools and motor dynamics functions."""
        # Clarke-Park transformation for abc<->dq conversions
        self.transformer = ClarkeParkTransform()
        
        # Flux and torque calculation functions
        self.psi_d = lambda id: self.ld*id + self.lambda_PM
        self.psi_q = lambda iq: self.lq*iq
        self.te_calculation = lambda id, iq: 3/2*self.p*(self.psi_d(id)*iq - self.psi_q(iq)*id)
        
        # Steady-state analysis tool
        self.ss_analysis = SteadyStateAnalysis(self.vdq_max, self.i_max)

    def _init_action_space(self):
        """Initialize the action space for three-phase voltages."""
        # Define normalized voltage limits for each phase [-1, 1]
        self.min_va, self.max_va = [-1.0, 1.0]
        self.min_vb, self.max_vb = [-1.0, 1.0]
        self.min_vc, self.max_vc = [-1.0, 1.0]
        
        # Define action bounds arrays
        self.low_actions = np.array(
            [self.min_va, self.min_vb, self.min_vc], dtype=np.float32
        )
        self.high_actions = np.array(
            [self.max_va, self.max_vb, self.max_vc], dtype=np.float32
        )
        
        # Create action space
        self.action_space = spaces.Box(
            low=self.low_actions, high=self.high_actions, shape=(3,), dtype=np.float32
        )

    def _init_observation_space(self):
        """Initialize the observation space."""
        # Define normalized observation bounds
        self.min_te, self.max_te = [-1.0, 1.0]
        self.min_ref_te, self.max_ref_te = [-1.0, 1.0]
        self.min_ia, self.max_ia = [-1.0, 1.0]
        self.min_ib, self.max_ib = [-1.0, 1.0]
        self.min_ic, self.max_ic = [-1.0, 1.0]
        self.min_we, self.max_we = [-1.0, 1.0]
        self.min_theta_e, self.max_theta_e = [-1.0, 1.0]
        self.min_va, self.max_va = [-1.0, 1.0]
        self.min_vb, self.max_vb = [-1.0, 1.0]
        self.min_vc, self.max_vc = [-1.0, 1.0]
        
        # Observation bounds arrays: [te, te_ref, ia, ib, ic, we, theta_e, prev_va, prev_vb, prev_vc]
        self.low_observations = np.array(
            [self.min_te, self.min_ref_te, 
             self.min_ia, self.min_ib, self.min_ic, 
             self.min_we, self.min_theta_e,
             self.min_va, self.min_vb, self.min_vc], 
            dtype=np.float32
        )
        self.high_observations = np.array(
            [self.max_te, self.max_ref_te, 
             self.max_ia, self.max_ib, self.max_ic, 
             self.max_we, self.max_theta_e,
             self.max_va, self.max_vb, self.max_vc], 
            dtype=np.float32
        )
        
        # Create observation space
        self.observation_space = spaces.Box(
            low=self.low_observations, high=self.high_observations, shape=(10,), dtype=np.float32
        )

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action (numpy.ndarray): Three-phase voltage commands [va, vb, vc]
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Process action and update system state
        va_t, vb_t, vc_t = self._process_action(action)
        self._update_electrical_angle()
        self._update_motor_state(va_t, vb_t, vc_t)
        
        # Generate observation
        obs = self._get_observation()
        
        # Calculate reward based on current state
        reward = self._calculate_reward(obs[0], obs[1], action)
        
        # Environment doesn't terminate
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _process_action(self, action):
        """
        Process and limit the input action.
        
        Args:
            action (numpy.ndarray): Three-phase voltage commands normalized to [-1, 1]
            
        Returns:
            tuple: (va, vb, vc) - The processed voltage commands
        """
        # Denormalize action (scale from [-1,1] to actual voltage)
        action_vabc = self.vabc_max * action
        
        # Calculate if the magnitude exceeds the limit
        norm_vabc = np.sqrt(np.sum(np.power(action_vabc, 2)))
        factor_vabc = self.vabc_max / norm_vabc if norm_vabc > self.vabc_max else 1
        
        # Apply limiting factor
        va, vb, vc = factor_vabc * action_vabc
        
        # Store these values for next step
        self.prev_va, self.prev_vb, self.prev_vc = va, vb, vc
        
        return va, vb, vc
        
    def _update_electrical_angle(self):
        """Update the electrical angle based on the current speed."""
        self.theta_e += self.we * self.dt
        self.theta_e = self.theta_e % (2 * np.pi)  # Keep angle within [0, 2π]
        
    def _update_motor_state(self, va, vb, vc):
        """
        Update motor state based on applied voltages.
        
        Args:
            va (float): Phase A voltage
            vb (float): Phase B voltage
            vc (float): Phase C voltage
        """
        # Convert abc voltages to dq voltages for simulation
        vd, vq, _ = self.transformer.abc_to_dq0_direct(va, vb, vc, self.theta_e)
        vdq = np.array([vd, vq])
        
        # Get current state in dq frame and apply system dynamics
        idq = np.array([self.id, self.iq])
        idq_next = self._apply_motor_dynamics(idq, vdq)
        
        # Apply current limiting if needed
        idq_next = self._apply_current_limit(idq_next)
        
        # Update state variables
        self.id, self.iq = idq_next
        
        # Convert dq currents to abc currents
        self.ia, self.ib, self.ic = self.transformer.dq0_to_abc_direct(
            self.id, self.iq, 0, self.theta_e
        )
    
    def _apply_motor_dynamics(self, idq, vdq):
        """
        Apply motor dynamics in the dq reference frame.
        
        Args:
            idq (numpy.ndarray): Current state [id, iq]
            vdq (numpy.ndarray): Input voltages [vd, vq]
            
        Returns:
            numpy.ndarray: Next state [id_next, iq_next]
        """
        # Apply discrete state-space model: idq(t+1) = ad*idq(t) + bd*vdq(t) + wd
        idq_next = self.ad @ idq + self.bd @ vdq + self.wd
        return idq_next
    
    def _apply_current_limit(self, idq):
        """
        Apply current limiting to ensure currents stay within bounds.
        
        Args:
            idq (numpy.ndarray): Current state [id, iq]
            
        Returns:
            numpy.ndarray: Limited current state [id_limited, iq_limited]
        """
        # Calculate current magnitude
        id_next, iq_next = idq
        i_mag = np.sqrt(id_next**2 + iq_next**2)
        
        # Apply limiting if needed
        if i_mag > self.i_max:
            scaling_factor = self.i_max / i_mag
            return scaling_factor * idq
        
        return idq
        
    def _get_observation(self):
        """
        Construct the observation vector.
        
        Returns:
            numpy.ndarray: The observation vector for the agent
        """
        # Calculate torque
        te = self.te_calculation(self.id, self.iq)
        
        # Normalize observations
        te_norm = te / self.te_max
        te_ref_norm = self.te_ref / self.te_max
        ia_norm = self.ia / self.i_max
        ib_norm = self.ib / self.i_max
        ic_norm = self.ic / self.i_max
        we_norm = self.we / self.we_nom
        prev_va_norm = self.prev_va / self.vabc_max
        prev_vb_norm = self.prev_vb / self.vabc_max
        prev_vc_norm = self.prev_vc / self.vabc_max
        theta_e_norm = (self.theta_e - np.pi) / np.pi

        # Construct observation vector
        obs = np.array([
            te_norm, te_ref_norm,
            ia_norm, ib_norm, ic_norm,
            we_norm, theta_e_norm,
            prev_va_norm, prev_vb_norm, prev_vc_norm
        ], dtype=np.float32)
        
        return obs
        
    def _calculate_reward(self, te_norm, te_ref_norm, action):
        """
        Calculate the reward based on torque error, current magnitude, and promoting constant dq voltages
        without directly using angles.
        
        Args:
            te_norm (float): Normalized torque
            te_ref_norm (float): Normalized reference torque
            action (numpy.ndarray): Current action [va, vb, vc]
            
        Returns:
            float: Calculated reward
        """
        # Calculate torque tracking error
        e_te = np.abs(te_norm - te_ref_norm)
        
        # Calculate current magnitude using abc-frame variables instead of dq
        # In a balanced three-phase system, the RMS value is sqrt(ia²+ib²+ic²)/sqrt(3)
        # and the magnitude is sqrt(ia²+ib²+ic²)
        i_abc_mag = np.sqrt(np.power(self.ia, 2) + np.power(self.ib, 2) + np.power(self.ic, 2)) / self.i_max
        
        # Calculate voltage changes in abc frame
        prev_va_norm = self.prev_va / self.vabc_max
        prev_vb_norm = self.prev_vb / self.vabc_max
        prev_vc_norm = self.prev_vc / self.vabc_max
        
        # Calculate dq voltage constancy metrics using the abc variables
        # We'll use invariant properties of symmetrical three-phase systems:
        # 1. Voltage magnitude stability: v_a²+v_b²+v_c² should be constant
        # 2. Phase angle relationships: 120° phase shifts should be maintained
        
        # Current and previous voltage magnitudes (3-phase RMS)
        v_rms_curr = np.sqrt((action[0]**2 + action[1]**2 + action[2]**2)/3)
        v_rms_prev = np.sqrt((prev_va_norm**2 + prev_vb_norm**2 + prev_vc_norm**2)/3)
        delta_v_rms = np.abs(v_rms_curr - v_rms_prev)
        
        # Phase voltage relationship metric (should remain constant for balanced system)
        # We use dot products to capture the angular relationships without explicit angles
        va_vb_curr = action[0] * action[1]  # Dot product of va and vb (captures angle)
        vb_vc_curr = action[1] * action[2]  # Dot product of vb and vc
        vc_va_curr = action[2] * action[0]  # Dot product of vc and va
        
        va_vb_prev = prev_va_norm * prev_vb_norm
        vb_vc_prev = prev_vb_norm * prev_vc_norm
        vc_va_prev = prev_vc_norm * prev_va_norm
        
        # Changes in phase relationships
        delta_va_vb = np.abs(va_vb_curr - va_vb_prev)
        delta_vb_vc = np.abs(vb_vc_curr - vb_vc_prev)
        delta_vc_va = np.abs(vc_va_curr - vc_va_prev)
        
        # Symmetry metric (for balanced system, these should be equal)
        phase_balance = np.abs(va_vb_curr - vb_vc_curr) + np.abs(vb_vc_curr - vc_va_curr)
        
        # Combine metrics for dq constancy without using angle
        dq_constancy = delta_v_rms + 0.5 * (delta_va_vb + delta_vb_vc + delta_vc_va) + 0.5 * phase_balance
        
        # Weighting factors
        w_iabc = 0.1  # Weight for current magnitude
        w_dq_const = 0  # Weight for dq constancy
        
        # Calculate reward based on selected reward function type
        if self.reward_function == "absolute":
            return -(e_te + w_iabc * i_abc_mag + w_dq_const * dq_constancy)
            
        elif self.reward_function == "quadratic":
            return -(np.power(e_te, 2) + w_iabc * np.power(i_abc_mag, 2) + 
                    w_dq_const * np.power(dq_constancy, 2))
                    
        elif self.reward_function == "quadratic_2":
            return -(np.power(e_te, 2) + w_iabc * np.power(i_abc_mag, 2) + 
                    w_dq_const * np.power(dq_constancy, 2))
                    
        elif self.reward_function == "square_root":
            return -(np.power(e_te, 1/2) + w_iabc * np.power(i_abc_mag, 1/2) + 
                    w_dq_const * np.power(dq_constancy, 1/2))
                    
        elif self.reward_function == "square_root_2":
            return -(np.power(e_te, 1/2) + w_iabc * np.power(i_abc_mag, 1/2) + 
                    w_dq_const * np.power(dq_constancy, 1/2))
                    
        elif self.reward_function == "quartic_root":
            return -(np.power(e_te, 1/4) + w_iabc * np.power(i_abc_mag, 1/4) + 
                    w_dq_const * np.power(dq_constancy, 1/4))
                    
        elif self.reward_function == "quartic_root_2":
            return -(np.power(e_te, 1/4) + w_iabc * np.power(i_abc_mag, 1/4) + 
                    w_dq_const * np.power(dq_constancy, 1/4))
                    
        elif self.reward_function == "constant_dq":
            # Special reward focused primarily on dq constancy
            return -(e_te + 0.05 * i_abc_mag + 0.5 * dq_constancy)
                    
        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to an initial state.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options for customization
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Initialize motor variables
        self._initialize_motor_variables(options)
        
        # Set up state-space model
        self._setup_state_space_model()
        
        # Convert dq currents to abc currents
        self.ia, self.ib, self.ic = self.transformer.dq0_to_abc_direct(
            self.id, self.iq, 0, self.theta_e
        )
        
        # Initialize previous voltages or do warmup steps
        if self._need_warmup_steps():
            return self._perform_warmup_steps()
        else:
            return self._create_observation_and_info()
    
    def _initialize_motor_variables(self, options):
        """
        Initialize motor variables, potentially using provided options.
        
        Args:
            options (dict, optional): Initialization options
        """
        # Boundary for initialization values
        low, high = 0.9 * np.array([-1, 1])
        
        # Initialize with random values
        we_norm = self.np_random.uniform(low=0, high=high)
        self.theta_e = self.np_random.uniform(low=0, high=2*np.pi)
        id_norm = self.np_random.uniform(low=low, high=high)
        iq_lim = np.sqrt(np.power(high, 2) - np.power(id_norm, 2))
        iq_norm = self.np_random.uniform(low=-iq_lim, high=iq_lim)
        te_ref_norm = self.np_random.uniform(low=low, high=high)
        
        # Overwrite with options if provided
        if options:
            we_norm = np.float32(options.get("we_norm", we_norm))
            self.theta_e = np.float32(options.get("theta_e", self.theta_e))
            id_norm = np.float32(options.get("id_norm", id_norm))
            iq_norm = np.float32(options.get("iq_norm", iq_norm))
            te_ref_norm = np.float32(options.get("te_ref_norm", te_ref_norm))
            
            # Initialize previous voltages if provided
            if all(k in options for k in ["prev_va_norm", "prev_vb_norm", "prev_vc_norm"]):
                prev_va_norm = np.float32(options["prev_va_norm"])
                prev_vb_norm = np.float32(options["prev_vb_norm"])
                prev_vc_norm = np.float32(options["prev_vc_norm"])
                
                self.prev_va = prev_va_norm * self.vabc_max
                self.prev_vb = prev_vb_norm * self.vabc_max
                self.prev_vc = prev_vc_norm * self.vabc_max
            else:
                self.prev_va = self.prev_vb = self.prev_vc = None
        else:
            self.prev_va = self.prev_vb = self.prev_vc = None
        
        # Set denormalized values
        self.we = self.we_nom * we_norm
        self.id = self.i_max * id_norm
        self.iq = self.i_max * iq_norm
        self.te_ref = self.te_max * te_ref_norm
    
    def _setup_state_space_model(self):
        """Set up the discrete state-space model for the motor."""
        # dq-frame continuous state-space model matrices
        a = np.array([
            [-self.r / self.ld, self.we * self.lq / self.ld],
            [-self.we * self.ld / self.lq, -self.r / self.lq]
        ])
        b = np.array([
            [1 / self.ld, 0],
            [0, 1 / self.lq]
        ])
        w = np.array([[0], [-self.we * self.lambda_PM]])
        c = np.eye(2)
        d = np.zeros((2, 2))
        
        # Augment system for discrete conversion
        bw = np.hstack((b, w))
        dw = np.hstack((d, np.zeros((2, 1))))
        
        # Convert to discrete time model
        (ad, bdw, _, _, _) = signal.cont2discrete((a, bw, c, dw), self.dt, method='zoh')
        
        # Store discrete system matrices
        self.ad = ad
        self.bd = bdw[:, :b.shape[1]]
        self.wd = bdw[:, b.shape[1]:].squeeze()
    
    def _need_warmup_steps(self):
        """Check if warmup steps are needed."""
        return self.prev_va is None or self.prev_vb is None or self.prev_vc is None
    
    def _perform_warmup_steps(self):
        """Perform warmup steps to initialize previous actions."""
        n_init_steps = 2
        self.prev_va = self.prev_vb = self.prev_vc = 0
        
        for _ in range(n_init_steps):
            obs, _, _, _, _ = self.step(action=self.action_space.sample())
            
        return obs, {}
    
    def _create_observation_and_info(self):
        """Create observation and info dictionaries without warmup steps."""
        # Calculate torque
        te = self.te_calculation(self.id, self.iq)
        te_norm = te / self.te_max
        
        # Normalize observations
        te_ref_norm = self.te_ref / self.te_max
        ia_norm = self.ia / self.i_max
        ib_norm = self.ib / self.i_max
        ic_norm = self.ic / self.i_max
        we_norm = self.we / self.we_nom
        prev_va_norm = self.prev_va / self.vabc_max
        prev_vb_norm = self.prev_vb / self.vabc_max
        prev_vc_norm = self.prev_vc / self.vabc_max
        theta_e_norm = (self.theta_e - np.pi) / np.pi

        # Observation: [te, te_ref, ia, ib, ic, we, prev_va, prev_vb, prev_vc]
        obs = np.array([
            te_norm, te_ref_norm,
            ia_norm, ib_norm, ic_norm,
            we_norm, theta_e_norm,
            prev_va_norm, prev_vb_norm, prev_vc_norm
        ], dtype=np.float32)
        
        # Create info dictionary with MTPA values
        info = {}
        info['mtpa_id'], info['mtpa_iq'] = self.mtpa()
        
        return obs, info

    def mtpa(self):
        """
        Calculate Maximum Torque Per Ampere (MTPA) current references.
        
        Returns:
            tuple: (id_ref, iq_ref) - Optimal d and q axis currents for MTPA
        """
        return self._solve_mtpa_optimization(self.te_ref)
    
    def _solve_mtpa_optimization(self, torque_reference):
        """
        Solve the MTPA optimization problem.
        
        Args:
            torque_reference (float): Target torque [Nm]
            
        Returns:
            tuple: (id_ref, iq_ref) - Optimal currents
        """
        # Setup cost function to minimize current magnitude
        cost_function = lambda idq: idq[0]**2 + idq[1]**2
        
        # Initial guess (typical for PMSM: id=0, iq calculated for desired torque)
        id0 = 0
        iq0 = torque_reference / (3/2 * self.p * self.lambda_PM)
        
        # Define constraints
        # 1. Current magnitude constraint
        current_constraint = {
            'type': 'ineq', 
            'fun': lambda idq: self.i_max**2 - (idq[0]**2 + idq[1]**2)
        }
        
        # 2. Torque constraint
        torque_constraint = {
            'type': 'eq', 
            'fun': lambda idq: torque_reference - self.te_calculation(idq[0], idq[1])
        }
        
        # Solve optimization problem
        result = minimize(
            cost_function, 
            [id0, iq0], 
            method='SLSQP',
            constraints=[current_constraint, torque_constraint],
            bounds=[(-self.i_max, self.i_max), (-self.i_max, self.i_max)]
        )
        
        # Return optimal currents
        return result.x[0], result.x[1]
    
    
if __name__ == "__main__":
    # Environment
    sys_params_dict = {"dt": 1 / 10e3,      # Sampling time [s]
                       "r": 29.0808e-3,     # Resistance [Ohm]
                       "id": np.array([-300,300]),       # Current in d-frame [H]
                       "iq": np.array([-300,300]),       # Current in q-frame [H]
                       "ld": 1e-1*np.array([[0,1],[2,3]]),       # Inductance d-frame [H]
                       "lq": 1e-1*np.array([[4,5],[6,7]]),       # Inductance q-frame [H]
                       "lambda_PM": 1e-1*np.array([[8,9],[10,11]]), # Flux-linkage due to permanent magnets [Wb]
                       "vdc": 1200,             # DC bus voltage [V]
                       "we_nom": 200*2*np.pi,   # Nominal speed [rad/s]
                       "i_max": 300,            # Maximum current [A]
                       }
    # Reward function
    sys_params_dict["reward"] = "quadratic"
    # Initialize and test environment
    env_test = EnvPMSMDataBased(sys_params=sys_params_dict)
    obs_test, _ = env_test.reset()
    env_test.step(action=env_test.action_space.sample())