import numpy as np
import scipy.signal as signal
import gymnasium as gym
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
        self.l      = sys_params["l"]      # Inductance [H]
        vdc         = sys_params["vdc"]     # DC bus voltage [V]

        # Reward function type
        self.reward_function = sys_params["reward"]

        # Maximum voltage [V]
        self.vdq_max = vdc/2

        # Maximum current [A]
        self.i_max  = sys_params["i_max"]

        # Steady-state analysis functions
        self.ss_analysis = SSAnalysis()

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
        obs = np.array([i_next_norm, i_ref_norm, i_ref_norm], dtype=np.float32)

        terminated = False

        # Reward function
        i_norm = self.i / self.i_max
        e_i = np.power(i_norm - i_ref_norm, 2)
        delta_v = np.power(action_clip - prev_v_norm, 2)

        if self.reward_function == "quadratic":
            reward = -(e_i + 0.1 * delta_v)
        elif self.reward_function == "absolute":
            reward = -(np.power(e_i, 1/2) + 0.1 * np.power(delta_v, 1/2))
        elif self.reward_function == "square_root":
            reward = -(np.power(e_i, 1/4) + 0.1 * np.power(delta_v, 1/4))
        elif self.reward_function == "quartic_root":
            reward = -(np.power(e_i, 1/8) + 0.1 * np.power(delta_v, 1/8))
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
        self.ss_analysis = SSAnalysis()

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
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[0], 2))
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
        prev_vq_norm = self.prev_vd / self.vdq_max
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
        self.ss_analysis = SSAnalysis()

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
        norm_vdq = np.sqrt(np.power(action_vdq[0], 2) + np.power(action_vdq[0], 2))
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
        prev_vq_norm = self.prev_vd / self.vdq_max
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

if __name__ == "__main__":
    # Environment
    idq_max_norm = lambda vdq_max, we, r, l: vdq_max / np.sqrt(np.power(r, 2) + np.power(we * l, 2))
    sys_params_dict = {"dt": 1 / 10e3,  # Sampling time [s]
                       "r": 1,  # Resistance [Ohm]
                       "l": 1e-2,  # Inductance [H]
                       "vdc": 500,  # DC bus voltage [V]
                       "we_nom": 200 * 2 * np.pi,  # Nominal speed [rad/s]
                       }
    # Maximum current [A]
    sys_params_dict["i_max"] = idq_max_norm(sys_params_dict["vdc"] / 2, sys_params_dict["we_nom"],
                                            sys_params_dict["r"], sys_params_dict["l"])
    env_test = EnvLoad3RL(sys_params=sys_params_dict)
    obs_test, _ = env_test.reset()
    env_test.step(action=env_test.action_space.sample())
