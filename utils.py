import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy as scp
from stable_baselines3.common.callbacks import EventCallback

## ------------ Clarke Park Tranformation ------------ ##
class ClarkePark:
    def __init__(self):
        return

    @staticmethod
    def abc_to_alphabeta0(a, b, c):
        alpha = (2 / 3) * (a - b / 2 - c / 2)
        beta = (2 / 3) * (np.sqrt(3) * (b - c) / 2)
        z = (2 / 3) * ((a + b + c) / 2)
        return alpha, beta, z

    @staticmethod
    def alphabeta0_to_abc(alpha, beta, z):
        a = alpha + z
        b = -alpha / 2 + beta * np.sqrt(3) / 2 + z
        c = -alpha / 2 - beta * np.sqrt(3) / 2 + z
        return a, b, c

    @staticmethod
    def abc_to_dq0_d(a, b, c, wt, delta=0):
        d = (2 / 3) * (a * np.cos(wt + delta) + b * np.cos(wt + delta - (2 * np.pi / 3)) + c * np.cos(wt + delta + (2 * np.pi / 3)))
        q = (2 / 3) * (-a * np.sin(wt + delta) - b * np.sin(wt + delta - (2 * np.pi / 3)) - c * np.sin(wt + delta + (2 * np.pi / 3)))
        z = (2 / 3) * (a + b + c) / 2
        return d, q, z

    @staticmethod
    def abc_to_dq0_q(a, b, c, wt, delta=0):
        d = (2 / 3) * (a * np.sin(wt + delta) + b * np.sin(wt + delta - (2 * np.pi / 3)) + c * np.sin(wt + delta + (2 * np.pi / 3)))
        q = (2 / 3) * (a * np.cos(wt + delta) + b * np.cos(wt + delta - (2 * np.pi / 3)) + c * np.cos(wt + delta + (2 * np.pi / 3)))
        z = (2 / 3) * (a + b + c) / 2
        return d, q, z

    @staticmethod
    def dq0_to_abc_d(d, q, z, wt, delta=0):
        a = d * np.cos(wt + delta) - q * np.sin(wt + delta) + z
        b = d * np.cos(wt - (2 * np.pi / 3) + delta) - q * np.sin(wt - (2 * np.pi / 3) + delta) + z
        c = d * np.cos(wt + (2 * np.pi / 3) + delta) - q * np.sin(wt + (2 * np.pi / 3) + delta) + z
        return a, b, c

    @staticmethod
    def dq0_to_abc_q(d, q, z, wt, delta=0):
        a = d * np.sin(wt + delta) + q * np.cos(wt + delta) + z
        b = d * np.sin(wt - (2 * np.pi / 3) + delta) + q * np.cos(wt - (2 * np.pi / 3) + delta) + z
        c = d * np.sin(wt + (2 * np.pi / 3) + delta) + q * np.cos(wt + (2 * np.pi / 3) + delta) + z
        return a, b, c

## ------------ Two level Inverter ------------ ##
class TwoLevelInverter():
    def __init__(self, VDC):
        self.VDC = VDC
    
    def switches_to_voltage(self, Sa, Sb, Sc):
        # Voltage in abc
        if type(Sa) == int and type(Sb) == int or type(Sc) == int:
            Va = -self.VDC/2 if Sa == 0 else self.VDC/2
            Vb = -self.VDC/2 if Sb == 0 else self.VDC/2
            Vc = -self.VDC/2 if Sc == 0 else self.VDC/2
        else:
            Va = [-self.VDC/2 if Sai == 0 else self.VDC/2 for Sai in Sa]
            Vb = [-self.VDC/2 if Sbi == 0 else self.VDC/2 for Sbi in Sb]
            Vc = [-self.VDC/2 if Sci == 0 else self.VDC/2 for Sci in Sc]

        return Va, Vb, Vc

## ------------ Pulse Width Modulation ------------ ##
class PWM():
    def __init__(self, VDC, steps=100, PWM_type="SPWM"):
        self.VDC = VDC       # DC-link voltage [V]
        self.steps = steps   # Steps per period
        
        self.PWM_type = PWM_type
        self.Vmax = VDC/2
        
    def normalize(self, Vm):
        return np.divide(Vm, self.Vmax)/2 + 0.5
    
    def triangular(self):
        carrier = np.empty(self.steps)
        MAX_COUNTER = self.steps/2

        for i in range(self.steps):
            if i < MAX_COUNTER:
                carrier[i] = (i % MAX_COUNTER)/MAX_COUNTER
            else:
                carrier[i] = (MAX_COUNTER - (i % MAX_COUNTER))/MAX_COUNTER

        return carrier

    def modulation(self, Vm):
        carrier = self.triangular()
        if self.PWM_type == "SPWM":
            modulation_signal = self.normalize(Vm)
            Sa = np.array((modulation_signal[0] >= carrier), dtype=int)
            Sb = np.array((modulation_signal[1] >= carrier), dtype=int)
            Sc = np.array((modulation_signal[2] >= carrier), dtype=int)
        elif self.PWM_type == "SVPWM":
            Vm_max = np.max(Vm)
            Vm_min = np.min(Vm)
            modulation_signal = self.normalize(Vm - (Vm_max + Vm_min)/2)
            Sa = np.array((modulation_signal[0] >= carrier), dtype=int)
            Sb = np.array((modulation_signal[1] >= carrier), dtype=int)
            Sc = np.array((modulation_signal[2] >= carrier), dtype=int)
        else:
            raise("NotImplementedError")

        return Sa, Sb, Sc
    
class PEHardware():
    def __init__(self, VDC, ts, we, steps=100, PWM_type="SPWM"):
        # Clarke-Park transformation
        self.abc_dq0 = ClarkePark().abc_to_dq0_d
        self.dq0_abc = ClarkePark().dq0_to_abc_d

        # PWM
        self.pwm = PWM(VDC, steps=steps, PWM_type=PWM_type).modulation

        # Two level Inverter
        self.inverter = TwoLevelInverter(VDC).switches_to_voltage

        # System parameters
        self.delta_t    = ts
        self.we         = we

        # Initialize time
        self.time       = 0

    def action(self, Vmd, Vmq):
        # Perform one time step
        self.time = self.time + self.delta_t

        # dq0 to abc
        Vma, Vmb, Vmc = self.dq0_abc(Vmd, Vmq, 0, self.we*self.time)

        # PWM
        Sa, Sb, Sc = self.pwm([Vma, Vmb, Vmc])
        
        # Switches to voltage
        Va, Vb, Vc = self.inverter(Sa, Sb, Sc)
        # Utilize the average value
        Va, Vb, Vc = [np.mean(Va), np.mean(Vb), np.mean(Vc)]

        # DQ Voltage values
        Vd, Vq, _ = self.abc_dq0(Va, Vb, Vc, self.we*self.time)

        return Vd, Vq

class SSAnalysis:
    def __init__(self):
        return

    def continuous(self,a, b, w=None, plot_current=False):
        if w is not None:
            # dx/dt = a*x + b*u + w
            # Steady-state
            # 0 = a * x_ss + b * u_ss + w
            # x_ss = - a^-1 * (b * u_ss + w)
            x_ss = lambda vdq: -np.linalg.inv(a) @ (b @ vdq + np.kron(np.ones((vdq.shape[1], 1)), w).T) if vdq.shape == (
            2, 1000) else -np.linalg.inv(a) @ (b @ vdq + w)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # 0 = a * x_ss + b * u_ss + w
            # u_ss = - b^-1 * (a * x_ss + w)
            u_ss = lambda idq: -np.linalg.inv(b) @ (a @ idq + w)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Continuous state-space model")
                # plt.show()
        else:
            # dx/dt = a*x + b*u
            # Steady-state
            # 0 = a * x_ss + b * u_ss
            # x_ss = - a^-1 * (b * u_ss )
            x_ss = lambda vdq: -np.linalg.inv(a) @ (b @ vdq)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # 0 = a * x_ss + b * u_ss
            # u_ss = - b^-1 * (a * x_ss)
            u_ss = lambda idq: -np.linalg.inv(b) @ (a @ idq)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Continuous state-space model")
                # plt.show()

    def discrete(self, ad, bd, wd=None, plot_current=False):
        if wd is not None:
            # x_k+1 = ad * x_k + bd * u_k + wd
            # Steady-state
            # x_ss = ad * x_ss + bd * u_ss + wd
            # x_ss = (I - ad)^-1 * (bd * u_ss + wd)
            x_ss = lambda vdq: -np.linalg.inv(np.eye(2) - ad) @ (
                        bd @ vdq + np.kron(np.ones((vdq.shape[1], 1)), wd).T) if vdq.shape == (
                2, 1000) else -np.linalg.inv(ad) @ (bd @ vdq + wd)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # x_ss = ad * x_ss + bd * u_ss + wd
            # u_ss = bd^-1 * ((I - ad) * x_ss - wd)
            u_ss = lambda idq: -np.linalg.inv(bd) @ ((np.eye(2) - ad) @ idq - wd)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Discrete state-space model")
                # plt.show()
        else:
            # x_k+1 = ad * x_k + bd * u_k
            # Steady-state
            # x_ss = ad * x_ss + bd * u_ss
            # x_ss = (I - ad)^-1 * (bd * u_ss)
            x_ss = lambda vdq: -np.linalg.inv(np.eye(2) - ad) @ (bd @ vdq)

            x_ss1 = x_ss(np.array([0, self.vdq_max]))
            x_ss2 = x_ss(np.array([self.vdq_max, 0]))
            x_ss3 = x_ss(np.array([0, -self.vdq_max]))
            x_ss4 = x_ss(np.array([-self.vdq_max, 0]))
            x_ss5 = x_ss(np.array([self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss6 = x_ss(np.array([self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))
            x_ss7 = x_ss(np.array([-self.vdq_max / np.sqrt(2), self.vdq_max / np.sqrt(2)]))
            x_ss8 = x_ss(np.array([-self.vdq_max / np.sqrt(2), -self.vdq_max / np.sqrt(2)]))

            # x_ss = ad * x_ss + bd * u_ss
            # u_ss = bd^-1 * ((I - ad) * x_ss)
            u_ss = lambda idq: -np.linalg.inv(bd) @ ((np.eye(2) - ad) @ idq)

            u_ss1 = u_ss([0, self.i_max])
            u_ss2 = u_ss([self.i_max, 0])
            u_ss3 = u_ss([0, -self.i_max])
            u_ss4 = u_ss([-self.i_max, 0])
            u_ss5 = u_ss([self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss6 = u_ss([self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])
            u_ss7 = u_ss([-self.i_max / np.sqrt(2), self.i_max / np.sqrt(2)])
            u_ss8 = u_ss([-self.i_max / np.sqrt(2), -self.i_max / np.sqrt(2)])

            if plot_current:
                v_d = np.linspace(-1, 1, 1000)
                v_q = np.sqrt(1 - np.power(v_d, 2))
                vdq = self.vdq_max * np.array([v_d, v_q])
                x_ss_data_pos = x_ss(vdq)
                vdq = self.vdq_max * np.array([v_d, -v_q])
                x_ss_data_neg = x_ss(vdq)
                id = np.concatenate((x_ss_data_pos[0], x_ss_data_neg[0]), 0)
                iq = np.concatenate((x_ss_data_pos[1], x_ss_data_neg[1]), 0)
                plt.plot(id, iq, label="Current by voltage limitation")
                id = np.linspace(-1, 1, 1000)
                iq = np.sqrt(1 - np.power(id, 2))
                id_circle = self.i_max * np.concatenate((id, id), 0)
                iq_circle = self.i_max * np.concatenate((iq, -iq), 0)
                plt.plot(id_circle, iq_circle, label="Maximum current circle")
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                           fancybox=True, shadow=True, ncol=2)
                plt.title("Discrete state-space model")
                # plt.show()

class PlotTest():
    def __init__(self):
        return

    def plot_single_phase(self, idx, observations, actions, reward, env_name, model_name, reward_type):
        # plt.clf()
        plt.close()
        plt.suptitle(f"Reward: {reward_type}\n")
        # Plot State
        fig = plt.gcf()
        ax = plt.subplot(131)
        fig.set_figheight(6)
        fig.set_figwidth(10)
        ax.set_title("State vs step")
        ax.plot(observations, label=['I', 'Iref'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(actions, label=['V'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        if not os.path.exists(f"plots/{env_name}/{reward_type}/"):
            os.makedirs(f"plots/{env_name}/{reward_type}/")
        plt.savefig(f"plots/{env_name}/{reward_type}/{model_name}_{idx}.pdf", bbox_inches='tight')
        plt.pause(0.001)  # pause a bit so that plots are updated


    def plot_three_phase(self, idx, observations, actions, reward, env_name, model_name, reward_type, speed=None):
        # plt.clf()
        plt.close()
        if speed is not None:
            plt.suptitle(f"Reward: {reward_type}\nSpeed = {speed} [rad/s]")
        # Plot State
        fig = plt.gcf()
        ax = plt.subplot(131)
        fig.set_figheight(6)
        fig.set_figwidth(10)
        ax.set_title("State vs step")
        ax.plot(observations, label=['Id', 'Iq', 'Idref', 'Iqref'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(actions, label=['Vd', 'Vq'])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        if not os.path.exists(f"plots/{env_name}/{reward_type}/"):
            os.makedirs(f"plots/{env_name}/{reward_type}/")
        plt.savefig(f"plots/{env_name}/{reward_type}/{model_name}_{idx}.pdf", bbox_inches='tight')
        # plt.pause(0.001)  # pause a bit so that plots are updated

        # plt.show()

    def plot_test_2(self, id, iq, speed):
        plt.figure()
        u = np.diff(speed)
        v = np.diff(id)
        w = np.diff(iq)
        pos_x = speed[:-1] + u/2
        pos_y = id[:-1] + v/2
        pos_w = iq[:-1] + w/2
        norm = np.sqrt(u**2 + v**2)
        norm2 = np.sqrt(u**2 + w**2) 

        fig = plt.gcf()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        ax = plt.subplot(121)
        ax.set_title("Id vs speed")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.set_ylabel("Id [A]")
        ax.plot(speed, id, marker="o")
        ax.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")
        ax = plt.subplot(122)
        ax.set_title("Iq vs speed")
        ax.set_ylabel("Iq [A]")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.plot(speed, iq, marker="o")
        ax.quiver(pos_x, pos_w, u/norm2, w/norm2, angles="xy", zorder=5, pivot="mid")

        plt.figure()
        fig = plt.gcf()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        ax = plt.subplot(121)
        ax.set_title("Id vs speed")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.set_ylabel("Id [A]")
        ax.plot(speed, id, label=['Id'])
        # ax.plot(speed, id, 'o', label=['Id'])
        ax = plt.subplot(122)
        ax.set_title("Iq vs speed")
        ax.set_ylabel("Iq [A]")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.plot(speed, iq, label=['Iq'])
        # ax.plot(speed, iq, 'o', label=['Iq'])

        print("Plot test 2")

class Metrics():
    def __init__(self, dt):
        self.dt = dt    # Sampling time [s]

    def settling_time(self, data):
        # Assume average of last 50 points is the steady-state value
        ss_val = np.mean(data[-50:])

        # Find first value out of a 5% steady-state value band
        band = 0.02
        if (np.abs(data - ss_val) < np.abs(band*ss_val)).all():
            index_ss = 0
        else:
            index_ss = np.max(np.argwhere(np.abs(data - ss_val) > np.abs(band*ss_val)))
        return index_ss, ss_val

    def overshoot(self, data, ss_val):
        if np.abs(data[0]) < np.abs(ss_val):
            return np.abs((np.max(data) - ss_val)/ss_val) if np.abs(np.max(data)) > np.abs(ss_val) else 0
        else:
            return np.abs((np.min(data) - ss_val)/ss_val) if np.abs(np.min(data)) < np.abs(ss_val) else 0
    
    def undershoot(self, data, ss_val):
        if np.abs(data[0]) < np.abs(ss_val):
            return np.abs((np.min(data) - ss_val)/ss_val) if np.abs(np.min(data)) < np.abs(data[0]) else 0
        else:
            return np.abs((np.max(data) - ss_val)/ss_val) if np.abs(np.max(data)) > np.abs(data[0]) else 0
    
    def error(self, ref, ss_val):
        return np.abs((ss_val - ref))

class RewardLoggingCallback(EventCallback):
    def __init__(self, csv_file="reward_log.csv", verbose=0):
        super(RewardLoggingCallback, self).__init__(verbose = verbose)
        self.csv_file = csv_file
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if a new episode has ended
        infos = self.locals["infos"]
        for info in infos:
            if "episode" in info:  # Check if episode info is available
                ep_reward = info["episode"]["r"]  # Total episode reward
                ep_length = info["episode"]["l"]  # Episode length
                avg_reward_per_step = ep_reward / ep_length  # Compute avg reward per step

                self.episode_rewards.append(avg_reward_per_step)
                self._save_to_csv(avg_reward_per_step)

        return True

    def _save_to_csv(self, avg_reward):
        df = pd.DataFrame({"average_reward_per_step": [avg_reward]})
        df.to_csv(self.csv_file, mode='a', index=False, header=not hasattr(self, "_header_written"))
        self._header_written = True  # Ensure header is written only once

class DataBasedParameter:
    def __init__(self, Id, Iq, values, interp_method='linear'):
        """
        Initializes the 2D interpolator.

        Args:
            x (array-like): 1D array of Id values.
            y (array-like): 1D array of Iq values.
            z (2D array): Values corresponding to the (Id, Iq) grid.
            interp_method (str): Type of interpolation ('linear', 'cubic', etc.).
        """
        self.Id = Id
        self.Iq = Iq
        self.values = values
        self.interp_method = interp_method
    
    def interp2d(self, Id_new, Iq_new):
        """
        Initializes the 2D interpolator.

        Args:
            x (array-like): 1D array of Id values.
            y (array-like): 1D array of Iq values.
            z (2D array): Values corresponding to (Id, Iq) grid.
            interp_method (str): Type of interpolation ('linear', 'cubic', etc.).
        """
        points = (self.Id, self.Iq)
        return scp.interpolate.interpn(points, self.values, [[Id_new, Iq_new]], method=self.interp_method)[0]

if __name__ == "__main__":
    # # Test PEHardware
    # test_PEHardware = PEHardware(VDC=1200, ts=1e-4, we=100, steps=1000, PWM_type="SVPWM")
    # Vmd, Vmq = [500, 0]
    # Vd, Vq = test_PEHardware.action(Vmd, Vmq)
    # print(Vd, Vq)

    # # Test Metrics
    # test_Metrics = Metrics(dt=1e-4)
    # data = np.linspace(0.0, 10.0, num=1000)
    # random_ss = np.array([0.05*(2*np.random.random() - 1) + 10 for i in range(500)])
    # data = np.append(data, random_ss)
    # index_ss = test_Metrics.settling_time(data)
    # print(data[index_ss])

    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5)
    z = np.array([[i + j for i in x] for j in y])  # Example function z = x + y
    
    interpolator = DataBasedParameter(x, y, z, interp_method='linear')
    result = interpolator.interp2d(5, 5)
    print(result)  # Should interpolate z at (5,5)