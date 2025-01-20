import numpy as np

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

# def plot_results():
