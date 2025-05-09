import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy as scp
from stable_baselines3.common.callbacks import EventCallback

"""
Power Electronics and Motor Control Utilities

This module provides classes and functions for electrical engineering applications:
- Clarke-Park transformations for three-phase systems
- Two-level inverter modeling
- PWM (Pulse Width Modulation) implementations
- State-space analysis tools
- Visualization utilities for control system performance
- Data interpolation for parameter lookup
"""

class ClarkeParkTransform:
    """
    Implements Clarke and Park transformations for three-phase power systems.
    These transformations convert between three-phase (abc) coordinates and 
    stationary (αβ0) or rotating (dq0) reference frames.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def abc_to_alpha_beta_zero(phase_a, phase_b, phase_c):
        """
        Convert three-phase quantities to stationary reference frame (αβ0).
        
        Args:
            phase_a: Phase A component
            phase_b: Phase B component
            phase_c: Phase C component
            
        Returns:
            tuple: (alpha, beta, zero) components
        """
        alpha = (2/3) * (phase_a - phase_b/2 - phase_c/2)
        beta = (2/3) * (np.sqrt(3) * (phase_b - phase_c)/2)
        zero = (2/3) * ((phase_a + phase_b + phase_c)/2)
        return alpha, beta, zero
    
    @staticmethod
    def alpha_beta_zero_to_abc(alpha, beta, zero):
        """
        Convert stationary reference frame (αβ0) to three-phase quantities.
        
        Args:
            alpha: Alpha component
            beta: Beta component
            zero: Zero-sequence component
            
        Returns:
            tuple: (phase_a, phase_b, phase_c) components
        """
        phase_a = alpha + zero
        phase_b = -alpha/2 + beta * np.sqrt(3)/2 + zero
        phase_c = -alpha/2 - beta * np.sqrt(3)/2 + zero
        return phase_a, phase_b, phase_c
    
    @staticmethod
    def abc_to_dq0_direct(phase_a, phase_b, phase_c, angle_wt, phase_offset=0):
        """
        Convert three-phase quantities to rotating reference frame (dq0)
        using direct (d-axis aligned) orientation.
        
        Args:
            phase_a: Phase A component
            phase_b: Phase B component
            phase_c: Phase C component
            angle_wt: Rotation angle (ωt)
            phase_offset: Additional phase offset (default=0)
            
        Returns:
            tuple: (direct, quadrature, zero) components
        """
        TWO_PI_THIRD = 2 * np.pi / 3
        SCALE = 2/3
        
        theta = angle_wt + phase_offset
        theta_b = theta - TWO_PI_THIRD
        theta_c = theta + TWO_PI_THIRD
        
        direct = SCALE * (
            phase_a * np.cos(theta) + 
            phase_b * np.cos(theta_b) + 
            phase_c * np.cos(theta_c)
        )
        
        quadrature = SCALE * (
            -phase_a * np.sin(theta) - 
            phase_b * np.sin(theta_b) - 
            phase_c * np.sin(theta_c)
        )
        
        zero = SCALE * (phase_a + phase_b + phase_c) / 2
        return direct, quadrature, zero
    
    @staticmethod
    def abc_to_dq0_quadrature(phase_a, phase_b, phase_c, angle_wt, phase_offset=0):
        """
        Convert three-phase quantities to rotating reference frame (dq0)
        using quadrature (q-axis aligned) orientation.
        
        Args:
            phase_a: Phase A component
            phase_b: Phase B component
            phase_c: Phase C component
            angle_wt: Rotation angle (ωt)
            phase_offset: Additional phase offset (default=0)
            
        Returns:
            tuple: (direct, quadrature, zero) components
        """
        TWO_PI_THIRD = 2 * np.pi / 3
        SCALE = 2/3
        
        theta = angle_wt + phase_offset
        theta_b = theta - TWO_PI_THIRD
        theta_c = theta + TWO_PI_THIRD
        
        direct = SCALE * (
            phase_a * np.sin(theta) + 
            phase_b * np.sin(theta_b) + 
            phase_c * np.sin(theta_c)
        )
        
        quadrature = SCALE * (
            phase_a * np.cos(theta) + 
            phase_b * np.cos(theta_b) + 
            phase_c * np.cos(theta_c)
        )
        
        zero = SCALE * (phase_a + phase_b + phase_c) / 2
        return direct, quadrature, zero
    
    @staticmethod
    def dq0_to_abc_direct(direct, quadrature, zero, angle_wt, phase_offset=0):
        """
        Convert rotating reference frame (dq0) to three-phase quantities
        using direct (d-axis aligned) orientation.
        
        Args:
            direct: Direct axis component
            quadrature: Quadrature axis component
            zero: Zero-sequence component
            angle_wt: Rotation angle (ωt)
            phase_offset: Additional phase offset (default=0)
            
        Returns:
            tuple: (phase_a, phase_b, phase_c) components
        """
        TWO_PI_THIRD = 2 * np.pi / 3
        
        theta = angle_wt + phase_offset
        theta_b = angle_wt - TWO_PI_THIRD + phase_offset
        theta_c = angle_wt + TWO_PI_THIRD + phase_offset
        
        phase_a = direct * np.cos(theta) - quadrature * np.sin(theta) + zero
        phase_b = direct * np.cos(theta_b) - quadrature * np.sin(theta_b) + zero
        phase_c = direct * np.cos(theta_c) - quadrature * np.sin(theta_c) + zero
        
        return phase_a, phase_b, phase_c
    
    @staticmethod
    def dq0_to_abc_quadrature(direct, quadrature, zero, angle_wt, phase_offset=0):
        """
        Convert rotating reference frame (dq0) to three-phase quantities
        using quadrature (q-axis aligned) orientation.
        
        Args:
            direct: Direct axis component
            quadrature: Quadrature axis component
            zero: Zero-sequence component
            angle_wt: Rotation angle (ωt)
            phase_offset: Additional phase offset (default=0)
            
        Returns:
            tuple: (phase_a, phase_b, phase_c) components
        """
        TWO_PI_THIRD = 2 * np.pi / 3
        
        theta = angle_wt + phase_offset
        theta_b = angle_wt - TWO_PI_THIRD + phase_offset
        theta_c = angle_wt + TWO_PI_THIRD + phase_offset
        
        phase_a = direct * np.sin(theta) + quadrature * np.cos(theta) + zero
        phase_b = direct * np.sin(theta_b) + quadrature * np.cos(theta_b) + zero
        phase_c = direct * np.sin(theta_c) + quadrature * np.cos(theta_c) + zero
        
        return phase_a, phase_b, phase_c
    
class TwoLevelInverter:
    """
    Represents a two-level voltage source inverter that converts DC voltage to AC
    through switching operations.
    """
    
    def __init__(self, dc_voltage):
        """
        Initialize the inverter with DC bus voltage.
        
        Args:
            dc_voltage: DC bus voltage value
        """
        self.dc_voltage = dc_voltage
    
    def switches_to_voltage(self, switch_a, switch_b, switch_c):
        """
        Convert switching states to phase voltages.
        
        The switching state can be either 0 (lower switch ON) or 1 (upper switch ON).
        For each phase, the output voltage is:
        - +dc_voltage/2 when the upper switch is ON (switch_x = 1)
        - -dc_voltage/2 when the lower switch is ON (switch_x = 0)
        
        Args:
            switch_a: Phase A switching state (0 or 1), or list of states
            switch_b: Phase B switching state (0 or 1), or list of states
            switch_c: Phase C switching state (0 or 1), or list of states
            
        Returns:
            tuple: (voltage_a, voltage_b, voltage_c) phase voltages
        """
        half_voltage = self.dc_voltage / 2
        
        # Check if inputs are scalar (single switching state)
        is_scalar = (isinstance(switch_a, int) and 
                     isinstance(switch_b, int) and 
                     isinstance(switch_c, int))
        
        if is_scalar:
            # Process single switching state
            voltage_a = half_voltage if switch_a == 1 else -half_voltage
            voltage_b = half_voltage if switch_b == 1 else -half_voltage
            voltage_c = half_voltage if switch_c == 1 else -half_voltage
        else:
            # Process lists of switching states
            voltage_a = [half_voltage if state == 1 else -half_voltage for state in switch_a]
            voltage_b = [half_voltage if state == 1 else -half_voltage for state in switch_b]
            voltage_c = [half_voltage if state == 1 else -half_voltage for state in switch_c]
        
        return voltage_a, voltage_b, voltage_c

class PulseWidthModulation:
    """
    Implements Pulse Width Modulation (PWM) strategies for inverter control.
    Supports Sinusoidal PWM (SPWM) and Space Vector PWM (SVPWM).
    """
    
    def __init__(self, dc_voltage, carrier_steps=100, pwm_strategy="SPWM"):
        """
        Initialize the PWM controller.
        
        Args:
            dc_voltage: DC link voltage [V]
            carrier_steps: Number of steps per carrier period (resolution)
            pwm_strategy: Modulation strategy ("SPWM" or "SVPWM")
        """
        self.dc_voltage = dc_voltage
        self.carrier_steps = carrier_steps
        self.pwm_strategy = pwm_strategy
        self.max_voltage = dc_voltage / 2  # Maximum output voltage amplitude
    
    def normalize_reference(self, voltage_references):
        """
        Normalize voltage references to match carrier range (0 to 1).
        
        Args:
            voltage_references: Three-phase voltage references
            
        Returns:
            np.array: Normalized reference signals (0 to 1 range)
        """
        return np.divide(voltage_references, self.max_voltage) / 2 + 0.5
    
    def generate_carrier(self):
        """
        Generate triangular carrier waveform.
        
        Returns:
            np.array: Triangular carrier waveform (0 to 1 range)
        """
        carrier = np.empty(self.carrier_steps)
        half_period = self.carrier_steps / 2
        
        # Rising edge (0 to half_period)
        carrier[:int(half_period)] = np.linspace(0, 1, int(half_period))
        # Falling edge (half_period to end)
        carrier[int(half_period):] = np.linspace(1, 0, self.carrier_steps - int(half_period))
        
        return carrier
    
    def generate_switching_signals(self, voltage_references):
        """
        Generate switching signals based on the selected PWM strategy.
        
        Args:
            voltage_references: Three-phase voltage references [Va, Vb, Vc]
            
        Returns:
            tuple: (switch_a, switch_b, switch_c) switching signals (0 or 1)
            
        Raises:
            ValueError: If an unsupported PWM strategy is specified
        """
        carrier = self.generate_carrier()
        
        if self.pwm_strategy == "SPWM":
            # Sinusoidal PWM - direct comparison with carrier
            modulation_signals = self.normalize_reference(voltage_references)
            
        elif self.pwm_strategy == "SVPWM":
            # Space Vector PWM - add common mode offset
            v_max = np.max(voltage_references)
            v_min = np.min(voltage_references)
            common_mode_offset = -(v_max + v_min) / 2
            
            # Apply common mode offset and normalize
            adjusted_references = voltage_references + common_mode_offset
            modulation_signals = self.normalize_reference(adjusted_references)
            
        else:
            raise ValueError(f"Unsupported PWM strategy: {self.pwm_strategy}")
        
        # Compare reference signals with carrier to generate switching signals
        switch_a = np.array(modulation_signals[0] >= carrier, dtype=int)
        switch_b = np.array(modulation_signals[1] >= carrier, dtype=int)
        switch_c = np.array(modulation_signals[2] >= carrier, dtype=int)
        
        return switch_a, switch_b, switch_c
    
class PowerElectronicsHardware:
    """
    Simulates a power electronics hardware system with:
    - Clarke-Park transformations
    - PWM controller
    - Two-level inverter
    
    The class handles the complete signal chain from reference voltages
    to actual inverter output voltages.
    """
    
    def __init__(self, dc_voltage, time_step, electrical_speed, carrier_steps=100, pwm_strategy="SPWM"):
        """
        Initialize the power electronics hardware simulator.
        
        Args:
            dc_voltage: DC link voltage [V]
            time_step: Simulation time step [s]
            electrical_speed: Electrical speed [rad/s]
            carrier_steps: PWM carrier steps per period
            pwm_strategy: PWM strategy ("SPWM" or "SVPWM")
        """
        
        # Initialize component objects
        clarke_park = ClarkeParTransform()
        pwm_controller = PulseWidthModulation(dc_voltage, carrier_steps, pwm_strategy)
        inverter = TwoLevelInverter(dc_voltage)
        
        # Store component methods for the signal chain
        self.abc_to_dq = clarke_park.abc_to_dq0_direct
        self.dq_to_abc = clarke_park.dq0_to_abc_direct
        self.generate_pwm = pwm_controller.generate_switching_signals
        self.convert_to_voltage = inverter.switches_to_voltage
        
        # System parameters
        self.time_step = time_step
        self.electrical_speed = electrical_speed
        
        # State variables
        self.current_time = 0.0
    
    def process_control_iteration(self, voltage_d_ref, voltage_q_ref):
        """
        Process one control iteration with the given voltage references.
        
        Performs the following steps:
        1. Update simulation time
        2. Convert d-q references to phase voltages
        3. Generate PWM switching signals
        4. Calculate inverter output voltages
        5. Convert output voltages back to d-q for verification
        
        Args:
            voltage_d_ref: Direct-axis voltage reference [V]
            voltage_q_ref: Quadrature-axis voltage reference [V]
            
        Returns:
            tuple: (voltage_d_actual, voltage_q_actual) - Actual output voltages in d-q frame
        """
        # Update time
        self.current_time += self.time_step
        
        # Calculate current angle
        current_angle = self.electrical_speed * self.current_time
        
        # Convert d-q references to three-phase voltage references
        voltage_a_ref, voltage_b_ref, voltage_c_ref = self.dq_to_abc(
            voltage_d_ref, voltage_q_ref, 0, current_angle)
        
        # Generate PWM switching signals
        switch_a, switch_b, switch_c = self.generate_pwm(
            [voltage_a_ref, voltage_b_ref, voltage_c_ref])
        
        # Convert switching signals to actual voltages
        voltage_a, voltage_b, voltage_c = self.convert_to_voltage(
            switch_a, switch_b, switch_c)
        
        # Calculate average voltages (effective values)
        voltage_a_avg = np.mean(voltage_a)
        voltage_b_avg = np.mean(voltage_b)
        voltage_c_avg = np.mean(voltage_c)
        
        # Convert actual voltages back to d-q for verification
        voltage_d_actual, voltage_q_actual, _ = self.abc_to_dq(
            voltage_a_avg, voltage_b_avg, voltage_c_avg, current_angle)
        
        return voltage_d_actual, voltage_q_actual

class SteadyStateAnalysis:
    """
    Provides methods to calculate steady-state responses and visualize
    operating limits in continuous and discrete domains.
    """
    
    def __init__(self, voltage_limit, current_limit):
        """
        Initialize the state-space analysis tool.
        
        Args:
            voltage_limit: Maximum voltage magnitude in d-q frame
            current_limit: Maximum current magnitude in d-q frame
        """
        self.voltage_limit = voltage_limit
        self.current_limit = current_limit
    
    def analyze_continuous_model(self, state_matrix, input_matrix, disturbance_vector=None, plot_current=False):
        """
        Analyze continuous state-space model and calculate steady-state responses.
        
        For system: dx/dt = A*x + B*u + w (where w is optional disturbance)
        
        Args:
            state_matrix: System matrix A
            input_matrix: Input matrix B
            disturbance_vector: Optional disturbance vector w
            plot_current: Whether to plot current limits
            
        Returns:
            dict: Steady-state operating points at key voltage and current points
        """
        # Create result dictionary
        results = {
            'voltage_points': {},
            'current_points': {}
        }
        
        # Check if disturbance vector is provided
        if disturbance_vector is not None:
            # Define steady-state calculation function with disturbance
            # For dx/dt = A*x + B*u + w, at steady-state: 0 = A*x_ss + B*u_ss + w
            # Therefore x_ss = -A^(-1) * (B*u_ss + w)
            def calc_steady_state_current(voltage):
                if voltage.shape == (2, 1000):  # Handle array input for plotting
                    # Create repeated disturbance vector for array calculations
                    repeated_disturbance = np.kron(np.ones((voltage.shape[1], 1)), disturbance_vector).T
                    return -np.linalg.inv(state_matrix) @ (input_matrix @ voltage + repeated_disturbance)
                else:  # Handle single voltage point
                    return -np.linalg.inv(state_matrix) @ (input_matrix @ voltage + disturbance_vector)
            
            # Define inverse calculation: u_ss = -B^(-1) * (A*x_ss + w)
            def calc_steady_state_voltage(current):
                return -np.linalg.inv(input_matrix) @ (state_matrix @ current + disturbance_vector)
        else:
            # Define steady-state calculation function without disturbance
            # For dx/dt = A*x + B*u, at steady-state: 0 = A*x_ss + B*u_ss
            # Therefore x_ss = -A^(-1) * B*u_ss
            def calc_steady_state_current(voltage):
                return -np.linalg.inv(state_matrix) @ (input_matrix @ voltage)
            
            # Define inverse calculation: u_ss = -B^(-1) * A*x_ss
            def calc_steady_state_voltage(current):
                return -np.linalg.inv(input_matrix) @ (state_matrix @ current)
        
        # Calculate steady-state at characteristic voltage points
        voltage_points = [
            ('v_q_max', np.array([0, self.voltage_limit])),
            ('v_d_max', np.array([self.voltage_limit, 0])),
            ('v_q_min', np.array([0, -self.voltage_limit])),
            ('v_d_min', np.array([-self.voltage_limit, 0])),
            ('v_q1_d1', np.array([self.voltage_limit/np.sqrt(2), self.voltage_limit/np.sqrt(2)])),
            ('v_q1_d-1', np.array([self.voltage_limit/np.sqrt(2), -self.voltage_limit/np.sqrt(2)])),
            ('v_q-1_d1', np.array([-self.voltage_limit/np.sqrt(2), self.voltage_limit/np.sqrt(2)])),
            ('v_q-1_d-1', np.array([-self.voltage_limit/np.sqrt(2), -self.voltage_limit/np.sqrt(2)]))
        ]
        
        # Store steady-state current for each voltage point
        for name, voltage in voltage_points:
            results['voltage_points'][name] = {
                'voltage': voltage,
                'current': calc_steady_state_current(voltage)
            }
        
        # Calculate steady-state at characteristic current points
        current_points = [
            ('i_q_max', np.array([0, self.current_limit])),
            ('i_d_max', np.array([self.current_limit, 0])),
            ('i_q_min', np.array([0, -self.current_limit])),
            ('i_d_min', np.array([-self.current_limit, 0])),
            ('i_q1_d1', np.array([self.current_limit/np.sqrt(2), self.current_limit/np.sqrt(2)])),
            ('i_q1_d-1', np.array([self.current_limit/np.sqrt(2), -self.current_limit/np.sqrt(2)])),
            ('i_q-1_d1', np.array([-self.current_limit/np.sqrt(2), self.current_limit/np.sqrt(2)])),
            ('i_q-1_d-1', np.array([-self.current_limit/np.sqrt(2), -self.current_limit/np.sqrt(2)]))
        ]
        
        # Store steady-state voltage for each current point
        for name, current in current_points:
            results['current_points'][name] = {
                'current': current,
                'voltage': calc_steady_state_voltage(current)
            }
        
        # Create visualization if requested
        if plot_current:
            self._generate_current_plot(calc_steady_state_current, "Continuous State-Space Model")
        
        return results
    
    def analyze_discrete_model(self, state_matrix, input_matrix, disturbance_vector=None, plot_current=False):
        """
        Analyze discrete state-space model and calculate steady-state responses.
        
        For system: x[k+1] = A*x[k] + B*u[k] + w (where w is optional disturbance)
        
        Args:
            state_matrix: Discrete system matrix A
            input_matrix: Discrete input matrix B
            disturbance_vector: Optional discrete disturbance vector w
            plot_current: Whether to plot current limits
            
        Returns:
            dict: Steady-state operating points at key voltage and current points
        """
        # Create result dictionary
        results = {
            'voltage_points': {},
            'current_points': {}
        }
        
        # Identity matrix for calculations
        I = np.eye(state_matrix.shape[0])
        
        # Check if disturbance vector is provided
        if disturbance_vector is not None:
            # Define steady-state calculation function with disturbance
            # For x[k+1] = A*x[k] + B*u[k] + w, at steady-state: x_ss = A*x_ss + B*u_ss + w
            # Therefore x_ss = (I-A)^(-1) * (B*u_ss + w)
            def calc_steady_state_current(voltage):
                if voltage.shape == (2, 1000):  # Handle array input for plotting
                    # Create repeated disturbance vector for array calculations
                    repeated_disturbance = np.kron(np.ones((voltage.shape[1], 1)), disturbance_vector).T
                    return np.linalg.inv(I - state_matrix) @ (input_matrix @ voltage + repeated_disturbance)
                else:  # Handle single voltage point
                    return np.linalg.inv(I - state_matrix) @ (input_matrix @ voltage + disturbance_vector)
            
            # Define inverse calculation: u_ss = B^(-1) * ((I-A)*x_ss - w)
            def calc_steady_state_voltage(current):
                return np.linalg.inv(input_matrix) @ ((I - state_matrix) @ current - disturbance_vector)
        else:
            # Define steady-state calculation function without disturbance
            # For x[k+1] = A*x[k] + B*u[k], at steady-state: x_ss = A*x_ss + B*u_ss
            # Therefore x_ss = (I-A)^(-1) * B*u_ss
            def calc_steady_state_current(voltage):
                return np.linalg.inv(I - state_matrix) @ (input_matrix @ voltage)
            
            # Define inverse calculation: u_ss = B^(-1) * (I-A)*x_ss
            def calc_steady_state_voltage(current):
                return np.linalg.inv(input_matrix) @ ((I - state_matrix) @ current)
        
        # Calculate steady-state at characteristic voltage points
        voltage_points = [
            ('v_q_max', np.array([0, self.voltage_limit])),
            ('v_d_max', np.array([self.voltage_limit, 0])),
            ('v_q_min', np.array([0, -self.voltage_limit])),
            ('v_d_min', np.array([-self.voltage_limit, 0])),
            ('v_q1_d1', np.array([self.voltage_limit/np.sqrt(2), self.voltage_limit/np.sqrt(2)])),
            ('v_q1_d-1', np.array([self.voltage_limit/np.sqrt(2), -self.voltage_limit/np.sqrt(2)])),
            ('v_q-1_d1', np.array([-self.voltage_limit/np.sqrt(2), self.voltage_limit/np.sqrt(2)])),
            ('v_q-1_d-1', np.array([-self.voltage_limit/np.sqrt(2), -self.voltage_limit/np.sqrt(2)]))
        ]
        
        # Store steady-state current for each voltage point
        for name, voltage in voltage_points:
            results['voltage_points'][name] = {
                'voltage': voltage,
                'current': calc_steady_state_current(voltage)
            }
        
        # Calculate steady-state at characteristic current points
        current_points = [
            ('i_q_max', np.array([0, self.current_limit])),
            ('i_d_max', np.array([self.current_limit, 0])),
            ('i_q_min', np.array([0, -self.current_limit])),
            ('i_d_min', np.array([-self.current_limit, 0])),
            ('i_q1_d1', np.array([self.current_limit/np.sqrt(2), self.current_limit/np.sqrt(2)])),
            ('i_q1_d-1', np.array([self.current_limit/np.sqrt(2), -self.current_limit/np.sqrt(2)])),
            ('i_q-1_d1', np.array([-self.current_limit/np.sqrt(2), self.current_limit/np.sqrt(2)])),
            ('i_q-1_d-1', np.array([-self.current_limit/np.sqrt(2), -self.current_limit/np.sqrt(2)]))
        ]
        
        # Store steady-state voltage for each current point
        for name, current in current_points:
            results['current_points'][name] = {
                'current': current,
                'voltage': calc_steady_state_voltage(current)
            }
        
        # Create visualization if requested
        if plot_current:
            self._generate_current_plot(calc_steady_state_current, "Discrete State-Space Model")
        
        return results
    
    def _generate_current_plot(self, calc_steady_state_current, title):
        """
        Generate a plot showing current boundaries due to voltage limitations.
        
        Args:
            calc_steady_state_current: Function to calculate steady-state current
            title: Plot title
        """
        # Generate voltage points on unit circle (normalized)
        v_d_normalized = np.linspace(-1, 1, 1000)
        v_q_positive = np.sqrt(1 - np.power(v_d_normalized, 2))
        v_q_negative = -v_q_positive
        
        # Scale by voltage limit and organize into array format
        vdq_positive = self.voltage_limit * np.array([v_d_normalized, v_q_positive])
        vdq_negative = self.voltage_limit * np.array([v_d_normalized, v_q_negative])
        
        # Calculate resulting steady-state currents
        current_positive = calc_steady_state_current(vdq_positive)
        current_negative = calc_steady_state_current(vdq_negative)
        
        # Combine current points for complete boundary
        i_d = np.concatenate((current_positive[0], current_negative[0]))
        i_q = np.concatenate((current_positive[1], current_negative[1]))
        
        # Create figure
        plt.figure(figsize=(8, 8))
        
        # Plot current boundary due to voltage limitation
        plt.plot(i_d, i_q, 'b-', label="Current boundary from voltage limitation")
        
        # Plot maximum current circle
        theta = np.linspace(0, 2*np.pi, 1000)
        i_d_circle = self.current_limit * np.cos(theta)
        i_q_circle = self.current_limit * np.sin(theta)
        plt.plot(i_d_circle, i_q_circle, 'r--', label="Maximum current circle")
        
        # Add labels and title
        plt.xlabel("i_d [A]")
        plt.ylabel("i_q [A]")
        plt.title(title)
        plt.grid(True)
        plt.axis('equal')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=2)
        
class ControlSystemMetrics:
    def __init__(self, sampling_time):
        """
        Initialize metrics calculator for control system analysis.
        
        Args:
            sampling_time: Sampling time in seconds
        """
        self.sampling_time = sampling_time
    
    def calculate_settling_time(self, response_data):
        """
        Calculate the settling time of a step response.
        
        Args:
            response_data: Array of system response values
            
        Returns:
            settling_index: Index where response settles within tolerance
            steady_state_value: Final steady-state value
        """
        # Assume average of last 50 points is the steady-state value
        steady_state_value = np.mean(response_data[-50:])
        
        # Find first value outside of a 2% steady-state value band
        tolerance = 0.02
        tolerance_band = np.abs(tolerance * steady_state_value)
        
        if (np.abs(response_data - steady_state_value) < tolerance_band).all():
            settling_index = 0
        else:
            settling_index = np.max(np.argwhere(np.abs(response_data - steady_state_value) > tolerance_band))
        
        return settling_index, steady_state_value
    
    def calculate_overshoot(self, response_data, steady_state_value):
        """
        Calculate percentage overshoot of response.
        
        Args:
            response_data: Array of system response values
            steady_state_value: Final steady-state value
            
        Returns:
            overshoot: Normalized overshoot as a ratio
        """
        initial_value = response_data[0]
        
        if np.abs(initial_value) < np.abs(steady_state_value):
            # Step-up response
            max_value = np.max(response_data)
            if np.abs(max_value) > np.abs(steady_state_value):
                return np.abs((max_value - steady_state_value) / steady_state_value)
            return 0
        else:
            # Step-down response
            min_value = np.min(response_data)
            if np.abs(min_value) < np.abs(steady_state_value):
                return np.abs((min_value - steady_state_value) / steady_state_value)
            return 0
    
    def calculate_undershoot(self, response_data, steady_state_value):
        """
        Calculate percentage undershoot of response.
        
        Args:
            response_data: Array of system response values
            steady_state_value: Final steady-state value
            
        Returns:
            undershoot: Normalized undershoot as a ratio
        """
        initial_value = response_data[0]
        
        if np.abs(initial_value) < np.abs(steady_state_value):
            # Step-up response
            min_value = np.min(response_data)
            if np.abs(min_value) < np.abs(initial_value):
                return np.abs((min_value - steady_state_value) / steady_state_value)
            return 0
        else:
            # Step-down response
            max_value = np.max(response_data)
            if np.abs(max_value) > np.abs(initial_value):
                return np.abs((max_value - steady_state_value) / steady_state_value)
            return 0
    
    def calculate_steady_state_error(self, reference_value, steady_state_value):
        """
        Calculate absolute steady-state error.
        
        Args:
            reference_value: Desired reference/setpoint value
            steady_state_value: Actual steady-state value achieved
            
        Returns:
            error: Absolute steady-state error
        """
        return np.abs(steady_state_value - reference_value)
    
class PlotUtility:
    """
    Visualization tools for electrical engineering systems.
    Provides methods for plotting single-phase and three-phase system behavior,
    including state variables, control actions, and performance metrics.
    """
    
    def __init__(self):
        """Initialize the plotting utility."""
        pass
    
    def plot_single_phase(self, idx, observations, actions, reward, env_name, model_name, reward_type):
        """
        Plot single-phase system behavior with states, actions, and rewards.
        
        Args:
            idx: Index or identifier for the plot
            observations: Array of system states/observations
            actions: Array of control actions
            reward: Array of reward values
            env_name: Environment name for file organization
            model_name: Model name for file naming
            reward_type: Type of reward function used
            
        Returns:
            None (saves plot to file)
        """
        plt.close()
        plt.suptitle(f"Reward: {reward_type}\n")
        
        # Configure figure
        fig = plt.gcf()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        
        # Plot State
        ax = plt.subplot(131)
        ax.set_title("State vs step")
        ax.plot(observations, label=['I', 'Iref'])
        self._adjust_subplot_layout(ax)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        
        # Plot action
        ax = plt.subplot(132)
        ax.set_title("Action vs step")
        ax.plot(actions, label=['V'])
        self._adjust_subplot_layout(ax)
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                  ncol=2, fancybox=True, shadow=True)
        
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        self._adjust_subplot_layout(ax)
        
        # Save figure
        self._save_plot(env_name, reward_type, model_name, idx)
        plt.pause(0.001)  # pause a bit so that plots are updated
    
    def plot_three_phase(self, idx, observations, actions, reward, env_name, model_name, 
                         reward_type, speed=None, save=False, show=False, mtpa=None ):
        """
        Plot three-phase system behavior with states, actions, and rewards.
        
        Args:
            idx: Index or identifier for the plot
            observations: Array of system states/observations
            actions: Array of control actions
            reward: Array of reward values
            env_name: Environment name for file organization
            model_name: Model name for file naming
            reward_type: Type of reward function used
            speed: Optional rotational speed value for title
            save: Whether to save the plot to file
            show: Whether to display the plot
            
        Returns:
            None (saves/shows plot as specified)
        """
        plt.close()
        
        # Add speed to title if provided
        if speed is not None:
            plt.suptitle(f"Reward: {reward_type}\nSpeed = {speed} [rad/s]")
        else:
            plt.suptitle(f"Reward: {reward_type}")
            
        # Configure figure
        fig = plt.gcf()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        
        if env_name == "PMSMTC":
            # Plot State
            ax = plt.subplot(131)
            ax.set_title("State vs step")
            ax.plot(observations, label=['Te', 'Teref', 'Id', 'Iq'])

            if mtpa is not None:
                mtpa = np.ones((observations.shape[0],1)) * mtpa
                ax.plot(mtpa, "--", label=['Id MTPA', 'Iq MTPA'])
            self._adjust_subplot_layout(ax)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=2, fancybox=True, shadow=True)
            # Plot action
            ax = plt.subplot(132)
            ax.set_title("Action vs step")
            ax.plot(actions, label=['Vd', 'Vq'])
            self._adjust_subplot_layout(ax)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=2, fancybox=True, shadow=True) 
        elif env_name == "PMSMTCABC":
            # Plot State
            ax = plt.subplot(131)
            ax.set_title("State vs step")
            ax.plot(observations, label=['Te', 'Teref', 'Ia', 'Ib', 'Ic'])

            if mtpa is not None:
                mtpa = np.ones((observations.shape[0],1)) * mtpa
                ax.plot(mtpa, "--", label=['Id MTPA', 'Iq MTPA'])
            self._adjust_subplot_layout(ax)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=3, fancybox=True, shadow=True)
            # Plot action
            ax = plt.subplot(132)
            ax.set_title("Action vs step")
            ax.plot(actions, label=['Va', 'Vb', 'Vc'])
            self._adjust_subplot_layout(ax)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=2, fancybox=True, shadow=True)
        else:
            # Plot State
            ax = plt.subplot(131)
            ax.set_title("State vs step")
            ax.plot(observations, label=['Id', 'Iq', 'Idref', 'Iqref'])
            self._adjust_subplot_layout(ax)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=2, fancybox=True, shadow=True)
            # Plot action
            ax = plt.subplot(132)
            ax.set_title("Action vs step")
            ax.plot(actions, label=['Vd', 'Vq'])
            self._adjust_subplot_layout(ax)
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25),
                        ncol=2, fancybox=True, shadow=True)
        
        # Plot reward
        ax = plt.subplot(133)
        ax.set_title("Reward vs step")
        ax.plot(reward)
        self._adjust_subplot_layout(ax)
        
        # Save if requested
        if save:
            self._save_plot(env_name, reward_type, model_name, idx)
        
        # Show if requested
        if show:
            plt.pause(0.001)
            plt.show()
    
    def plot_current_speed_relationship(self, id_current, iq_current, speed):
        """
        Plot the relationship between d-q currents and rotational speed.
        Creates vector field visualizations showing the relationship dynamics.
        
        Args:
            id_current: Array of d-axis current values
            iq_current: Array of q-axis current values
            speed: Array of rotational speed values
            
        Returns:
            None (displays plots)
        """
        # Calculate differentials for vector field
        u = np.diff(speed)
        v = np.diff(id_current)
        w = np.diff(iq_current)
        
        # Calculate positions for vector arrows
        pos_x = speed[:-1] + u/2
        pos_y = id_current[:-1] + v/2
        pos_w = iq_current[:-1] + w/2
        
        # Normalize vector magnitudes
        norm_id = np.sqrt(u**2 + v**2)
        norm_iq = np.sqrt(u**2 + w**2)
        
        # First figure with vector fields
        plt.figure()
        fig = plt.gcf()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        
        # Id vs speed with vector field
        ax = plt.subplot(121)
        ax.set_title("Id vs speed")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.set_ylabel("Id [A]")
        ax.plot(speed, id_current, marker="o")
        ax.quiver(pos_x, pos_y, u/norm_id, v/norm_id, angles="xy", zorder=5, pivot="mid")
        
        # Iq vs speed with vector field
        ax = plt.subplot(122)
        ax.set_title("Iq vs speed")
        ax.set_ylabel("Iq [A]")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.plot(speed, iq_current, marker="o")
        ax.quiver(pos_x, pos_w, u/norm_iq, w/norm_iq, angles="xy", zorder=5, pivot="mid")
        
        # Second figure with line plots only
        plt.figure()
        fig = plt.gcf()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        
        # Id vs speed line plot
        ax = plt.subplot(121)
        ax.set_title("Id vs speed")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.set_ylabel("Id [A]")
        ax.plot(speed, id_current, label=['Id'])
        
        # Iq vs speed line plot
        ax = plt.subplot(122)
        ax.set_title("Iq vs speed")
        ax.set_ylabel("Iq [A]")
        ax.set_xlabel("Normalized electrical speed [rad/s]")
        ax.plot(speed, iq_current, label=['Iq'])
    
    def _adjust_subplot_layout(self, ax):
        """
        Helper method to adjust subplot positioning for consistent layout.
        
        Args:
            ax: Matplotlib axis object to adjust
            
        Returns:
            None (modifies axis in place)
        """
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                       box.width, box.height * 0.9])
    
    def _save_plot(self, env_name, reward_type, model_name, idx):
        """
        Helper method to save plot to file with proper directory structure.
        
        Args:
            env_name: Environment name for directory
            reward_type: Reward type for directory
            model_name: Model name for filename
            idx: Index for filename
            
        Returns:
            None (saves file to disk)
        """
        # Create directory if it doesn't exist
        save_dir = f"plots/{env_name}/{reward_type}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save figure
        plt.savefig(f"{save_dir}/{model_name}_{idx}.pdf", bbox_inches='tight')


class PerformanceMetrics:
    """
    Analysis tools for evaluating control system performance.
    Provides methods for calculating common control metrics such as
    settling time, overshoot, undershoot, and steady-state error.
    """
    
    def __init__(self, dt):
        """
        Initialize performance metrics calculator.
        
        Args:
            dt: Sampling time in seconds
        """
        self.dt = dt    # Sampling time [s]
    
    def settling_time(self, data):
        """
        Calculate the settling time of a response.
        Uses a 2% band around steady-state value to determine settling.
        
        Args:
            data: Array of time-series data
            
        Returns:
            tuple: (index of settling time, steady-state value)
        """
        # Assume average of last 50 points is the steady-state value
        ss_val = np.mean(data[-50:])
        
        # Find first value out of a 2% steady-state value band
        band = 0.02
        if (np.abs(data - ss_val) < np.abs(band*ss_val)).all():
            index_ss = 0  # Already settled from beginning
        else:
            # Find last time system was outside settling band
            index_ss = np.max(np.argwhere(np.abs(data - ss_val) > np.abs(band*ss_val)))
            
        return index_ss, ss_val
    
    def overshoot(self, data, ss_val):
        """
        Calculate the percentage overshoot in a response.
        
        Args:
            data: Array of time-series data
            ss_val: Steady-state value
            
        Returns:
            float: Percentage overshoot (0 if none)
        """
        # Different calculation based on whether initial value is below or above steady state
        if np.abs(data[0]) < np.abs(ss_val):
            # Starting below steady state
            return np.abs((np.max(data) - ss_val)/ss_val) if np.abs(np.max(data)) > np.abs(ss_val) else 0
        else:
            # Starting above steady state
            return np.abs((np.min(data) - ss_val)/ss_val) if np.abs(np.min(data)) < np.abs(ss_val) else 0
    
    def undershoot(self, data, ss_val):
        """
        Calculate the percentage undershoot in a response.
        
        Args:
            data: Array of time-series data
            ss_val: Steady-state value
            
        Returns:
            float: Percentage undershoot (0 if none)
        """
        # Different calculation based on whether initial value is below or above steady state
        if np.abs(data[0]) < np.abs(ss_val):
            # Starting below steady state
            return np.abs((np.min(data) - ss_val)/ss_val) if np.abs(np.min(data)) < np.abs(data[0]) else 0
        else:
            # Starting above steady state
            return np.abs((np.max(data) - ss_val)/ss_val) if np.abs(np.max(data)) > np.abs(data[0]) else 0
    
    def error(self, ref, ss_val):
        """
        Calculate the absolute steady-state error.
        
        Args:
            ref: Reference (setpoint) value
            ss_val: Achieved steady-state value
            
        Returns:
            float: Absolute error
        """
        return np.abs(ss_val - ref)
    
    def performance_summary(self, data, reference=None):
        """
        Generate a comprehensive performance summary.
        
        Args:
            data: Array of time-series data
            reference: Reference (setpoint) value (optional)
            
        Returns:
            dict: Dictionary of performance metrics
        """
        idx_settling, ss_value = self.settling_time(data)
        settling_time_sec = idx_settling * self.dt
        
        metrics = {
            'settling_time_idx': idx_settling,
            'settling_time_sec': settling_time_sec,
            'steady_state_value': ss_value,
            'overshoot_pct': self.overshoot(data, ss_value) * 100,
            'undershoot_pct': self.undershoot(data, ss_value) * 100
        }
        
        if reference is not None:
            metrics['steady_state_error'] = self.error(reference, ss_value)
            metrics['steady_state_error_pct'] = (self.error(reference, ss_value) / reference) * 100 if reference != 0 else float('inf')
            
        return metrics


class RewardLoggingCallback(EventCallback):
    """
    Callback for logging rewards during reinforcement learning training.
    Extends the EventCallback from stable-baselines3 to track and record
    average rewards over specified intervals.
    """
    
    def __init__(self, csv_file="reward_log.csv", log_interval=1, verbose=0):
        """
        Initialize the reward logging callback.
        
        Args:
            csv_file: Path to CSV output file
            log_interval: Number of episodes to average rewards over
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        super(RewardLoggingCallback, self).__init__(verbose=verbose)
        self.csv_file = csv_file
        self.log_interval = log_interval
        self.avg_rewards = []
        
        # Episode tracking variables
        self.episode_counter = 0
        self.episode_acc_rewards = 0
        self._header_written = False
    
    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        Tracks episode rewards and logs average rewards at specified intervals.
        
        Returns:
            bool: Whether training should continue
        """
        infos = self.locals["infos"]
        
        # Process episode information if available
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_acc_rewards += ep_reward
                
                self.episode_counter += 1
                
                # When we've reached the logging interval, record the average
                if self.episode_counter == self.log_interval:
                    avg_reward = self.episode_acc_rewards / self.log_interval
                    self.avg_rewards.append(avg_reward)
                    
                    # Reset counters for next interval
                    self.episode_counter = 0
                    self.episode_acc_rewards = 0
        
        return True
    
    def on_training_end(self):
        """
        Called at the end of training.
        Saves accumulated reward data to CSV.
        """
        self._save_to_csv()
        super(RewardLoggingCallback, self).on_training_end()
    
    def _save_to_csv(self):
        """
        Save recorded average rewards to CSV file.
        """
        df = pd.DataFrame({"average_reward_per_step": self.avg_rewards})
        df.to_csv(self.csv_file, mode='w', index=False, header=not self._header_written)
        self._header_written = True  # Ensure header is written only once


class DataBasedParameter:
    """
    Two-dimensional interpolation for parameter lookup.
    Used for estimating values based on tabular data, particularly
    useful for motor and electrical system parameters that vary with
    operating conditions.
    """
    
    def __init__(self, Id, Iq, values, interp_method='linear'):
        """
        Initialize the 2D parameter interpolator.
        
        Args:
            Id: 1D array of d-axis current values
            Iq: 1D array of q-axis current values
            values: 2D array of parameter values corresponding to (Id, Iq) grid
            interp_method: Interpolation method ('linear', 'cubic', etc.)
        """
        self.Id = Id
        self.Iq = Iq
        self.values = values
        self.interp_method = interp_method
    
    def interp2d(self, Id_new, Iq_new):
        """
        Interpolate parameter value at specified (Id, Iq) point.
        
        Args:
            Id_new: d-axis current value to interpolate at
            Iq_new: q-axis current value to interpolate at
            
        Returns:
            float: Interpolated parameter value
        """
        points = (self.Id, self.Iq)
        return scp.interpolate.interpn(points, self.values, [[Id_new, Iq_new]], method=self.interp_method)[0]
    
    def get_parameter_map(self, resolution=100):
        """
        Generate a full parameter map across the Id-Iq space.
        
        Args:
            resolution: Number of points in each dimension
            
        Returns:
            tuple: (Id_grid, Iq_grid, parameter_values_grid)
        """
        # Create grid points
        Id_min, Id_max = np.min(self.Id), np.max(self.Id)
        Iq_min, Iq_max = np.min(self.Iq), np.max(self.Iq)
        
        Id_grid = np.linspace(Id_min, Id_max, resolution)
        Iq_grid = np.linspace(Iq_min, Iq_max, resolution)
        
        # Create meshgrid for visualization
        Id_mesh, Iq_mesh = np.meshgrid(Id_grid, Iq_grid)
        values_grid = np.zeros_like(Id_mesh)
        
        # Fill grid with interpolated values
        for i in range(resolution):
            for j in range(resolution):
                values_grid[i, j] = self.interp2d(Id_mesh[i, j], Iq_mesh[i, j])
        
        return Id_mesh, Iq_mesh, values_grid
    
    def visualize_parameter_map(self, title=None, colormap='viridis', resolution=50):
        """
        Visualize the parameter map as a 2D contour plot.
        
        Args:
            title: Plot title (optional)
            colormap: Matplotlib colormap name
            resolution: Grid resolution for visualization
            
        Returns:
            tuple: (figure, axis) Matplotlib objects
        """
        Id_mesh, Iq_mesh, values_grid = self.get_parameter_map(resolution)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(Id_mesh, Iq_mesh, values_grid, 50, cmap=colormap)
        
        ax.set_xlabel('Id [A]')
        ax.set_ylabel('Iq [A]')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Parameter Map')
            
        plt.colorbar(contour, ax=ax)
        
        return fig, ax