clear all
close all
% System parameters
Rs = 29.0808e-3;     % Stator resistance
Ld = 0.91e-3;        % Inductance d-frame [H]
Lq = 1.17e-3;        % Inductance q-frame [H]
A_pm = 0.172312604;  % Flux-linkage due to permanent magnets [Wb]
Vdc = 1200;          % DC bus voltage
we_nom = 200*2*pi;   % Electric nominal speed [rad/s]
I_max = 300;         % Maximum current [A]

tolerance = 1e-1;    % Tolerance for finding the border 

% Set test speed
% we = -we_nom;

% Steady-state equation
% [Vd] = [ Rs     -we*Lq][Id] + [   0   ]
% [Vq]   [we*Ld      Rs ][Iq]   [we*A_pm]

% Voltage limitation
% |Vdq| <= (Vdc/2)^2
Vd = @(Id, Iq, we) Rs*Id - we*Lq*Iq;
Vq = @(Id, Iq, we) Rs*Iq + we*Ld*Id + we*A_pm;
V_dq_quadratic_norm = @(Id, Iq, we) Vd(Id, Iq, we).^2 + Vq(Id, Iq, we).^2 - (Vdc/2)^2;

% Current limitation
I_dq_quadratic_norm = @(Id, Iq) Id.^2 + Iq.^2 - I_max^2;

data_points = 1000;
Id_data = -3*I_max:I_max/data_points:3*I_max;
Iq_data = -3*I_max:I_max/data_points:3*I_max;
[Id_data_grid, Iq_data_grid] = meshgrid(Id_data, Iq_data);

% Define the limits
we = -we_nom;
voltage_limitation = V_dq_quadratic_norm(Id_data_grid,Iq_data_grid, we);
current_limitation = I_dq_quadratic_norm(Id_data_grid,Iq_data_grid);

% Plot the limit
figure;
contour(Id_data_grid, Iq_data_grid, voltage_limitation, [0, 0], 'b', 'LineWidth', 2);
hold on;
contour(Id_data_grid, Iq_data_grid, current_limitation, [0, 0], 'r', 'LineWidth', 2);
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title({'V_d^2 + V_q^2 = (V_{DC}/2)^2', 'I_d^2 + I_q^2 = I_{max}^2'});