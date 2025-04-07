clear all
close all

pmsm_data = load("look_up_table_based_pmsm_prius_motor_data.mat");
Id_data  = pmsm_data.imd;     % Current d-frame [A]
Iq_data  = pmsm_data.imq;     % Current q-frame [A]
Ldd_data = pmsm_data.Lmidd;        % Autoinductance d-frame [H]
Ldq_data = pmsm_data.Lmidq;        % Cross-coupling inductance dq-frame [H]
Lqq_data  = pmsm_data.Lmiqq ;      % Autoinductance q-frame [H]
Psid_data  = pmsm_data.Psid;  % Flux-linkage d-frame [Wb]
Psiq_data  = pmsm_data.Psiq;  % Flux-linkage q-frame [Wb]
A_pm_data  = Psid_data - (Ldd_data.*Id_data' + Ldq_data.*Iq_data);  % Flux-linkage due to permanent magnets [Wb]

% System parameters
Rs = 0.015;     % Stator resistance
Vdc = 1200;          % DC bus voltage
we_nom = 200*2*pi;   % Electric nominal speed [rad/s]
I_max = 200;         % Maximum current [A]
Lss = 0.0001;        % Leakage inductance [H]

Ldd = @(Id,Iq) interpn(Id_data, Iq_data, Ldd_data, Id, Iq);
Ldq = @(Id,Iq) interpn(Id_data, Iq_data, Ldq_data, Id, Iq);
Lqq = @(Id,Iq) interpn(Id_data, Iq_data, Lqq_data, Id, Iq);
A_pm = @(Id,Iq) interpn(Id_data, Iq_data, A_pm_data, Id, Iq);
Psid = @(Id,Iq) interpn(Id_data, Iq_data, Psid_data, Id, Iq);
Psiq = @(Id,Iq) interpn(Id_data, Iq_data, Psiq_data, Id, Iq);

tolerance = 1e-3;    % Tolerance for finding the border 

% Set test speed
% we = -we_nom;

% Steady-state equation
% [Vd] = [ Rs           -we*Lq(Id,Iq)][Id] + [      0       ]
% [Vq]   [we*Ld(Id,Iq)            Rs ][Iq]   [we*A_pm(Id,Iq)]
% which is equivalent to
% [Vd] = [ Rs           -we*Lq(Id,Iq)][Id] + [      0       ]
% [Vq]   [we*Ld(Id,Iq)            Rs ][Iq]   [we*A_pm(Id,Iq)]
% Voltage limitation
% |Vdq| <= (Vdc/2)^2
Psi_d = @(Id, Iq) Ldd(Id,Iq).*Id + Ldq(Id,Iq).*Iq + A_pm(Id,Iq);
Psi_q = @(Id, Iq) (Lqq(Id,Iq) + Lss).*Iq + Ldq(Id,Iq).*Id;

% Vd1 = @(Id, Iq, we) Rs*Id - we*(Lqq(Id,Iq).*Iq + Ldq(Id,Iq).*Id);
% Vq1 = @(Id, Iq, we) Rs*Iq + we*(Ldd(Id,Iq).*Id + Ldq(Id,Iq).*Iq + A_pm(Id,Iq));
Vd2 = @(Id, Iq, we) Rs*Id + we*Lss*Iq - we*Psi_q(Id,Iq);
Vq2 = @(Id, Iq, we) Rs*Iq + we*Lss*Id + we*Psi_d(Id,Iq);
Vd3 = @(Id, Iq, we) Rs*Id + we*Lss*Iq - we*Psiq(Id,Iq);
Vq3 = @(Id, Iq, we) Rs*Iq + we*Lss*Id + we*Psid(Id,Iq);

Vd = @(Id, Iq, we) Vd3(Id, Iq, we);
Vq = @(Id, Iq, we) Vq3(Id, Iq, we);
V_dq_quadratic_norm = @(Id, Iq, we) Vd(Id, Iq, we).^2 + Vq(Id, Iq, we).^2 - (Vdc/2)^2;


% Current limitation
I_lim = 150;
I_dq_quadratic_norm = @(Id, Iq) Id.^2 + Iq.^2 - I_lim^2;

data_points = 1000;
Id_data_plot = -I_max:I_max/data_points:I_max;
Iq_data_plot = -I_max:I_max/data_points:I_max;
[Id_data_grid, Iq_data_grid] = meshgrid(Id_data_plot, Iq_data_plot);

% Define the limits
we = we_nom;
voltage_limitation = V_dq_quadratic_norm(Id_data_grid,Iq_data_grid, we);
current_limitation = I_dq_quadratic_norm(Id_data_grid,Iq_data_grid);

%
Id_voltage_limitation = unique(Id_data_grid(voltage_limitation <= tolerance));
Iq_voltage_limitation = unique(Iq_data_grid(voltage_limitation <= tolerance));
fprintf("Id = [%.2f, %.2f]\nIq=[%.2f, %.2f]\n", ...
         min(Id_voltage_limitation),max(Id_voltage_limitation), ...
         min(Iq_voltage_limitation),max(Iq_voltage_limitation))


% Plot current-dependant values
figure;
subplot(2,2,1);
minColorLimit = min([Ldd(Id_data_grid,Iq_data_grid)],[],"all");  
maxColorLimit = max([Ldd(Id_data_grid,Iq_data_grid)],[],"all");  
h = surf(Id_data_grid, Iq_data_grid, Ldd(Id_data_grid,Iq_data_grid));
set(h,'LineStyle','none')
clim([minColorLimit,maxColorLimit]); 
colorbar;
view(2)
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title({'L_dd [H]'});

subplot(2,2,2);
minColorLimit = min([Ldq(Id_data_grid,Iq_data_grid)],[],"all");  
maxColorLimit = max([Ldq(Id_data_grid,Iq_data_grid)],[],"all");  
h = surf(Id_data_grid, Iq_data_grid, Ldq(Id_data_grid,Iq_data_grid));
set(h,'LineStyle','none')
clim([minColorLimit,maxColorLimit]); 
colorbar;
view(2)
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title({'L_dq [H]'});

subplot(2,2,3);
minColorLimit = min([Lqq(Id_data_grid,Iq_data_grid)],[],"all");  
maxColorLimit = max([Lqq(Id_data_grid,Iq_data_grid)],[],"all");  
h = surf(Id_data_grid, Iq_data_grid, Lqq(Id_data_grid,Iq_data_grid));
set(h,'LineStyle','none')
clim([minColorLimit,maxColorLimit]); 
colorbar;
view(2)
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title({'L_qq [H]'});

% Plot the limit
figure;
C = Id_data_grid.*Iq_data_grid;

minColorLimit = min([Psid(Id_data_grid,Iq_data_grid), Psi_d(Id_data_grid,Iq_data_grid)],[],"all");  
maxColorLimit = max([Psid(Id_data_grid,Iq_data_grid), Psi_d(Id_data_grid,Iq_data_grid)],[],"all");  
subplot(2,2,1);
h = surf(Id_data_grid, Iq_data_grid, Psid(Id_data_grid,Iq_data_grid));
clim([minColorLimit,maxColorLimit]); 
set(h,'LineStyle','none')
view(2)
hold on;
colorbar;
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title('Psi_d DATA');

subplot(2,2,2);
h = surf(Id_data_grid, Iq_data_grid, Psi_d(Id_data_grid,Iq_data_grid));
clim([minColorLimit,maxColorLimit]); 
set(h,'LineStyle','none')
view(2)
hold on;
colorbar;
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title('Psi_d MATLAB');

minColorLimit = min([Psiq(Id_data_grid,Iq_data_grid), Psi_q(Id_data_grid,Iq_data_grid)],[],"all");  
maxColorLimit = max([Psiq(Id_data_grid,Iq_data_grid), Psi_q(Id_data_grid,Iq_data_grid)],[],"all");  
subplot(2,2,3);
h = surf(Id_data_grid, Iq_data_grid, Psiq(Id_data_grid,Iq_data_grid));
clim([minColorLimit,maxColorLimit]); 
set(h,'LineStyle','none')
view(2)
hold on;
colorbar;
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title('Psi_q DATA');

subplot(2,2,4);
h = surf(Id_data_grid, Iq_data_grid, Psi_q(Id_data_grid,Iq_data_grid));
clim([minColorLimit,maxColorLimit]); 
set(h,'LineStyle','none')
view(2)
hold on;
colorbar;
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title('Psi_q MATLAB');

% Plot the limit
figure;
contour(Id_data_grid, Iq_data_grid, voltage_limitation, [0, 0], 'b', 'LineWidth', 2, 'ShowText','on');
hold on;
contour(Id_data_grid, Iq_data_grid, current_limitation, [0, 0], 'r', 'LineWidth', 2, 'ShowText','on');
axis equal;
grid on;
xlabel('I_d');
ylabel('I_q');
title({'V_d^2 + V_q^2 = (V_{DC}/2)^2', 'I_d^2 + I_q^2 = I_{max}^2'});