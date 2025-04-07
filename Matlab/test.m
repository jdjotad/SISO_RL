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
we_nom = 400*2*pi;   % Electric nominal speed [rad/s]
I_max = 200;         % Maximum current [A]
Lss = 0.0000;        % Leakage inductance [H]

Ldd = @(Id,Iq) interpn(Id_data, Iq_data, Ldd_data, Id, Iq);
Ldq = @(Id,Iq) interpn(Id_data, Iq_data, Ldq_data, Id, Iq);
Lqq = @(Id,Iq) interpn(Id_data, Iq_data, Lqq_data, Id, Iq);
A_pm = @(Id,Iq) interpn(Id_data, Iq_data, A_pm_data, Id, Iq);
Psid = @(Id,Iq) interpn(Id_data, Iq_data, Psid_data, Id, Iq);
Psiq = @(Id,Iq) interpn(Id_data, Iq_data, Psiq_data, Id, Iq);

Psi_d = @(Id, Iq) Ldd(Id,Iq).*Id + Ldq(Id,Iq).*Iq + A_pm(Id,Iq);
Psi_q = @(Id, Iq) Lqq(Id,Iq).*Iq + Ldq(Id,Iq).*Id;

% Current limitation
I_lim = 200;

data_points = 1000;
Id_data_plot = -I_max:I_max/data_points:I_max;
Iq_data_plot = -I_max:I_max/data_points:I_max;
[Id_data_grid, Iq_data_grid] = meshgrid(Id_data_plot, Iq_data_plot);

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
minColorLimit = min([Psiq(Id_data_grid,Iq_data_grid), Psi_q(Id_data_grid,Iq_data_grid)],[],"all");  
maxColorLimit = max([Psiq(Id_data_grid,Iq_data_grid), Psi_q(Id_data_grid,Iq_data_grid)],[],"all");  
subplot(2,1,1);
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

subplot(2,1,2);
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