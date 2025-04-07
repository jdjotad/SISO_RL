%% ------- SYSTEM PARAMETERS ------- %%
Vdc = 310;  % DC-Link Voltage       [V]
Fs  = 10e3; % PWM Carrier Frequency [Hz]
TCs = 1/Fs; % Control Sampling Time [s]

%% ------- MOTOR PARAMETERS ------- %%
J     = 0;	        % Inertia               [kg*m2]
p     = 4;		    % Pairs of poles        [-]
F     = 0;          % Friction coefficient

%% ------- LOAD MTPA data from 'data_MTPA_PMSM_Prius.mat' ------- %%
% MTPA_data = load("data_MTPA_PMSM_Prius.mat");
% MTPA.Id = MTPA_data.MTPA.Id;
% MTPA.Iq = MTPA_data.MTPA.Iq;
% MTPA.Te = MTPA_data.Te_ref_vec;


%% ------- NON-SATURATION/SATURATION REFERENCE TORQUE ------- %%
% Reference Torque for non-saturated operation
Tref = 60;
Kp_Id = 4.746;
Ki_Id = 5.274e3;
Kp_Iq = 8.861;
Ki_Iq = 9.846e3;
% Reference Torque for saturated operation
% Tref = 120;
% Kp_Id = 2.957;
% Ki_Id = 3.285e3;
% Kp_Iq = 2.004e3;
% Ki_Iq = 9.846e3;
%% PMSM Data
% Hande FEM
% speedam_plecs_config; 
% Hande Greybox
% speedam_plecs_config_v2;

% PLECS dataset
Mot.mag_t = load("look_up_table_based_pmsm_prius_motor_data.mat");
Mot.Rs = 0.015;
Mot.Lss = 0.0001;
Mot.p = 4;

Id0 = 0;
Iq0 = 0;

Mot.thm0 = 0;
% assumes initial position is zero!!!
Mot.is0 = [Id0 (-0.5*Id0 + sqrt(3)/2* Iq0)];	

% %% ------- MPC ------- %%
% % P Q R gains
% P_gain = 1;
% Q_gain = 10;
% R_gain = Q_gain*20;
% 
% % Constraints
% % Output
% ymin = -300;    % [V]
% ymax =  300;    % [V]
% 
% % Control input
% umin = -Vdc/sqrt(3);    % [V]
% umax =  Vdc/sqrt(3);    % [V]
% 
% % Control input rate
% dumin = -1e6;   % [V]
% dumax =  1e6;   % [V]
