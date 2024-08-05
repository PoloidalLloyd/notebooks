clear;
close all;

% load the data
load("siso_model.mat");

ss_model = mastu_VS_siso_sys;

% control design 
%-----------------------------------------------------------------
% Convert the state space model to a transfer function
[num, den] = ss2tf(ss_model.A, ss_model.B, ss_model.C, ss_model.D);

siso_tf = tf(num, den);

% Due to phase cancellation at 0 between poles and zeroes we can use the minreal() function 
% to remove these from the transfer function

min_tol = 1e-6;
min_siso_tf = minreal(siso_tf, min_tol);

% true if want manual input of control parameters
if 1

    % Define a PID controller and weights
    kp = 621.5;
    ki = 1713.5762;
    kd = 34.1550;

    % Derivative timescale
    td = 0.005;  % 5 milliseconds 
    
    % Integral timescale
    ti = 0.5;  % 500 milliseconds
    
    % Derivative and integral gains based on timescales (uncomment for
    % parameterisation

    % kd = kp * td;  
    % ki = kp / ti;  
    
    controller = pid(kp, ki, kd);

end

% true if want to use the PID tuner
if 0
    pidTuner(min_siso_tf, 'PID');

end

% Define the open loop response (for nyquist diagram)
mastu_open_loop = controller*min_siso_tf;

% Now add this to the system transfer function as feedback
mastu_closed_loop = feedback(controller*min_siso_tf, 1);

info = stepinfo(mastu_closed_loop);
fprintf('The settling time is %f seconds.\n', info.SettlingTime');


sisotool(mastu_closed_loop)


% Check for stability of the closed loop system
isstable(mastu_closed_loop)

show_plot = true;
save_plot = false;

% time = 0:0.01:0.025;
% if show_plot
%     figure;
%     impulse(mastu_closed_loop, time)
% end
% if save_plot
%     saveas(gcf, "/home/johnlloydbaker/Documents/matlab_scripts/plasma_control/figures/vertical_stability_problem/mastu_impulse_response.png")
% end

% if show_plot
%     figure;
%     pzmap(mastu_open_loop)
% end
% if save_plot
%     saveas(gcf, "/home/johnlloydbaker/Documents/matlab_scripts/plasma_control/figures/vertical_stability_problem/mastu_pzmap.png")
% end

if show_plot
    figure;
    bode(mastu_closed_loop)
    margin(mastu_closed_loop)
end
if save_plot
    saveas(gcf, "/home/johnlloydbaker/Documents/matlab_scripts/plasma_control/figures/vertical_stability_problem/mastu_bode.png")
end

if show_plot
    figure;
    nyquist(mastu_open_loop)
    
end
if save_plot
    saveas(gcf, "/home/johnlloydbaker/Documents/matlab_scripts/plasma_control/figures/vertical_stability_problem/mastu_nyquist.png")
end
time = 0:0.01:2;
if show_plot
    figure;
    step(mastu_closed_loop, time)
end
if save_plot
    saveas(gcf, "/home/johnlloydbaker/Documents/matlab_scripts/plasma_control/figures/vertical_stability_problem/mastu_step_response.png")
end
