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

sisotool(min_siso_tf);