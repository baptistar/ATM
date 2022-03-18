clear all
close all

addpath(genpath('../../src'))

Z=load('MULT_IDXS_mat_10.txt');


V_writing=1:length(Z);
V_color=(randn(1,length(Z)));


plot_multi_idxs2d(Z)

plot_multi_idxs2d(Z,V_writing)

plot_multi_idxs2d(Z,V_writing,V_color)

plot_multi_idxs2d(Z,V_writing,V_color,'tatata')

