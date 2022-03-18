clear all
close all

addpath(genpath('../../src'))

S=load('./KL_1/offline_maps/offline_map1.mat');

%Z=S.x.S{2}.S.S{3}.multi_idxs;
Z=load('multi_idxs5.txt');

V_writing=1:length(Z);
V_color=V_writing;

plot_multi_idxs(Z,V_writing,V_color)



