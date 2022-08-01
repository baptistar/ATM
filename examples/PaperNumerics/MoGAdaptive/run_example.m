%clear; close all; clc
addpath(genpath('../../../src'))

% define folders
if ~exist('./samples', 'dir')
  mkdir('samples');
end
if ~exist('./data', 'dir')
  mkdir('data');
end
if ~exist('./figures', 'dir')
  mkdir('figures');
end

% define problems, sample sizes and number of runs
d_vect = 3;
N_vect = floor(logspace(1,4,25));
N_vect = N_vect(1:2:end);
MCruns = 1:20;

% generate data
generate_samples

% run non-adaptive code
run_non_adaptive

% run adaptive code
run_adaptive_cv

% plot likelihood
plot_loglik
