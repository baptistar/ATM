%clear; close all; clc
sd = 1; rng(sd);

% set parameters for matrix plot
params = struct;
params.alw = 1.2;    % AxesLineWidth
params.fsz = 14;     % Fontsize
params.lw  = 2;      % LineWidth
params.msz = 4;      % MarkerSize
params.flabels = 24; % Labels font size

%% --- Define test case ---

% define orders, sample size, and MC iterations
MCruns = 2;
d      = 3;
N_vect = 100;
comp   = 1:d;
Nplot  = 1e4;

% plot true density
load(['samples/MoG_d' num2str(d)],'P')
Xtrue = P.sample(1e4);
matrix_plot(Xtrue, [], params);
file_name = ['figures/PDF_true_d' num2str(d)];
print('-depsc', file_name)
close all

% run each test case
for i=1:length(N_vect)
    for j=MCruns

        N = N_vect(i);
        fprintf('Processing adaptive map: d = %d, N = %d, Run = %d\n', d, N, j);
        
        % generate reference samples
        Z = randn(Nplot, d);
        
        % sample from the map
        load(['data/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(j) '.mat'],'CM');
        X = CM.inverse(Z);
        
        save(['figures/PDFdata_d' num2str(d) '_N' num2str(N) '_run' num2str(j)],'X');

        % plot density
        matrix_plot(X, [], params);
        file_name = ['figures/PDF_approx_d' num2str(d) '_N' num2str(N) '_run' num2str(j)];
        print('-depsc', file_name)
        close all

    end
end

% -- END OF FILE --