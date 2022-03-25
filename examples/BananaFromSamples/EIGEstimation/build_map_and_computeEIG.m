clear; close all; clc
addpath(genpath('../../../src'))
sd = 2; rng(sd);

%% Generate data

% define model
sigma_x = 0.5;
sigma_q = 0.1;

% define x,y
x_sample = @(N) sigma_x*randn(N,1);

% define observables
Q1_sample = @(x) -1*x + sigma_q*randn(size(x,1),1);
Q2_sample = @(x) x.^2 + sigma_q*randn(size(x,1),1);

% sample X, Q1,Q2
N_train  = 10000;
X_train  = x_sample(N_train);
Q1_train = Q1_sample(X_train);
Q2_train = Q2_sample(X_train);

%% Build map for each observable

% set max number of iterations
max_iter = 25;

% build map for (X,Q1)
fprintf('Posterior for Q1\n')
[CM_Q1X, output_Q1X] = build_conditional_map(Q1_train, X_train, max_iter);

fprintf('Posterior for Q2\n')
[CM_Q2X, output_Q2X] = build_conditional_map(Q2_train, X_train, max_iter);

%% Generate plots for Q1

% define grid for plotting
xdom = [-2,2];
ydom = [-2,2];
yst = linspace(ydom(1),ydom(2),6);

% evaluate each conditional density
comps = 2;
xx = linspace(xdom(1),xdom(2),1000);
cond_pdf = zeros(length(yst),length(xx));
for l=1:length(yst)
   cond_pdf(l,:) = exp(CM_Q1X.log_pdf([repmat(yst(l),numel(xx),1), xx'],comps));
end

% plot conditional PDF
figure
hold on
for l=2:length(yst)-1
    plot3(xx,repmat(yst(l),numel(xx),1),cond_pdf(l,:), '-k')
end
view([-37.5,40])
plot3(X_train, Q1_train, zeros(size(X_train,1),1),'.', ...
    'Color',[0.8500, 0.3250, 0.0980],'MarkerSize',6)
set(gca,'FontSize',20)
xlabel('$X$','FontSize',24)
ylabel('$Q_1$','FontSize',24)
zlabel('$\pi_{X|Q_1}$','FontSize',24)
xlim(xdom)
ylim(ydom)
hold off
    
%% Generate plots for Q2
   
% define grid for plotting
xdom = [-2,2];
ydom = [-0.5,2];
yst = linspace(ydom(1), ydom(2), 6);

% evaluate each conditional density
comps = 2;
xx = linspace(xdom(1),xdom(2),1000);
cond_pdf = zeros(length(yst),length(xx));
for l=1:length(yst)
   cond_pdf(l,:) = exp(CM_Q2X.log_pdf([repmat(yst(l),numel(xx),1), xx'],comps));
end

% plot conditional PDF
figure
hold on
for l=2:length(yst)-1
    plot3(xx,repmat(yst(l),numel(xx),1),cond_pdf(l,:), '-k')
end
view([-37.5,40])
plot3(X_train, Q2_train, zeros(size(X_train,1),1),'.',...
    'Color',[0.8500, 0.3250, 0.0980],'MarkerSize',6)
set(gca,'FontSize',20)
xlabel('$X$','FontSize',24)
ylabel('$Q_2$','FontSize',24)
zlabel('$\pi_{X|Q_2}$','FontSize',24)
xlim(xdom)
ylim(ydom)
hold off

%% Estimate EIG

% extract test data
N_test  = 100000;
X_test  = x_sample(N_test);
Q1_test = Q1_sample(X_test);
Q2_test = Q2_sample(X_test);

% evaluate log-prior
log_prior = -1/2*log(2*pi*exp(1)*sigma_x^2);

% evaluate log posterior for Q1 at test samples
comps = 2;
log_post = CM_Q1X.log_pdf([Q1_test, X_test],comps);
log_post_quantiles = quantile(log_post, [0.005, 0.995]);
log_post = log_post(log_post > log_post_quantiles(1) & log_post < log_post_quantiles(2));

% evaluate EIG for Q2
EIG_Q1_mean = mean(log_post - log_prior);
EIG_Q1_stde = 1.96*std(log_post - log_prior)/sqrt(size(X_test,1));
fprintf('EIG(X, Q1): %.3f \\pm %.3f \n', EIG_Q1_mean, EIG_Q1_stde);

% evaluate log posterior for Q2 at test samples
comps = 2;
log_post = CM_Q2X.log_pdf([Q2_test, X_test],comps);
log_post_quantiles = quantile(log_post, [0.005, 0.995]);
log_post = log_post(log_post > log_post_quantiles(1) & log_post < log_post_quantiles(2));

% evaluate EIG
EIG_Q2_mean = mean(log_post - log_prior);
EIG_Q2_stde = 1.96*std(log_post - log_prior)/sqrt(size(X_test,1));
fprintf('EIG(X, Q2): %.3f \\pm %.3f \n', EIG_Q2_mean, EIG_Q2_stde);

%% --- Helper functions ---

function [CM, output] = build_conditional_map(Y, X, max_iter)

    % combine samples
    YX = [Y,X];

    % define d and comps
    d = size(YX,2);
    comps = (size(Y,2)+1):d;

    % define reference
    ref = IndependentProductDistribution(repmat({Normal()},1,d));
        
    % define and evaluate Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(YX);
    Z = G.S.evaluate(YX);
    
    % define TM
    TM = identity_map(1:d, ProbabilistHermiteFunction());
    PB = PullbackDensity(TM, ref);
    % run greedy optimization for PB
    [PB, output] = PB.greedy_optimize(Z, [], max_iter, 'Split', comps);

    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);
    
end

% -- END OF FILE --