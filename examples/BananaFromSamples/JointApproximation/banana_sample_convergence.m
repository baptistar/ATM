clear; clc; close all;
sd = 1; rng(sd);
addpath(genpath('../../../src'))

%% -- PLOT APPROXIMATION VS SAMPLE-SIZE --

% define dimension of parameters 
d = 2;

% define sample-sizes
M_vect = [100,300,1000,3000,10000];  % define sample-sizes

% defeine basis
basis = HermiteProbabilistPolyWithLinearization();

% define map order
order = 4;

% define reference
ref = IndependentProductDistribution({Normal(), Normal()});

% define map
S = total_order_map(1:d, basis, order);
PB = PullbackDensity(S, ref);

% define conditioning variable and grid for evaluating density
yst = 2;
xx = linspace(-4,4,100);

% define cell to store results
approx_pi = cell(length(M_vect),1);

for i=1:length(M_vect)
    
    % generate samples
    X = sample_banana(M_vect(i));

    % flip order of samples
    X = fliplr(X);
 
    % standardize samples with a Gaussian linear diagonal map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Xnorm = G.S.evaluate(X);

    % optimize non-linear map
    PB = PB.optimize(Xnorm);

    % compose map with linear transformation for pre-conditioning
    CM = ComposedPullbackDensity({G, PB}, ref);

    % evaluate and plot approximate density
    approx_pi{i} = exp(CM.log_pdf([repmat(yst,length(xx),1), xx.'],2));

end

%% -- PLOT RESULTS --

true_cond_pi_tilde = exp(log_pdf_banana([xx.', yst*ones(length(xx),1)]));
true_cond_pi_norm_const = trapz(xx, true_cond_pi_tilde);
true_cond_pi = true_cond_pi_tilde/true_cond_pi_norm_const;

figure
hold on
plot(xx, true_cond_pi, '-k','LineWidth',2,'DisplayName','Truth')
for i=1:length(M_vect)
    plot(xx, approx_pi{i}, 'DisplayName', ['N = ' num2str(M_vect(i))])
end
legend('show')
xlabel('$x$')
ylabel('$\pi(x|y^*)$')
title(['Order ' num2str(order) ' approximation'])
hold off

%% -- DEFINE MODEL --

function X = sample_banana(N)
    x1 = randn(N,1);
    x2 = x1.^2 + randn(N,1);
    X = [x1, x2];
end

function log_pi = log_pdf_banana(X)
    log_pi_x1 = log(normpdf(X(:,1)));
    log_pi_x2 = log(normpdf(X(:,2) - X(:,1).^2));
    log_pi = log_pi_x1 + log_pi_x2;
end

% -- END OF FILE --