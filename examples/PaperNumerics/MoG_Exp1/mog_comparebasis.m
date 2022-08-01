%clear; close all; clc
addpath(genpath('../../../src'))

% define parameters
N = 1e4;
order = 10;

% define density
mu = [-2; 2];
sigma = cat(3,0.5,2);
p = ones(1,2)/2;
gm = gmdistribution(mu,sigma,p);

%% Approximate map
    
% generate samples from gumbel density
X = gm.random(N);

% find the Gaussian approximation
G = GaussianPullbackDensity(1, true);
G = G.optimize(X);
Z = G.S.evaluate(X);

% find the linearized Hermite polynomial approximation 
basis = HermiteProbabilistPolyWithLinearization();
basis.bounds = quantile(Z,[0.01,0.99]).';
S = total_order_map(1, basis, order);
ref = IndependentProductDistribution({Normal()});
PB = PullbackDensity(S, ref);
PB = PB.optimize(Z);
CM_HPolyLin = ComposedPullbackDensity({G.S, PB.S}, ref);

% find the Hermite function approximation 
basis = ProbabilistHermiteFunction();
S = total_order_map(1, basis, order);
ref = IndependentProductDistribution({Normal()});
PB = PullbackDensity(S, ref);
PB = PB.optimize(Z);
CM_HF = ComposedPullbackDensity({G.S, PB.S}, ref);
    
% find the Hermite polynomial approximation 
basis = HermiteProbabilistPoly();
S = total_order_map(1, basis, order);
ref = IndependentProductDistribution({Normal()});
PB = PullbackDensity(S, ref);
PB = PB.optimize(Z);
CM_HPoly = ComposedPullbackDensity({G.S, PB.S}, ref);

%% Plot results

% define domain
dom = [-6,8];
xx = linspace(dom(1),dom(2),100000);

% evaluate true pdf and map
true_pdf = gm.pdf(xx');
true_cdf = gm.cdf(xx');
true_KR  = sqrt(2)*erfinv(2*true_cdf - 1);

% evaluate true function f(x)
dxS_eval = sqrt(2)*sqrt(pi)/2*exp(erfinv(2*true_cdf - 1).^2) * 2.*true_pdf;
ginv = @(x) log(2.^(x) - 1)/log(2);
dxf_eval = ginv(dxS_eval);
[~,zero_idx] = min(abs(xx));
true_f = cumtrapz(xx, dxf_eval); true_f = true_f - true_f(zero_idx);
true_f = true_f + sqrt(2)*erfinv(2*gm.cdf(0) - 1);

% plot density
figure
hold on
plot(xx, true_pdf, '-k', 'DisplayName', 'True PDF', 'LineWidth', 4)
plot(xx, exp(CM_HPolyLin.log_pdf(xx')), 'DisplayName', 'Modified Hermite polynomials', 'LineWidth', 3)
plot(xx, exp(CM_HPoly.log_pdf(xx')), 'DisplayName', 'Hermite polynomials', 'LineWidth', 3)
plot(xx, exp(CM_HF.log_pdf(xx')), 'DisplayName', 'Hermite functions', 'LineWidth', 3)
xlim(dom)
set(gca,'FontSize',24)
xlabel('$x$','FontSize',24)
ylabel('$\pi(x)$','FontSize',24)
h=legend('show','location','northeast');
set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
hold off
print('-depsc','pdf_comparebasis')
close all

% plot map S(x)
figure
hold on
plot(xx, true_KR, '-k', 'DisplayName', 'True KR map', 'LineWidth', 3)
plot(xx, CM_HPolyLin.evaluate(xx'), 'DisplayName', 'Modified Hermite polynomials','LineWidth', 3)
plot(xx, CM_HPoly.evaluate(xx'), 'DisplayName', 'Hermite polynomials', 'LineWidth', 3)
plot(xx, CM_HF.evaluate(xx'), 'DisplayName', 'Hermite functions','LineWidth', 3)
xlim(dom)
ylim([-15,15])
set(gca,'FontSize',24)
xlabel('$x$','FontSize',24)
ylabel('$S(x)$','FontSize',24)
h = legend('show','location','northwest');
set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
hold off
print('-depsc','map_comparebasis')
close all

figure
hold on
plot(xx, true_f, '-k', 'DisplayName', 'True KR map', 'LineWidth', 3)
plot(xx, compute_f(CM_HPolyLin, xx'), 'DisplayName', 'Modified Hermite polynomials','LineWidth', 3)
plot(xx, compute_f(CM_HPoly, xx'), 'DisplayName', 'Hermite polynomials', 'LineWidth', 3)
plot(xx, compute_f(CM_HF, xx'), 'DisplayName', 'Hermite functions','LineWidth', 3)
xlim(dom)
ylim([-15,5])
set(gca,'FontSize',24)
xlabel('$x$','FontSize',24)
ylabel('$f(x)$','FontSize',24)
%h = legend('show','location','northwest');
%set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
hold off
print('-depsc','f_comparebasis')
close all

%% -- Helper function for computing f = R^{-1}(S) with composed S

function f_eval = compute_f(CM, x)

    % apply linear function
    Lx = CM.S{1}.evaluate(x);
    dLx = CM.S{1}.grad_x(x);
    
    % compute derivative
    dxSL_eval = CM.S{2}.S{1}.grad_x(Lx) .* dLx;
    
    % check zero index    
    [val,zero_idx] = min(abs(x));
    if val > 1e-3
        error('Check that x includes zero')
    end
    
    % evaluate integral of nonlinear function of derivative
    dxf_eval = log(2.^(dxSL_eval) - 1)/log(2);
    f_eval = cumtrapz(x, dxf_eval); f_eval = f_eval - f_eval(zero_idx);

    % evaluate map at L(0)
    f0_eval = CM.S{2}.evaluate(CM.S{1}.evaluate(0));  
    f_eval = f0_eval + f_eval;
    
end

% -- END OF FILE --
