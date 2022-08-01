%clear; close all; clc
addpath(genpath('../../../src/'))
sd = 1; rng(sd);

% setup density
m1 = 0;
m2 = 0;
sigma1 = 1;
sigma2 = 0.05;
pdf = @(x) 0.5*normpdf(x,m1,sigma1) + 0.5*normpdf(x,m2,sigma2);
cdf = @(x) 0.5*normcdf(x,m1,sigma1) + 0.5*normcdf(x,m2,sigma2);
sample = @(N) [sigma1*randn(N/2,1) + m1; sigma2*randn(N/2,1) + m2];

% define true map
S_t = @(x) sqrt(2)*erfinv(2*cdf(x) - 1);

%% -- Define data --

Ntrain = 10000;
Ntest = 10000;
X = sample(Ntrain);
Xtest = sample(Ntest);

% gaussianize data
G = GaussianPullbackDensity(1, true);
G = G.optimize(X);
Z = G.S.evaluate(X);
Ztest = G.S.evaluate(Xtest);

%% -- Define basis --

% define basis
basis = RickerWavelet();
basis.sigma = 1;
basis.f_dom = [-4,4];%quantile(Z,[0.001,0.999]);
basis.psi_dom = [-.5,.5];

% plot the basis functions at each level
xx = linspace(-8,8,1000).';
figure('position',[0,0,600,700])
max_level = 6;
midx = zeros(0,2);
for j=0:max_level
     m_idx_j = (0:2^j-1).';
     m_idx_j = [j*ones(size(m_idx_j,1),1) m_idx_j];
     midx = [midx; m_idx_j];
     subplot(max_level+1,1,j+1)
     precomp = Wprecomp(basis);
     precomp = precomp.setup(xx, m_idx_j);
     plot(xx, precomp.Psi);
     ylabel(['Level ' num2str(j)])
     xlim([-6,6])
     set(gca,'FontSize',14)
end

%% -- Optimize map using wavelets --

% define reference
ref = Normal();

% define number of terms
max_terms = 15;

% optimize coeffs
[opt_c, T, train_err, test_err] = ATM(Z, Ztest, ref, basis, max_terms);

%% -- Optimize map using polynomials --

% define array to store results
test_err_Hpoly = zeros(max_terms,1);
ref_I = IndependentProductDistribution({ref});

for i=1:max_terms
    disp(i)
    % find map using Hermite polynomials
    basis_poly = HermiteProbabilistPolyWithLinearization();
    basis_poly.bounds = quantile(Z,[0.01,0.99]).';
    S = total_order_map(1, basis_poly, i);
    PB = PullbackDensity(S, ref_I);
    [PB, ~] = PB.optimize(Z);
    loglik_test = PB.log_pdf(Ztest); loglik_test(isinf(loglik_test)) = [];
    test_err_Hpoly(i) = -1 * mean(loglik_test);
end

%% -- Post-process results --

%xx = linspace(basis.f_dom(1), basis.f_dom(2), 500).';
xx = linspace(-4,4,500).';
precomp_eval = Wprecomp(basis);
precomp_eval = precomp_eval.setup(xx, T.m_idxs);

% evaluate PDFs
[Sx_approx,dSx_approx] = evaluate_map(opt_c, precomp_eval);
pdf_approx = exp(ref.log_pdf(Sx_approx) + log(dSx_approx) + G.S.logdet_Jacobian(0));
pdf_approx_poly = exp(PB.log_pdf(xx) + G.S.logdet_Jacobian(0));

figure
hold on
entropy = mean(log(pdf(Xtest)));
G_jac = mean(G.S.logdet_Jacobian(Xtest));
plot(1:max_terms, entropy + test_err_Hpoly - G_jac, 'DisplayName', 'Polynomials', 'LineWidth', 3)
plot(1:max_terms, entropy + test_err - G_jac, 'DisplayName', 'Wavelets', 'LineWidth', 3)
set(gca,'FontSize',24)
xlabel('Number of coefficients, $m$', 'FontSize', 22)
ylabel('$D_{KL}(\pi||\widehat{S}^\sharp\eta)$', 'FontSize', 22)
h = legend('show','location','southwest');
set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
xlim([1,15])
set(gca,'YScale','log')
hold off
print('-depsc','KL_wavelets_polynomials')
close all

figure
hold on
plot(xx, pdf_approx_poly, 'DisplayName', 'Polynomials', 'LineWidth', 3)
plot(xx, pdf_approx, 'DisplayName', 'Wavelets', 'LineWidth', 3)
p=plot(xx, pdf(xx), '-k', 'DisplayName', 'True PDF', 'LineWidth', 3);
uistack(p,'bottom')
%histogram(X,100,'normalization','pdf')
set(gca,'FontSize',24)
xlabel('$x$', 'FontSize', 22)
ylabel('$\pi(x)$', 'FontSize', 22)
h = legend('show');
set(h, 'FontSize', 22)
set(gca,'LineWidth',2)
xlim([-3,3])
ylim([0,5])
hold off
print('-depsc','density_wavelets_polynomials')
close all

%% -- Helper Functions --

function g = softplus(x)
    g = log(1 + 2.^(x))/log(2);
end
function g = dxsoftplus(x)
    g = 1./(1 + 2.^(-x));
end
function g = d2xsoftplus(x)
    g = log(2)*2.^(-x)./(1 + 2.^(-1*x)).^2;
end

function [coeff, T, train_err, test_err] = ATM(X, Xtest, ref, basis, max_terms)

    % intialize multi-indices and errors
    train_err = [];
    test_err = [];
    
    % define tree
    midx0 = [0,0];
    T = BinaryTree(midx0);
    
    % define precomp objects    
    precomp_train = Wprecomp(basis);
    precomp_train = precomp_train.setup(X, midx0);
    precomp_test  = Wprecomp(basis);
    precomp_test  = precomp_test.setup(Xtest, midx0);
    
    % define initial coefficients
    %coeff = 0;
    coeff = optimize_map(ref, precomp_train, 0);

    while(size(T.m_idxs,1) <= max_terms)
                
        % find reduced margin
        margin = T.get_margin();
        
        % update precomp_margin object
        precomp_margin = copy(precomp_train);
        precomp_margin = precomp_margin.update( margin );

        % compute gradients with respect to all coefficients
        coeff_margin = [coeff; zeros(size(margin,1),1)];
        coeff_idx = length(coeff)+1:length(coeff)+size(margin,1);
        %[~,dcL] = negative_log_likelihood_t(coeff_margin, ref, precomp_margin, coeff_idx);
        %ATM_criteria  = abs(dcL);
        [~,dcL,d2cL] = negative_log_likelihood_t(coeff_margin, ref, precomp_margin, coeff_idx);
        ATM_criteria = abs(dcL).^2 ./ diag(d2cL).';
        
        % select optimal element
        [~, opt_margin_idx] = max(ATM_criteria);
        opt_m_idx = margin(opt_margin_idx,:);
        fprintf('Selected index: [%d,%d]\n', opt_m_idx(:,1), opt_m_idx(:,2));
        
        % add index to binary tree
        T = T.add_node(opt_m_idx);
        
        % update precomp
        precomp_train = precomp_train.update( opt_m_idx );
        precomp_test = precomp_test.update( opt_m_idx );

        % optimize coefficients
        coeff = optimize_map(ref, precomp_train, [coeff; 0]);
        
        % evaluate error
        train_err = [train_err, negative_log_likelihood_t(coeff, ref, precomp_train)];
        test_err = [test_err, negative_log_likelihood_t(coeff, ref, precomp_test)];
        
    end
    %disp(precomp_train.midxs)
end

function coeff_opt = optimize_map(ref, precomp, coeff0)
    % define regularized objective
    obj = @(a) negative_log_likelihood_t(a, ref, precomp);
    % set options and run optimization
    options = optimoptions('fminunc','SpecifyObjectiveGradient', true, 'Display', 'off');
    [coeff_opt, ~, exit_flag] = fminunc(obj, coeff0, options);
    if exit_flag <= -1
        error('Mistake in the optimization')
    end
end

function [L,dcL,d2cL] = negative_log_likelihood_t(coeff, ref, precomp, coeff_idx)
    if nargin < 4
        coeff_idx = 1:length(coeff);
    end

    % evaluate map
    if nargout == 1
        [Sx, dxdS] = evaluate_map(coeff, precomp);
    elseif nargout == 2
        [Sx, dxdS, dcS, dcdxdS] = evaluate_map(coeff, precomp, coeff_idx);
    else
        [Sx, dxdS, dcS, dcdxdS, d2cS, d2cdxdS] = evaluate_map(coeff, precomp, coeff_idx);
    end
    
    % add small regularization term to map
    delta = 1e-9;
    Sx = Sx + delta*precomp.X;
    dxdS = dxdS + delta;
    
    % evaluate log_pi(x)
    L = ref.log_pdf(Sx) + log(dxdS);
    L = -1 * mean(L,1);
    
    % evaluate gradient \nabla_c log_pi(x)
    if nargout > 1
        dcL = ref.grad_x_log_pdf(Sx) .* dcS + dcdxdS ./ dxdS;
        dcL = -1 * mean(dcL,1);
    end

    % evaluate Hessian \nabla^2_c log_pi(x)
    if nargout > 2
        d2cL = ref.hess_x_log_pdf(Sx) .* OuterProd(dcS, dcS) -  OuterProd(dcdxdS,dcdxdS) ./ dxdS.^2 + ...
            ref.grad_x_log_pdf(Sx) .* d2cS + d2cdxdS ./ dxdS;
        d2cL = -1 * squeeze(mean(d2cL,1));
    end

end

function [S, dxS, dcS, dcdxdS, d2cS, d2cdxdS] = evaluate_map(coeff, precomp, coeff_idx)
    if nargin < 3
        coeff_idx = 1:length(coeff);
    end
    if size(coeff,1) == 1
        coeff=coeff.';
    end

    % evaluate f0
    Psi0 = precomp.Psi0;
    f0 = Psi0 * coeff;
    
    % define function to retrieve quadrature points
    quad_pts = @(N) precomp.evaluate_quadrature(N);
    % define function to be integrated
    dxSxi = @(dxPsi) softplus( dxPsi * coeff );
    % evaluate \int_0^x_d g(\partial_x_d f) dt
    [Int_dxS, precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi] = ...
        adaptive_integral(dxSxi, quad_pts, ...
        precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi, ...
        precomp.tol, precomp.pts_per_level, precomp.max_levels);
    
    % evaluate S
    S = f0 + Int_dxS;
    
    % evaluate dxS
    dxPsi = precomp.grad_x_Psi;
    dxf = dxPsi * coeff;
    dxS = softplus(dxf);
    
    if nargout > 2
        
        % evaluate gradients of f0 with respect to coeffs
        dcf0 = Psi0(:,coeff_idx);
        
        % define function to retrieve quadrature points
        quad_pts = @(N) precomp.evaluate_quadrature(N);
        % define function to be integrated
        dcdxSxi = @(dxPsi) dxsoftplus(dxPsi * coeff) .* dxPsi(:,coeff_idx);
        % evaluate \int_0^x_d g(\partial_x_d f) dt
        [dcInt_dxS, precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi] = ...
            adaptive_integral(dcdxSxi, quad_pts, ...
            precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi, ...
            precomp.tol, precomp.pts_per_level, precomp.max_levels);
        
        % evaluate dcS
        dcS = dcf0 + dcInt_dxS;

        % evaluate gradient of dxS with respect to coeffs
        dcdxdS = dxsoftplus(dxf) .* dxPsi(:,coeff_idx);
        
    end
    
    if nargout > 4
                
        % define function to be integrated
        d2cdxSxi = @(dxPsi) d2xsoftplus(dxPsi * coeff) .* OuterProd(dxPsi(:,coeff_idx), dxPsi(:,coeff_idx));
        % evaluate \int_0^x_d g(\partial_x_d f) dt
        [d2cInt_dxS, precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi] = ...
            adaptive_integral(d2cdxSxi, quad_pts, ...
            precomp.quad_dxPsii, precomp.quad_xi, precomp.quad_wi, ...
            precomp.tol, precomp.pts_per_level, precomp.max_levels);
        
        % evaluate dcS
        d2cS = d2cInt_dxS;

        % evaluate Hessian of dxS with respect to coeffs
        d2cdxdS = d2xsoftplus(dxf) .* OuterProd(dxPsi(:,coeff_idx), dxPsi(:,coeff_idx));
        
    end

end

% -- END OF FILE --
