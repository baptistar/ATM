clear; close all; clc
addpath(genpath('../../'))
sd = 1; rng(sd);

% set parameters
d_x = 2; d_y = 2;
d = d_x + d_y;
comps = d_y + (1:d_x);

% generate data
N = 1e4;
X = randn(N,d_x);
A = randn(d_y,d_x);
B = randn(d_y,d_y);
Y = (A*X.').' + (B*randn(N,d_y).').';
YX = [Y,X];

% split data
N_train = 1e3;
YX_train = YX(1:N_train,:);
YX_test  = YX(N_train+1:end,:);

% compute mean and covariance
mu_train = mean(YX_train,1);
C_train = cov(YX_train);
L_train = chol(C_train,'lower');
Linv_train = inv(L_train);

Ldiag = diag(std(YX_train,[],1));
Linv_diag = diag(1./diag(Ldiag));

% define tolerance
tol = 1e-10;

%% diagonal = true

% optimize map
LM = GaussianPullbackDensity(d_x+d_y, true);
LM = LM.optimize(YX_train);

% check L and c
assert(norm(LM.c - mu_train.') < tol);
assert(norm(LM.L - Linv_diag) < tol);

% define comp_idx
comp_idx = 3:4;

% check evaluate
Sx = LM.evaluate(YX_test, comp_idx);
Sx_true = (Linv_diag(comp_idx,:) * (YX_test - mu_train).').';

assert(norm(Sx(:) - Sx_true(:)) < tol);

% check inverse
ncomp_idx = 1:2;
Xp = YX_test(:,ncomp_idx);
Z  = randn(size(Xp,1),2);
Xd = LM.inverse(Z, Xp, comp_idx);
Xd_true = (Ldiag(comp_idx,comp_idx) * Z.').' + mu_train(comp_idx);

assert(norm(Xd(:) - Xd_true(:)) < tol);

% check log_pdf
log_pdf = LM.log_pdf(YX_test, comp_idx);
C_true = diag(1./diag(Linv_diag(comp_idx,comp_idx).*Linv_diag(comp_idx,comp_idx)));
m_true = mu_train(comp_idx);
log_pdf_true = log(mvnpdf(YX_test(:,comp_idx), m_true, C_true));

assert(norm(log_pdf(:) - log_pdf_true(:)) < tol);

% check conditional samples
N_samples = 100;
Z = randn(N_samples,d_x);
y_vect = mean(YX(:,1:d_y));

Y_vect = repmat(y_vect, N_samples, 1);
post_samples = LM.inverse(Z, Y_vect, comp_idx);
post_samples_true = (Ldiag(comp_idx,comp_idx)*Z.').' + mu_train(comp_idx);

assert(norm(post_samples_true - post_samples,'fro') < 1e-10)

%% diagonal = false

LM = GaussianPullbackDensity(d_x+d_y, false);
LM = LM.optimize(YX_train);

% check L and c
assert(norm(LM.c - mu_train.') < tol);
assert(norm(LM.L - Linv_train) < tol);

% check evaluate
Sx = LM.evaluate(YX_test, comp_idx);
Sx_true = (Linv_train(comp_idx,:) * (YX_test - mu_train).').';

assert(norm(Sx(:) - Sx_true(:)) < tol);

% check inverse
ncomp_idx = 1:2;
Xp = YX_test(:,ncomp_idx);
Z  = randn(size(Xp,1),2);
Xd = LM.inverse(Z, Xp, comp_idx);

Z_id = LM.evaluate([Xp,Xd], comp_idx); 

assert(norm(Z(:) - Z_id(:)) < tol);

% check log_pdf
log_pdf = LM.log_pdf(YX_test, comp_idx);
ncomp_idx = setdiff(1:d, comp_idx);
C_true = C_train(comp_idx,comp_idx) - C_train(comp_idx,ncomp_idx)*(C_train(ncomp_idx,ncomp_idx)\C_train(ncomp_idx,comp_idx));
m_true = mu_train(comp_idx) + (C_train(comp_idx,ncomp_idx)*(C_train(ncomp_idx,ncomp_idx)\(YX_test(:,ncomp_idx) - mu_train(ncomp_idx)).')).';
log_pdf_true = log(mvnpdf(YX_test(:,comp_idx), m_true, C_true));

assert(norm(log_pdf(:) - log_pdf_true(:)) < tol);

% check conditional samples
N_samples = 100;
Z = randn(N_samples,d_x);
y_vect = mean(YX(:,1:d_y));

Z_cond = Z - (Linv_train(comp_idx,ncomp_idx)*(y_vect(ncomp_idx) - mu_train(ncomp_idx)).').';
post_samples_true = (L_train(comp_idx,comp_idx)*Z_cond.').' + mu_train(comp_idx);

Y_vect = repmat(y_vect, N_samples, 1);
post_samples = LM.inverse(Z, Y_vect, comp_idx);

assert(norm(post_samples_true - post_samples,'fro') < 1e-10)

% -- END OF FILE --