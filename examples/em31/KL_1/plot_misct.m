clear all
close all

sigma_w=2500;
sigma_i=0.22;

mu_0=1.8;
sigma_0=0.3;

Rz_iw=@(z) 1./sqrt(4*z.^2+1);

Sigma_eff=@(z) sigma_i*(1-(1./sqrt(4*z.^2+1)))+sigma_w.*(1./sqrt(4*z.^2+1));


Z=linspace(0,2,100);

figure
plot(Z,Rz_iw(Z))

figure
plot(Z,Sigma_eff(Z))