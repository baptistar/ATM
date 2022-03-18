clear; close all; clc
addpath(genpath('../../../src'))

sd = 2; rng(sd);

X=load('X_2d_7.txt');
W=load('W_2d_7.txt');

XW.X=X;
XW.W=W;

%XW=randn(20000,2);
%List_obs=load('obs_wihtout_noise_neat.txt')%+0*randn(50,1);
%List_obs=load('obs_wihtout_noise_neat.txt')+7*randn(50,1);
%List_obs=load('List_obs_neat.txt');
List_obs=load('obs_wihtout_noise_2.txt')+7*randn(50,1);

x=linspace(-5,5,100)';

d=2;
ref2 = IndependentProductDistribution(repmat({Normal()},1,d+1));

yobs=List_obs(1);
%CM=load('./offline_maps/offline_map_poly_linear_neat2_1.mat');
CM=load('./offline_maps/offline_map_poly_linear2_1.mat');

CM=CM.x;
L_M={CM.S{1}.S CM.S{2}.S};


CM=ComposedPullbackDensity(L_M,ref2);

lkl = LikelihoodFunction(CM, yobs);

mu_0=[0.5 3];
sigma_0=[0.64 0; 0 0.25];

prior=MultivariateGaussian(d,mu_0,sigma_0);
post=BayesianDistribution(d,prior,lkl);

x0=mu_0;
[mu,Sigma] = optimize_laplace(post,x0);

Ls=chol(Sigma);

G_lap=GaussianPullbackDensity(d,false);

for k=1:d
    G_lap.S.S{k}.c=mu(k);
    G_lap.S.S{k}.L=Ls(k,1:k);
end

pi=PullbackDensity(G_lap.S,post);


% define reference and samples
N = 2000;
Zvalid = randn(N,d);
ref = IndependentProductDistribution(repmat({Normal()},1,d));
max_terms=[30];
max_patience=[];
var_tol=-3;

% define total-order identity map using Hermite functions
%basis = ProbabilistHermiteFunction();
%basis=HermiteProbabilistPoly();
%basis=ConstExtProbabilistHermiteFunction();
basis=HermiteProbabilistPolyWithLinearization();
T = identity_map(1:d, basis);
T = TriangularTransportMap(T);

% define and optimize pullback-density
[Topt, output] = adaptive_transport_map(T, pi, XW, Zvalid, max_terms,max_patience,var_tol);


CM_post=ComposedPullbackDensity({Topt, G_lap.S}, ref);

L_mu_post=[];
L_std_post=[];

Z=CM_post.evaluate(randn(50000,2));
mu_post=mean(Z);
std_post=std(Z);

L_mu_post=[L_mu_post, mu_post];
L_std_post=[L_std_post, std_post];
L_best_var=[output.best_var_d];

L_idxs=[length(Topt.coeff)];

r=1;

N=50;
for k=2:N
    disp(['------ Iteration : ',num2str(k),'-----------'])
    %PB_off=load(['./offline_maps/offline_map_poly_linear_neat2_',num2str(k),'.mat']);
    PB_off=load(['./offline_maps/offline_map_poly_linear2_',num2str(k),'.mat']);

    PB_off=PB_off.x;
    L_M={PB_off.S{1}.S PB_off.S{2}.S};
    PB_off=ComposedPullbackDensity(L_M,ref2);
    
    yobs=List_obs(k);
    
    lkl_seq=LikelihoodFunction_seq(PB_off,CM_post, yobs);
    
    post_seq=BayesianDistribution(d,ref,lkl_seq);
    
    max_terms=10;

    % define total-order identity map using Hermite functions
    %basis = ProbabilistHermiteFunction();
    T = identity_map(1:d, basis);
    T = TriangularTransportMap(T);
    
    [Topt_seq, output] = adaptive_transport_map(T, post_seq, XW, Zvalid, max_terms,max_patience,var_tol);
    
    L_S=[{Topt_seq} CM_post.S];
    
    CM_post=ComposedPullbackDensity(L_S, ref);
    Z=CM_post.evaluate(randn(50000,d));
    
    mu_post=mean(Z);
    std_post=std(Z);
    
    L_mu_post=[L_mu_post, mu_post];
    L_std_post=[L_std_post, std_post];
    L_idxs=[L_idxs,length(Topt_seq.coeff)];
    
    r=r+1;
    
    if r==2
        L_S2=L_S;
        L_S2(end)=[];
        CM_2=ComposedPullbackDensity(L_S2, ref);
        
        disp(['------ Regression :',num2str(k),' -----------'])
        
        basis=HermiteProbabilistPoly();
        Sreg = identity_map(1:d, basis);
        PB = PullbackDensity(Sreg, ref);
        max_terms=[25,40];
        tol=1e-5;
        PB=PB.greedy_optimize_regression(CM_2,XW, ...
                                    max_terms, tol);
        
        L_Sreg=[{PB.S},L_S(end)];
        
        CM_post=ComposedPullbackDensity(L_Sreg, ref);
        r=0;
    end
    L_best_var=[L_best_var,output.best_var_d];
end


Xtest=linspace(-4,4,100);
X=[Xtest(:) zeros(100,1)];


fX=lkl_seq.log(X);

DY=gradient(fX(:,1),Xtest);
DY2=lkl_seq.grad_x_log(X);

figure
hold on
plot(Xtest,DY)
plot(Xtest,DY2(:,1),'o')

X=[zeros(100,1) Xtest(:)];

fX=lkl_seq.log(X);

DY=gradient(fX,Xtest);
DY2=lkl_seq.grad_x_log(X);

figure
hold on
plot(Xtest,DY)
plot(Xtest,DY2(:,2),'o')


%%%%%
mu_post1=L_mu_post(1:2:end);
std_post1=L_std_post(1:2:end);

mu_post1=[0.5,mu_post1];
std_post1=[0.8,std_post1];

mu_post2=L_mu_post(2:2:end);
std_post2=L_std_post(2:2:end);

mu_post2=[3,mu_post2];
std_post2=[0.5,std_post2];

figure
plot([0 N],[1 1],'r','linewidth',1.5)
hold on
plot(0:N,mu_post1,'k','linewidth',1.5)
plot(0:N,mu_post1-3*std_post1,'--k','linewidth',1.5)
plot(0:N,mu_post1+3*std_post1,'--k','linewidth',1.5)
xlabel('Time steps')
ylabel('c_0')

figure
plot([0 N],[2.5 2.5],'r','linewidth',1.5)
hold on
plot(0:N,mu_post2,'k','linewidth',1.5)
plot(0:N,mu_post2-3*std_post2,'--k','linewidth',1.5)
plot(0:N,mu_post2+3*std_post2,'--k','linewidth',1.5)
xlabel('Time steps')
ylabel('c_1')

save('L_best_var_noise_noise.mat','L_best_var')
save('L_terms_noise_noise.mat','L_idxs')
%save('CM_post_noise_10.mat','CM_post')