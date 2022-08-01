%clear; close all; clc
addpath(genpath('../../../src'))
addpath('problems')
sd = 1; rng(sd);

%% --- Define test case ---

% define problem
d = 2;
problems = {@Banana, @MoG, @Funnel, @Cosine, @Ring};

% define sample size and run
N_vect = 10000;
MCruns = 1;

% define reference, basis, and order
max_terms = 200;

% define grid of tests
[testP, testN, testR] = meshgrid(problems, N_vect, MCruns);

% create data folder
if ~exist('./data', 'dir')
    mkdir('./data')
end

% set parallel processing
n_proc = 2;
c = parcluster('local');
c.NumWorkers = n_proc;
parpool(n_proc);

parfor i = 1 : numel(testP)
    
    % extract P, N, run
    P = testP{i}(d);
    N = testN(i);
    run = testR(i);

    % load random rotation Q
    Q = parLoadQ(['rotation_data/' P.name '_N10000']);
    
    % determine Ntrain and Nvalid
    Ntrain = floor(0.8*N);
    Nvalid = floor(0.2*N);
    
    % generate samples
    Xtrain = P.sample(Ntrain);
    Xtrain = Xtrain*Q;
    Xvalid = P.sample(Nvalid);
    Xvalid = Xvalid*Q;
    
    % learn map
    [CM, output] = buildComposedMap(Xtrain, Xvalid, max_terms);
    test_case = ['data/' P.name '_N' num2str(N) '_run' num2str(run) '.mat'];
    parSave(test_case, CM, output);
    
end

delete(gcp('nocreate'))
    
%% -- Define Helper Functions --

function Q = parLoadQ(file_name)
    load(file_name, 'Q');
end

function parSave(file_name, CM, output)
    save(file_name, 'CM', 'output');
end

function [CM, output] = buildComposedMap(Xtrain, Xvalid, max_terms)

    % define reference
    d = size(Xtrain,2);
    ref = IndependentProductDistribution(repmat({Normal()},1,d));

    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(Xtrain);
    Ztrain = G.S.evaluate(Xtrain);
    Zvalid = G.S.evaluate(Xvalid);

    % define basis
    basis = cell(d,1);
    for k=1:d
        basis{k} = HermiteProbabilistPolyWithLinearization();
        basis{k}.bounds = quantile(Ztrain(:,k),[0.01,0.99]).';
    end
    
    % define TM
    TM = identity_map(1:d, basis);
    PB = PullbackDensity(TM, ref);
    
    % optimize map
    output = cell(d,1);
    for k=1:d
        % set defaults
        max_patience = 20;
        % run greedy approximation on S_valid
        S_valid = PB.S.S{k};
        ZWtrain.X = Ztrain(:,1:k); ZWtrain.W = ones(size(Ztrain,1),1)/size(Ztrain,1);
        ZWvalid.X = Zvalid(:,1:k); ZWvalid.W = ones(size(Zvalid,1),1)/size(Zvalid,1);
        [S_valid, output{k}] = greedy_fit(S_valid, ref.factors{k}, ...
            max_terms, ZWtrain, ZWvalid, max_patience);
        % find optimal number of terms (adding terms originally in S)
        % remove one to account for initial condition
        [~, n_added_terms] = min(output{k}.valid_err);
        n_added_terms = n_added_terms(1) - 1;
        opt_terms = PB.S.S{k}.n_coeff + n_added_terms;
        fprintf('Final map: %d terms\n', opt_terms);
        % extract optimal multi-indices
        midx_opt = S_valid.multi_idxs();
        midx_opt = midx_opt(1:opt_terms,:);
        PB.S.S{k} = PB.S.S{k}.set_multi_idxs(midx_opt);
        % run greedy_fit up to opt_terms with all data
        a0 = zeros(opt_terms,1);
        PB.S.S{k} = optimize_component(PB.S.S{k}, ref.factors{k}, a0, ZWtrain, []);
    end
    
    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

end

% -- END OF FILE --
