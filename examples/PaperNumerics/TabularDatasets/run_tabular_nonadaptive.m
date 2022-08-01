%clear; close all; clc
addpath(genpath('../../../src'))
addpath(genpath('testProblems'))
sd = 1; rng(sd);

% define folder to save data
if ~exist('./data', 'dir')
  mkdir('data');
end

%% --- Define test case ---

% define test cases
problems = {'parkinsons', 'housing', 'redwine', 'whitewine'};
n_folds = 10;
orders = [1,2];

for i=1:length(problems)
    for k=1:n_folds
        % load training and test samples
        file_name = ['data/' problems{i} '_cv_fold' num2str(k) '.mat']; 
        load(file_name, 'points');
        % learn map and save results
        for order = orders
            test_case = ['data/' problems{i} '_fold' num2str(k) '_nonadapt_order' num2str(order) '.mat']; 
            try
                [CM, output] = buildComposedMap(points.training, order);
                save(test_case, 'CM');
                disp([test_case ' is complete!']);
            catch
                disp(['Error: ' test_case ' did not complete!']);
            end
        end
    end
    
end

%% --- Helper Functions ---

function [CM, output] = buildComposedMap(X, order)

    % define reference
    d = size(X,2);
    ref = IndependentProductDistribution(repmat({Normal()},1,d));

    % define Gaussian map
    G = GaussianPullbackDensity(d, true);
    G = G.optimize(X);
    Z = G.S.evaluate(X);

    % define bounds
    basis = cell(d,1);
    for k=1:d
        basis{k} = HermiteProbabilistPolyWithLinearization();
        basis{k}.bounds = quantile(Z(:,k),[0.01,0.99]).';
    end
    
    % define TM
    TM = total_order_map(1:d, basis, order);
    PB = PullbackDensity(TM, ref);
    [PB, output] = PB.optimize(Z);

    % compose map
    CM = ComposedPullbackDensity({G, PB}, ref);

end

% -- END OF FILE --
