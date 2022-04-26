%clear; close all; clc
sd = 1; rng(sd);

%% --- Define test case ---

% set orders
orders = 1:5;

% define grid of tests
[testD, testN, testP] = meshgrid(d_vect, N_vect, MCruns);

for i = 1 : numel(testP)

    % extract d, P and N
    d  = testD(i);
    N  = testN(i);
    j  = testP(i);
    
    % load training and test samples
    file_name = ['samples/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(j)];
    load(file_name, 'Xtrain');

    % learn map and save results
    for order = orders
        test_case = ['data/MoG_nonadapt_d' num2str(d) '_N' num2str(N) '_run' num2str(j) '_order' num2str(order) '.mat'];
        try
            [CM, output] = buildComposedMap(Xtrain, order);
            save(test_case, 'CM');
            disp([test_case ' is complete!']);
        catch
            disp(['Error: ' test_case ' did not complete!']);
        end
    end
    
end

%% -- Helper function --

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
