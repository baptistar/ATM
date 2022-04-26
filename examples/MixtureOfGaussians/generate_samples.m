% clear; close all; clc
sd = 10; rng(sd);

%% --- Define test case ---

% define number of test samples
Ntest = 1e5;

for i=1:length(d_vect)

    % define problems
    d = d_vect(i);
    P = MixtureOfGaussiansND(d);

    % generate all test samples
    Xtest = P.sample(Ntest);

    % save model and test samples
    save(['samples/MoG_d' num2str(d)], 'P', 'Xtest');

    % generate training samples
    for j=1:length(N_vect)
        for k=MCruns

            N = N_vect(j);
            fprintf('Samples: d = %d, N = %d, run = %d\n', d, N, k)

            % generating training samples
            Xtrain = P.sample(N);
            
            % save training samples
            save(['samples/MoG_d' num2str(d) '_N' num2str(N) '_run' num2str(k)], 'Xtrain');

        end
    end

end

% -- END OF FILE --
