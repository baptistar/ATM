addpath('problems')
addpath(genpath('../../../src'))
sd = 1; rng(sd);

%% --- Define test case ---

% define problems
d = 2;
problems = {@Banana, @MoG, @Funnel, @Cosine, @Ring};

% define sample size and run
N_vect = 10000;
MCruns = 1;

for l=1:length(problems) 

    % define problems and sample sizes
    P = problems{l}(d);
    
    for j=1:length(N_vect)
        for run=MCruns

            N = N_vect(j);
            fprintf('========================================\n')
            fprintf('Problem: %s, N = %d, run = %d\n', P.name, N, run);
    
            % load data for map
            file_name = ['data/' P.name '_N' num2str(N) '_run' num2str(run) '.mat'];
            load(file_name, 'CM')

            % find number of terms
            tot_terms = 0;
            for k=1:d 
                CM_midx = CM.S{2}.S{k}.f.multi_idxs;
                max_deg = max(CM_midx,[],1);
                n_terms = size(CM_midx,1);
                tot_terms = tot_terms + n_terms;
                if k==1
                fprintf('Comp %d, max_deg: [%d], %d terms\n', k, max_deg(1), n_terms);
                elseif k==2
                fprintf('Comp %d, max_deg: [%d, %d], %d terms\n', k, max_deg(1), max_deg(2), n_terms);
                end
            end
            fprintf('Total terms: %d\n', tot_terms)

        end
    end
    
end 

% -- END OF FILE --
