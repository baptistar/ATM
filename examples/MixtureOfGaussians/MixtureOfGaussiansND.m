classdef MixtureOfGaussiansND

   properties
       dim      % dimension of distribution
       n_modes  % number of modes in each dimension
       loc      % vector of location of each mode
       w        % vector of weight for each mode
       Sigma    % vector of covariance for each mode
   end

   methods
       function MoG = MixtureOfGaussiansND(dim)
           
            % set parameters in MoG
            MoG.dim = dim;
            MoG.n_modes = 2^dim;

            % define weight, locations and covariance
            elements = repmat({[-4,4]},1,MoG.dim);   %cell array with N vectors to combine
            combinations = cell(1, numel(elements)); %set up the varargout result
            [combinations{:}] = ndgrid(elements{:});
            combinations = cellfun(@(x) x(:), combinations,'uniformoutput',false); 
            MoG.loc = [combinations{:}]; % NumberOfCombinations by N matrix. Each row is unique.

            MoG.w = 0.9*rand(MoG.n_modes,1) + 0.1;
            MoG.w = MoG.w / sum(MoG.w);
            MoG.Sigma = eye(MoG.dim);

       end
       % -----------------------------------------------------
       % -----------------------------------------------------
       function X = sample(MoG, N)
            X = ones(0,MoG.dim);
            for i=1:MoG.n_modes-1
                X = [X; mvnrnd(MoG.loc(i,:), MoG.Sigma, floor(MoG.w(i)*N))];
            end
            X = [X; mvnrnd(MoG.loc(end,:), MoG.Sigma, N - size(X,1))];
            X = X(randperm(N),:);
       end
       % -----------------------------------------------------
       % -----------------------------------------------------
       function log_pi = log_pdf(MoG, X)
	    pi = zeros(size(X,1),1);
            for i=1:MoG.n_modes
               pi = pi + MoG.w(i)*mvnpdf(X, MoG.loc(i,:), MoG.Sigma);
            end
            log_pi = log(pi);
       end 
       % -----------------------------------------------------
       % -----------------------------------------------------
   end %endMethods

end %endClass
