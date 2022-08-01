classdef MoG

   properties
       d        % dimension of distribution
       n_modes  % number of modes in each dimension
       loc      % vector of location of each mode
       w        % vector of weight for each mode
       sigma    % vector of variances for each mode
       name
   end

   methods
       function MoG = MoG(d)
           
            % set parameters in MoG
            MoG.d = d;
            MoG.n_modes = 2;

            % define weight, locations and covariance
            MoG.loc = zeros(MoG.n_modes, d); MoG.loc(:,1) = [-3,3];
            MoG.w = 1/MoG.n_modes*ones(MoG.n_modes,1);
            MoG.sigma = 0.2*ones(MoG.n_modes,1);
            
            % define name
            MoG.name = 'mog';
            
       end
       % -----------------------------------------------------
       % -----------------------------------------------------
       function X = sample(MoG, N)
            X = ones(0,MoG.d);
            for i=1:MoG.n_modes-1
                C_i = MoG.sigma(i)*eye(MoG.d);
                X_i = mvnrnd(MoG.loc(i,:), C_i, floor(MoG.w(i)*N));
                X = [X; X_i];
            end
            C_end =  MoG.sigma(end)*eye(MoG.d);
            X = [X; mvnrnd(MoG.loc(end,:), C_end, N - size(X,1))];
       end
       % -----------------------------------------------------
       % -----------------------------------------------------
       function log_pi = log_pdf(MoG, X)
	    pi = zeros(size(X,1),1);
            for i=1:MoG.n_modes
               C_i = MoG.sigma(i)*eye(MoG.d);
               pi_i = MoG.w(i)*mvnpdf(X, MoG.loc(i,:), C_i);
               pi = pi + pi_i;
            end
        log_pi = log(pi);
       end 
       % -----------------------------------------------------
       % -----------------------------------------------------
   end %endMethods

end %endClass
