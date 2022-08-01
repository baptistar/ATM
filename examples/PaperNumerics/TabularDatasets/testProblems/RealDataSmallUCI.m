classdef RealDataSmallUCI

	% Abstract class implements splitting and 
	% normalization procedure for ML datasets
	% after loading data from file and applying
	% specific post-processing.
	% 
	% Author: Ricardo Baptista
	% Date:   September 2019

	properties
            d 		% dimensions of data
            data    % (N x d) array of data 
            folder  % string for folder containing data
            name    % string for name of problem
	end

	methods
		function RD = RealDataSmallUCI(varargin)
			
			% declare F object
			p = ImprovedInputParser;
			parse(p,varargin{:});
			RD = passMatchedArgsToProperties(p, RD);

            % set folder with data
            RD.folder = './testProblems/';

			% load and process data
			data = RD.load_data();
            RD.data = RD.process_data(data);
            
            % save data dimensions
			RD.d = size(RD.data,2);

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function data = load_data(~)
			error('RD: Implemented in child class')
		end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function counts = get_correlation_numbers(~, data)

        	% compute correlations and find large values
        	C = corr(data);
        	Clarge = C > 0.98;
        	
			% compute count of large values in columns
        	counts = sum(Clarge, 1);

		end %endFunction
        %------------------------------------------------------------------
        %------------------------------------------------------------------
        function data = process_data(RD, data)

        	% compute correlations of columns
			counts = RD.get_correlation_numbers(data);

            % run until all correlations are smaller than one
            while any(counts > 1)

                % extract first column where count > 1
                col_to_remove = find(counts > 1, 1, 'first');

                % remove column and recompute correlations
                data(:,col_to_remove) = [];
                counts = RD.get_correlation_numbers(data);

            end
            
        end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function [Xtrain, Xtest] = split_data(~, data)

			% determine size of dataset
			N = size(data,1);

			% separate last 10% of data for testing
			N_test = floor(0.1*N);
			Xtest  = data(N-N_test+1:end,:);
			Xtrain = data(1:N-N_test,:);
            %Xtest  = data(1:N_test,:);
			%Xtrain = data(N_test+1:end,:);
                
		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function data = normalize_data(~, data)%[Xtrain, Xtest] = normalize_data(~, Xtrain, Xtest)

			% compute mean and standard deviation 
			%Xmean = mean(Xtrain, 1);
			%Xstd  = std(Xtrain, [], 1);

			% normalize all data
			%Xtrain = (Xtrain - Xmean)./Xstd;
			%Xtest  = (Xtest - Xmean)./Xstd;

            data_mean = mean(data, 1);
			data_std  = std(data, [], 1);

			% normalize all data
			data = (data - data_mean)./data_std;

 		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass
