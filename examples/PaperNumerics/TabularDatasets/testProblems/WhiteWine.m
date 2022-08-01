classdef WhiteWine < RealDataSmallUCI

	% Load and process WhiteWine dataset: 
	% 11-dimensional attributes and 4898 samples
	%
	% Author: Ricardo Baptista
	% Date:   January 2020

	properties
	end

	methods
		function W = WhiteWine(varargin)

			% declare G object from RealDataSmallUCI
			W@RealDataSmallUCI(varargin{:});
			W.name = 'whitewine';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function data = load_data(W)

			% load data from csv file
			file_name = [W.folder, 'winequality-white.csv'];
			table = readtable(file_name,'HeaderLines',1);
			data  = table2array(table);

			% remove last columns (quality)
			data(:,end) = [];

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass