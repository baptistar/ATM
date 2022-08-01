classdef RedWine < RealDataSmallUCI

	% Load and process RedWine dataset: 
	% 11-dimensional attributes and 1599 samples
	%
	% Author: Ricardo Baptista
	% Date:   January 2020

	properties
	end

	methods
		function R = RedWine(varargin)

			% declare G object from RealDataSmallUCI
			R@RealDataSmallUCI(varargin{:});
			R.name = 'redwine';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function data = load_data(R)

			% load data from csv file
			file_name = [R.folder, 'winequality-red.csv'];
			table = readtable(file_name,'HeaderLines',1);
			data  = table2array(table);

			% remove last columns (quality)
			data(:,end) = [];

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass