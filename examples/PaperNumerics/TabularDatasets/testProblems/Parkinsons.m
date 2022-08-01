classdef Parkinsons < RealDataSmallUCI

	% Load and process Parkinsons dataset: 
	% 15-dimensional attributes and 1599 samples
	%
	% Author: Ricardo Baptista
	% Date:   January 2020

	properties
	end

	methods
		function P = Parkinsons(varargin)

			% declare G object from RealDataSmallUCI
			P@RealDataSmallUCI(varargin{:});
			P.name = 'parkinsons';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function data = load_data(R)

			% load data from csv file
			file_name = [R.folder, 'parkinsons_updrs.csv'];
			table = readtable(file_name,'HeaderLines',1);

			% remove first three columns (string) and convert to array
			table(:,1:3) = [];
			data  = table2array(table);

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass