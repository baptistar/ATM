classdef Housing < RealDataSmallUCI

	% Load and process Housing dataset: 
	% 10-dimensional attributes and 506 samples
	%
	% Author: Ricardo Baptista
	% Date:   January 2020

	properties
	end

	methods
		function H = Housing(varargin)

			% declare G object from RealDataSmallUCI
			H@RealDataSmallUCI(varargin{:});
			H.name = 'housing';

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
		function data = load_data(H)

			% load data from csv file
			file_name = [H.folder, 'housing.txt'];
			table = readtable(file_name);
			data  = table2array(table);

			% remove 2nd, 4th, 9th, and 10th columns
			data(:,[2,4,9,10]) = [];

		end %endFunction
		%------------------------------------------------------------------
		%------------------------------------------------------------------
	end %endMethods

end %endClass
