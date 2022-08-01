classdef ConditionalDatasets

    properties
        folder   % folder containing raw data
        name     % name of specific dataset
    end
    
    methods
        % -----------------------------------------------------------------
        function CD = ConditionalDatasets(name)
            CD.folder = './testProblems/';
            CD.name = name;
        end
        % -----------------------------------------------------------------
        function [X,Y] = load_data(CD)
            % load from file
            if strcmp(CD.name, 'Energy')
                data = CD.read_excel('energy.xlsx');
            elseif strcmp(CD.name, 'Concrete')
                data = CD.read_excel('concrete.xlsx');
            elseif strcmp(CD.name, 'Housing')
                data = CD.read_csv('housing.data');
            elseif strcmp(CD.name, 'Yacht')
                data = CD.read_csv('yacht.data');
            end
            % remove rows with missing data
            row_isnotnan = all(~isnan(data),2);
            data = data(row_isnotnan,:);
            % get X (target columns) and Y (feature columns)
            X = data(:,end);
            Y = data(:,1:end-1);
        end
        % -----------------------------------------------------------------
        function data = read_excel(CD, file_name)
            file_name = [CD.folder, file_name];
            [data,~,~] = xlsread(file_name) ;
        end
        % -----------------------------------------------------------------
        function data = read_csv(CD, file_name)
            file_name = [CD.folder, file_name];
            data = importdata(file_name);
            if isstruct(data)
                data = data.data;
            end
        end
        % -----------------------------------------------------------------
    end
    
end
