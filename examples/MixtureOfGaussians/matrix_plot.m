function [fig,axes] = matrix_plot(data, data2, labels, plot_params)
% MATRIX_PLOT: Function generates a matrix plot of the parameter 2D
% contours for each dimension based on Kernel Density Estimation
%
% Author:
% Date:   November 21, 2015
    
% Determine the number of dimensions and datapoints
N_dims = size(data,2);

% define default parameters
if (nargin < 4)
    plot_params = struct;
    plot_params.alw = 1.2;    % AxesLineWidth
    plot_params.fsz = 14;     % Fontsize
    plot_params.lw  = 2;      % LineWidth
    plot_params.msz = 4;      % MarkerSize
    plot_params.flabels = 24; % Labels font size
    plot_params.kde = true;
end

% define default labels
if (nargin < 3)
    labels = cell(N_dims,1);
    for i=1:N_dims 
        labels{i} = ['$x_{' num2str(i) '}$'];
    end
end

% check if data2 is specified
if (nargin < 2) || isempty(data2)
    data2 = data;
end

% reshape data arrays
data = data.';
data2 = data2.';

% Tolerance for variable bound
tol = 0;

% Resolution for plotting
N_points = 40;

%% Determine Bounds for Variables

% Declare matrix to store bounds
bounds = zeros(2,N_dims);

% Find all bounds
for i=1:N_dims
    
    % Find bounds
    temp_min = min(data(i,:));
    temp_max = max(data(i,:));
    offSet = tol*(temp_max - temp_min)/2;
    iymin  = temp_min - offSet;
    iymax  = temp_max + offSet;

    % Save in bounds
    bounds(1,i) = iymin;
    bounds(2,i) = iymax;

end

%% Generate all subplots

% Mesh for plotting
[x0, y0] = meshgrid(linspace(0,1,N_points),linspace(0,1,N_points));
x1 = reshape(x0,N_points*N_points,1);
y1 = reshape(y0,N_points*N_points,1);

% Right Subplot axes
Nh = N_dims;
Nw = N_dims;
gap    = [.01 .01];
marg_h = [.1 .1];%marg_h = [.1 .1];
marg_w = [.1 .1];%marg_w = [.1 .1];
axh = (1-sum(marg_h)-(Nh-1)*gap(1))/Nh; 
axw = (1-sum(marg_w)-(Nw-1)*gap(2))/Nw;
py = 1-marg_h(2)-axh; 

% declare cell to store axes limits
axes = cell(N_dims, N_dims);

fig = figure;

for i=1:N_dims
    
    px = marg_w(1);
    for j=1:i

        subplot('Position',[px py axw axh]);
        px = px+axw+gap(2);
        
        hold on
        
        % If j==i (then plot the density)
        if j==i
            
            [fi,xi] = ksdensity(data(i,:));
            plot(xi,fi)
            xlim([bounds(1,i) bounds(2,i)])
            %title(labels{i},'interpreter','latex')

            % plot samples
            plot(data(i,:), zeros(size(data,2),1), '.r', 'MarkerSize', plot_params.msz);

        else
            
            % Evaluate bounds
            x = x1.*(bounds(2,i)-bounds(1,i)) + bounds(1,i);
            y = y1.*(bounds(2,j)-bounds(1,j)) + bounds(1,j);
            
            % plot samples
            plot(data(j,:), data(i,:), '.', 'MarkerSize', plot_params.msz)
            plot(data2(j,:), data2(i,:), '.r', 'MarkerSize', plot_params.msz)
            
            % plot KDE
            if plot_params.kde == true
                % Evaluate KDE
                temp = kde(data([i j],:),'rot');
                % Generate contour
                contour(reshape(y, N_points, N_points), ...
                        reshape(x, N_points, N_points), ...
                        reshape(evaluate(temp, [x y]'), N_points, N_points))
            end
            
            xlim([bounds(1,j) bounds(2,j)])
            ylim([bounds(1,i) bounds(2,i)])
            colormap winter
            
        end

        % Plot details
        set(gca,'TickLabelInterpreter','latex','FontSize', plot_params.fsz)
        set(gca,'LineWidth', plot_params.alw);
        
        % Remove ylabels if plot is on the interior of matrix
        if(j~=1)
            set(gca,'yticklabel',[])
        else
            set(gca,'TickLabelInterpreter','latex','FontWeight','normal')
        end
                
        % Remove xlabels if plot is on the interior of matrix
        if(i~=N_dims)
            set(gca,'xticklabel',[])
        else
            set(gca,'TickLabelInterpreter','latex','FontWeight','normal')
        end
        
        hold off
        
        % save axes labels
        axes{i,j} = {get(gca,'XLim'), get(gca,'YLim')};

    end
    py = py-axh-gap(1);
    
end

end
