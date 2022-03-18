function fig = niceSpy(A)
%
%Source: stack overflow
%

m = size(A,1);
n = size(A,2);
nonzeroInd = find(A);
[x, y] = ind2sub([m n], nonzeroInd);
%fig = figure();
hp = patch(y, x, A(nonzeroInd), ...
           'Marker', 's', 'MarkerFaceColor', 'flat', 'MarkerSize', 8, ...
           'EdgeColor', 'none', 'FaceColor', 'none');
set(gca, 'XLim', [0, n + 1], 'YLim', [0, m + 1], 'YDir', 'reverse', ...
    'PlotBoxAspectRatio', [n + 1, m + 1, 1]);

set(gca, 'XTick', 1:m); set(gca,'XTickLabel',1:m, 'FontSize',10);
set(gca, 'YTick', 1:n); set(gca,'YTickLabel',1:n, 'FontSize',10);

colormap(cool)
colorbar();

end
