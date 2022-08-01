function map_sparsity(Aapprox, Atrue)

% determine size of A
m = size(Aapprox,1);
n = size(Aapprox,2);

% check inputs
if nargin > 1
    assert(all(size(Atrue) == [m,n]))
end

% plot Aapprox
nonzeroInd = find(Aapprox);
[x, y] = ind2sub([m n], nonzeroInd);
hold on;
h(1) = patch(y, x, Aapprox(nonzeroInd), ...
       'Marker', 's', 'MarkerFaceColor', 'flat', 'MarkerSize', 8, ...
       'EdgeColor', 'none', 'FaceColor', 'none');
% plot Atrue
if nargin > 1
    nonzeroInd = find(Atrue);
    [x, y] = ind2sub([m n], nonzeroInd);
    for i=1:length(x)
        patch(y(i), x(i), Atrue(nonzeroInd(i,:)), ...
               'Marker', 's', 'MarkerFaceColor', 'none', 'MarkerSize', 10, ...
               'EdgeColor', 'k', 'FaceColor', 'none', 'LineWidth', 1.5);
    end
end
% set 
set(gca, 'XLim', [0, n + 1], 'YLim', [0, m + 1], 'YDir', 'reverse', ...
    'PlotBoxAspectRatio', [n + 1, m + 1, 1]);
set(gca,'TickDir','out');
set(gca, 'XTick', 1:m); 
set(gca,'XTickLabel',1:m, 'FontSize',10);
set(gca, 'YTick', 1:n); 
set(gca,'YTickLabel',1:n, 'FontSize',10);
set(gca, 'LineWidth', 1.5);
colormap(flipud(bone))
caxis([0,max([max(max(Aapprox)), max(max(Atrue))])]);
c = colorbar(); c.FontSize = 18;
set(c,'TickLabelInterpreter','latex')
set(gca,'box','off')
%ax=gca;
%axes('position',ax.Position,'box','on','ytick',[],'xtick',[],'color','none')

hold off

[~,hl] = legend({'ATM map','$S_{KR}$ sparsity'},'FontSize',18);
PatchInLegend = findobj(hl, 'type', 'patch');
PatchInLegend(1).MarkerFaceColor = 'flat';%
%PatchInLegend(1).FaceAlpha = 0.2;
PatchInLegend(1).FaceVertexAlphaData = 1;
PatchInLegend(1).FaceVertexCData = PatchInLegend(1).FaceVertexCData(1);
PatchInLegend(1).Faces = 1;
PatchInLegend(1).Vertices = [0.1245,0.73];%PatchInLegend(1).Vertices(1,:);

end
