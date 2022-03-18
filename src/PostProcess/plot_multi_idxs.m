function [] = plot_multi_idxs(Mat_idxs,V_writing,V_color,colorbar,titleString)
% Plot multi idxs in 2D
% Mat_idxs: list of multi_idxs (Nx2)
% V_writing: information to display on cells (default=order of addition)
% V_color: field by index (default=none)
% colorbar:
% titleString: legend of the colorbar if colobar exist

if (nargin < 2)
    V_writing = 1:length(Mat_idxs);
    V_color=ones(length(Mat_idxs),1);
end
if (nargin < 3)
    V_color=ones(length(Mat_idxs),1);
end
if (nargin < 4)
    colorbar=false;
    titleString='';
end
if (nargin < 5)
    titleString='';
end


V_color=V_color(:)';
V_writing=V_writing(:)';

Z=Mat_idxs;
order_max=max(max(Z))+1;

d=size(Z,2);

figure
for i=1:d
    for j=1:i
       if i ~= j
         p=(j-1)*(d-1)+i-j;
         subplot(d-1,d-1,p)
         [Z_red,idx_red]=unique(Z(:,[i j]),'rows','stable');
         plot_multi_idxs2d_raw(Z_red,order_max,V_writing(idx_red),V_color(idx_red),colorbar,titleString)
         xlabel(['variable ',num2str(i)])
         ylabel(['variable ',num2str(j)])
       end
    end
end


end

function [] = plot_multi_idxs2d_raw(Mat_idxs,d,V_writing,V_color,colorbar,titleString)
% Plot multi idxs in 2D
% Mat_idxs: list of multi_idxs (Nx2)
% V_writing: information to display on cells (default=order of addition)
% V_color: field by index (default=none)
% titleString: legend of the colorbar if V_color exist

V_color=V_color(:)';
V_writing=V_writing(:)';

Z=Mat_idxs;

Z=Z+1;


idx = sub2ind([d d], Z(:,1)', Z(:,2)');

m=max(Z);
nx=m(1);
ny=m(2);

Mat=zeros(d,d);
Mat(idx)=1;
Mat=Mat';

Cmat=zeros(d,d);
Clik=zeros(d,d);

Cmat(idx)=V_writing;
Clik(idx)=V_color;

Cmat=Cmat';
Clik=Clik';
Cmat(Cmat==0)=NaN;
Clik(Clik==0)=NaN;

fig=gcf;
fig.Color=[1 1 1];
imagesc(Clik)
set(gca,'Ydir','normal')
hold on
for k=1:ny
    l=Mat(k,:);
    L=find(l,1,'last');
    plot([0 L+0.5],[k+0.5 k+0.5],'k','linewidth',1.25)
end
for k=1:nx
    l=Mat(:,k);
    L=find(l,1,'last');
    plot([k+0.5 k+0.5],[0 L+0.5],'k','linewidth',1.25)
end
t = num2cell(Cmat(:)); % extact values into cells
t = cellfun(@num2str, t, 'UniformOutput', false); % convert to string
IndexC = strfind(t,'NaN');
Index = find(not(cellfun('isempty',IndexC)));
for i=Index'
    t{i}=' ';
end
[X,Y]=meshgrid(1:d,1:d);
text(X(:)-0.15, Y(:), t,'Color',[0.17 0.21 0.76],'FontSize',16);

xticks(1:d)
xticklabels(string(0:d-1))
yticks(1:d)
yticklabels(string(0:d-1))
ax = gca;
ax.FontSize = 16;
box off

if nargin>3
    colormap( [1 1 1;flip(hot(256))] )
    caxis( [min(Clik(:)) max(Clik(:))] )
    if colorbar
        hcb=colorbar('northoutside');
        colorTitleHandle = get(hcb,'Title');
        set(colorTitleHandle ,'String',titleString);
    end
else
    map=[1 1 1;  0 0.75 0.75];
    colormap(map)
end

end
