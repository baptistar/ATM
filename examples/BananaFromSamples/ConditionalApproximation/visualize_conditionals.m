clear; close all; clc

%% Plot joint density

% define prior and likelihood
prior_x = @(x) normpdf(x,0,0.5);
likelihood = @(y,x) normpdf(y - x.^2,0,0.1);

% define domain
dom_x = [-1.2,1.2];
dom_y = [-0.3,1.1];
xx = linspace(dom_x(1),dom_x(2),1000);
yy = linspace(dom_y(1),dom_y(2),1000);
[X,Y] = meshgrid(xx,yy);

% evaluate joint
joint_pdf = prior_x(X(:)) .* likelihood(Y(:), X(:));
joint_pdf = reshape(joint_pdf, size(X,1), size(X,2));

figure
hold on
contourf(X,Y,joint_pdf, 'LineWidth', 1);
z = brewermap([],'Blues');
colormap(z(1:200,:))
yst = 0; plot(xx,yst*ones(1,1000),'--k','LineWidth',2);
yst = 0.25; plot(xx,yst*ones(1,1000),'--k','LineWidth',2);
yst = 0.5; plot(xx,yst*ones(1,1000),'--k','LineWidth',2);
set(gca,'FontSize',16)
xlabel('$x$','FontSize',22)
ylabel('$y$','FontSize',22)
xlim(dom_x)
ylim(dom_y)
set(gca,'YTick',[-0.25,0,0.25,0.5,0.75,1]);
%set(gca,'XTick',[]);
%set(gca,'YTickLabels',[]);
%set(gca,'XTickLabels',[]);
hold off
print('-depsc','joint_xy')
%close all

%% Plot conditionals

yst_vect = [0, 0.25, 0.5];
yst_str = {'y=0','y=0.25','y=0.5'};
for i=1:length(yst_vect)

    % extract yst
    yst = yst_vect(i);
    
    % define true posterior
    xx = linspace(dom_x(1),dom_x(2),10000);
    post_tilde = @(x) prior_x(x) .* likelihood(yst,x);
    norm_const = trapz(xx, post_tilde(xx'));
    post = @(x) post_tilde(x)/norm_const;
    pi_post = post(xx');
    
    figure
    hold on
    plot(xx,pi_post,'-k');
    xlim(dom_x)
    xlabel('$x$','FontSize',22)
    ylabel(['$\pi_{\mathbf{Y}|\mathbf{X}}(x|' yst_str{i} ')$'],'FontSize',22)
    set(gca,'YTick',[]);
    set(gca,'XTick',[]);
    set(gca,'YTickLabels',[]);
    set(gca,'XTickLabels',[]);
    hold off
    %print('-depsc',['post_y' num2str(i)])
    %close all
    
    % define true map
    C_pi_x = cumtrapz(xx, post_tilde(xx')/norm_const);
    S_KR = sqrt(2)*erfinv(2*C_pi_x - 1);
    
    figure
    hold on
    plot(xx,S_KR,'-k');
    %plot(xx, TM.evaluate([yst*ones(numel(xx),1),xx']))
    xlabel('$x$','FontSize',22)
    ylabel(['$S^{\mathcal{X}}(' yst_str{i} ',x)$'],'FontSize',22)
    set(gca,'YTick',[]);
    set(gca,'XTick',[]);
    set(gca,'YTickLabels',[]);
    set(gca,'XTickLabels',[]);
    xlim([-0.75,0.75])
    hold off
    %print('-depsc',['map_y' num2str(i)])

end

% 