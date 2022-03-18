clear all
close all

out=load('SAMP_mes7_i_1mode_lkl.txt');
List_obs=load('List_obs_neat.txt');
%List_obs=load('obs_wihtout_noise_neat.txt');
mu_0=[0.5 3];
sigma_0=[0.64 0; 0 0.25];


prior=MultivariateGaussian(2,mu_0,sigma_0);


x2=linspace(0.8,3.8,50);
x1=linspace(-3,3,50);
[X,Y]=meshgrid(x1,x2);

% 
% figure
% for k=23
% surf(X,Y,reshape(log(out(:,k)),50,50))
% title(['log - Model output ',num2str(k)])
% zlim([5.48 5.85])
% caxis([5.48 5.85])
% end
% 
% figure
% for k=23
% surf(X,Y,reshape(out(:,k),50,50))
% title(['Model output ',num2str(k)])
% zlim([240 370])
% caxis([240 370])
% end
% 
% return
% figure
% for k=1:50
% subplot(5,10,k)
% pcolor(X,Y,reshape(out(:,k),50,50))
% title(['Model output ',num2str(k)])
% shading interp
% end
% 
% 
% figure
% for k=1:50
% subplot(5,10,k)
% 
% FX=out(:,k);
% 
% lkl=(1/7*sqrt(2*pi)).*exp(-0.5*((List_obs(k)-FX)/7).^2);
% pcolor(X,Y,reshape(lkl,50,50))
% title(['True lkl ',num2str(k)])
% shading interp
% end

lkl=exp(prior.log_pdf([X(:) Y(:)]));

figure
for k=1:50
subplot(5,10,k)

FX=out(:,k);

lkl=(1/7*sqrt(2*pi)).*exp(-0.5*((List_obs(k)-FX)/7).^2).*lkl;
pcolor(X,Y,reshape(lkl,50,50))
title(['Tr. post',num2str(k)])
shading interp
end


figure
pcolor(X,Y,reshape(lkl,50,50))
title(['True final post ',num2str(k)])
shading interp
