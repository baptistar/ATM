clear all
close all

L_S=load('SAMP_mes7_i_1mode_2.txt');
L_S_geo=load('LS_KL_1mode_geo_2.txt');
geo_true=load('iceWater.txt');
geo_true_2=load('iceWater_2.txt');


figure
plot(linspace(0,20,200)',L_S_geo(7254,:))
hold on
plot([0 20],[5 5],'k')
plot([0 20],[0 0],'k')
plot([0 20],[8 8],'k')
ylim([0 8])
axis image



figure
plot(L_S(:,30))



figure
plot(linspace(0,20,200)',geo_true)
hold on
plot(linspace(0,20,200)',geo_true_2)
plot([0 20],[5 5],'k')
plot([0 20],[-2 -2],'k')
plot([0 20],[8 8],'k')
ylim([0 8])
axis image

KL_5=load('KL_modes_5.txt');


figure
set(gca,'fontsize', 16)
plot(linspace(0,20,200)',2.5*KL_5,'linewidth',1.5)

figure
set(gca,'fontsize', 16)
plot(linspace(0,20,200)',geo_true-2.5*KL_5(:,1)-2.5,'linewidth',1.5)


figure
set(gca,'fontsize', 16)
plot(linspace(0,20,200)',2.5*ones(1,200),'linewidth',1.5)
