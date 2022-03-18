clear all
close all

Xtest=linspace(-5,5,100);

X=[Xtest(:) zeros(100,1)];

[FY,GY]=fun_f(map_f(X));

[FY3,G3]=truth_f(X);


G2=gradient(FY3,Xtest);
G1=gradient(FY,Xtest);

grad_map=grad_x_map_f(X);
[FY,GY]=fun_f(map_f(X));

Gtest=zeros(100,2);

for k=1:100
    Gtest(k,:)=mtimes(reshape(grad_map(k,:,:),2,2)',GY(k,:)');
end

grad_map2=permute(grad_map,[3 2 1]);

GY2=reshape(GY,100,1,2);
GY2=permute(GY2,[2 3 1]);
GY3=reshape(GY2,2,1,100);

Gtest2=pagemtimes(grad_map2,GY3);
Gtest2=permute(Gtest2,[3 1 2]);

figure
hold on
plot(Xtest,G3(:,1),'k--') % grad truth ana
%plot(Xtest,G2(:,1)) % grad truth num
%plot(Xtest,G1(:,1),'d') % grad num
plot(Xtest,Gtest(:,1),'*') %grad test
plot(Xtest,Gtest2(:,1),'o');


figure
hold on
plot(Xtest,G3(:,2),'k--') % grad truth ana
%plot(Xtest,G2(:,1)) % grad truth num
%plot(Xtest,G1(:,1),'d') % grad num
plot(Xtest,Gtest(:,2),'*') %grad test
plot(Xtest,Gtest2(:,2),'o');

