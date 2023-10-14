clear ;clc
N = 400;T = 1;
x0 = [1000,10,1000,10]';
xA = [];zA = [x0(1),x0(3)];
%model-1,匀速运动
% x = A1*x + G1*sqrt(Q1)*[randn,randn]';
A1 = [1,T,0,0;
      0,1,0,0;
      0,0,1,T;
      0,0,0,1];
G1=[T^2/2,    0;
    T,      0;
    0,      T^2/2;
    0,      T] ;
Q1=[0.1^2 0;
    0 0.1^2];
%model-2,匀速转弯模型
A2=CreatCTF(-pi/360,T);
G2=CreatCTT(T);
Q2=[0.0144^2 0;
    0 0.0144^2];

% 产生真实数据
x = x0;
for k = 1:60%匀速直线
   x = A1*x + G1*sqrt(Q1)*[randn,randn]';
    xA =[xA x];    
end
for k = 61:90%匀速圆周转弯
   x = A2*x + G2*sqrt(Q2)*[randn,randn]';  
    xA =[xA x];
end
for k = 91:130%匀速直线
   x = A1*x + G1*sqrt(Q1)*[randn,randn]';
    xA =[xA x];
end 
plot(xA(1,:),xA(3,:),'b-')
save('data','xA','A1','N','x0','G1','Q1','A2','G2','Q2');
title('运动轨迹')
figure
plot(xA(2,:),xA(4,:),'g-')




