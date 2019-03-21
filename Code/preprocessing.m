clear all
clc
%读取数据
%data = load('../Data/Normal Baseline Data/Normal_0.mat','X097_DE_time');
%data = data.X097_DE_time;
tau=unidrnd(1,2500,1)*pi;
n=linspace(0,2500,2500);
r=linspace(0,1/20480*2500,2500);
x=linspace(0,2*pi,2500);
%x=n(i)-r(i)*20480/40-tau(i))/20480
for i=1:2500
    data(i) = exp(-500*x(i))*sin(2*pi*1500-x(i));
end
plot(data);
%小波包分解
%{
N = 5;
num = 1024;
[tt] = wpdec(data(1:num,:),N,'db10');
plot(tt);
wpviewcf(tt,1);
count = 1;
for i = 0:N
    for j = 0:2^i-1
        WF(count,:) = wprcoef(tt,[i,j])';
        count = count+1;
    end
end
%LTSA
min_val=1;
record_k=0;
for i = 10:20
    i
    [T,B,idxNN] = LTSA(WF,i,1);
    [H] = PE(T',6,3)
    if H < min_val
        record_k = i;
        min_val = H;
    end
end
%}