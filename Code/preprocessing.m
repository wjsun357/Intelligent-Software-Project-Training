clear all
clc
%读取数据
data = load('../Data/Normal Baseline Data/Normal_0.mat','X097_DE_time');
data = data.X097_DE_time;
%小波包分解
N = 4;
num = 1024;
[tt] = wpdec(data(1:num,:),N,'dmey');
plot(tt);
count = 1;
for i = 0:N
    for j = 0:2^i-1
        WF(count,:) = wprcoef(tt,[i,j])';
        count = count+1;
    end
end
%LTSA
[T,B,idxNN] = LTSA(WF,20,1);
[H] = PE(T',6,3);