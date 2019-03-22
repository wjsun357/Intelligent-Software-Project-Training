clear all
clc
%读取数据
data = load('../Data/Normal Baseline Data/Normal_0.mat','X097_DE_time');
data = data.X097_DE_time;
%小波包分解

N = 5;
num = 8192;
%{
[tt] = wpdec(data(1:num,:)',N,'db10');
plot(tt);
%wpviewcf(tt,1);
count = 1;
for i = 0:N
    for j = 0:2^i-1
        WF(count,:) = wprcoef(tt,[i,j])';
        count = count+1;
    end
end
%}
%{
%LTSA
min_val=1;
record_k=0;
for i = 10:20
    i
    [T,B,idxNN] = LTSA(WF,i,1);
    %PE
    [H] = PE(T',6,3)
    if H < min_val
        record_k = i;
        min_val = H;
    end
end
%}
ALPHA = 2;
%此处参考https://wenku.baidu.com/view/1971380bf12d2af90242e690.html
%http://blog.sina.com.cn/s/blog_6c41e2f30101b0se.html
%https://blog.csdn.net/zhaoyuxia517/article/details/78013139
%https://blog.csdn.net/u010060391/article/details/42709317
[C,L] = wavedec(data(1:num,:)',N,'db10');
cA5 = appcoef(C,L,'db10',5);
cD5 = detcoef(C,L,5);
cD4 = detcoef(C,L,4);
cD3 = detcoef(C,L,3);
cD2 = detcoef(C,L,2);
cD1 = detcoef(C,L,1);
SIGMA1 = median(abs(cD1))/0.6745;
SIGMA2 = median(abs(cD2))/0.6745;
SIGMA3 = median(abs(cD3))/0.6745;
SIGMA4 = median(abs(cD4))/0.6745;
SIGMA5 = median(abs(cD5))/0.6745;
THR1 = wbmpen(C,L,SIGMA1,ALPHA);
THR2 = wbmpen(C,L,SIGMA2,ALPHA);
THR3 = wbmpen(C,L,SIGMA3,ALPHA);
THR4 = wbmpen(C,L,SIGMA4,ALPHA);
THR5 = wbmpen(C,L,SIGMA5,ALPHA);
%C=[cA5,cD5,cD4,cD3,cD2,cD1];
THTR = [THR1,THR2,THR3,THR4,THR5];
%[thr,nkeep]=wdcbm(C,L,3);
[XC,CXC,LXC,PERF0,PERFL2]=wdencmp('lvd',C,L,'db10',N,THTR,'s');
figure(1);
subplot(211);
plot(XC);
subplot(212);
plot(data(1:num,:)');
%A0=waverec(C,L,'db10');
