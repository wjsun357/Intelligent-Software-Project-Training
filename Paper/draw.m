[x,y,z]=peaks(50);
surf(x,y,z);%注意图像2和图像3坐标的差异性，相当于平移了
%%设置三维曲面x轴，y轴，z轴，标题对应内容及三个坐标轴的取值范围%%
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
set(gca,'zticklabel',[])
zlabel('Error');
title('Error Surface');