[x,y,z]=peaks(50);
surf(x,y,z);%ע��ͼ��2��ͼ��3����Ĳ����ԣ��൱��ƽ����
%%������ά����x�ᣬy�ᣬz�ᣬ�����Ӧ���ݼ������������ȡֵ��Χ%%
set(gca,'xticklabel',[])
set(gca,'yticklabel',[])
set(gca,'zticklabel',[])
zlabel('Error');
title('Error Surface');