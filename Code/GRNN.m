function [result] = GRNN(x,data,num,o)
    %�����
    layer1 = data;%ά��Ϊ1
    %ģʽ��
    for i = 1:num
        layer2(i) = exp(-(x-layer1(i,1)).^2/(2*o.^2));
    end
    %��Ͳ�
    layer3D = sum(layer2);
    layer3j = 0;
    for i = 1:num
        layer3j = layer3j+layer2(i)*layer1(i,1);
    end
    %�����
    result=layer3j/layer3D;
end