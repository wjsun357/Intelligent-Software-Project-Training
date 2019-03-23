function [result] = GRNN(x,data,num,o)
    %输入层
    layer1 = data;%维度为1
    %模式层
    for i = 1:num
        layer2(i) = exp(-(x-layer1(i,1)).^2/(2*o.^2));
    end
    %求和层
    layer3D = sum(layer2);
    layer3j = 0;
    for i = 1:num
        layer3j = layer3j+layer2(i)*layer1(i,1);
    end
    %输出层
    result=layer3j/layer3D;
end