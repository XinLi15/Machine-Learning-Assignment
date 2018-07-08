function [m] = bagembed(X, Bag, theta)
% input: instances from all bags, a set of labeled bags, parameter theta^2
[n, m] = size(Bag);
[x, y] = size(X); 
m = zeros(1, x);
for i = 1:x
    d_2 = sum(abs(getdata(Bag)-getdata(X(i,:))).^2,2);
    d_exp = exp(-d_2/theta^2);
    %d_2 = sum(abs(getdata(Bag)-getdata(X(i,:))),2);
    %d_exp = exp(-d_2.^2S/theta^2);
    m(i) = max(d_exp);
end
end 