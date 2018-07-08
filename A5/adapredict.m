function [result]= adapredict(f,theta,y, beta, feature,label,weight)
[n,m] = size(feature);
result = zeros(n,1);
[iter,m] = size(theta);
h = zeros(iter,n);
for t = 1:iter
    [hypothesis] = testweaklearner(feature, label,theta(t),f(t), y(t),weight);
    h(t,:) = hypothesis;
end
for i = 1:n
    if sum((log(1./beta)).* h(:,i)) >= 0.5 * sum(log(1./beta))
        result(i) = 1;
    else
        result(i) =0;
    end
end
end