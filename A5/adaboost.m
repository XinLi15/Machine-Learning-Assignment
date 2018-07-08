function [theta,f,y,beta,weight]= adaboost(feature,label,iter)
%%input: feature, label, number of iteration
[n,m]=size(feature);
%initialize the weights
weight = ones(n,1)./n;
beta = zeros(iter,1);
theta = zeros(iter,1);
f = zeros(iter,1);
y = zeros(iter,1);
h = zeros(iter,n);
for t = 1:iter
    p = weight./sum(weight);
    [opt_theta,opt_f, opt_y] = weightedweaklearner(weight,feature, label);
    theta(t) = opt_theta;
    f(t) = opt_f;
    y(t) = opt_y;
    [hypothesis] = testweaklearner(feature, label,opt_theta,opt_f, opt_y,p);
    err = sum(p'*abs(hypothesis - label)); 
    %disp(err);
    h(t,:) = hypothesis;
    if err == 0
        err = 0.000000001;
    end
    beta(t) = err/(1-err);
    weight = weight.*(beta(t).^(1-abs(hypothesis - label)));
end

end