% Exercise: Aggregating Algorithm (AA)

clear all;
load coin_data;

d = 5;
n = 213;

% compute adversary movez z_t
%%% your code here %%%
z = -log(r);

% compute strategy p_t (see slides)
%%% your code here %%%
% compute cumulative loss
L = nan(size(z));
for i = 1:size(z,1)
    L(i,:)=sum(z(1:i,:),1);
end

%compute strategy
p = nan(size(z));
p(1,:) = ones(1,size(z,2))/size(z,2);
for i = 2:size(z,1)
    p(i,:)=exp(-L(i-1,:))/sum(exp(-L(i-1,:)));
end
% compute loss of strategy p_t
%%% your code here %%%
loss_p = -log(sum(p.*exp(-z),2));

% compute losses of experts
%%% your code here %%%
losses_e =  L(213,:);

% compute regret
%%% your code here %%%
regret = sum(loss_p)- min(losses_e,[],2);

% compute total gain of investing with strategy p_t
%%% your code here %%%
total_gain =exp(-sum(loss_p));
%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
