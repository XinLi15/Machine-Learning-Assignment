function [W2, second_label_vote] = mi_svm2(train_data)
% MI-SVM
bagidlist = getident(train_data,'milbag');
train_feature = getdata(train_data);
true_label = getlab(train_data);
[n,m] = size(train_data);
postive_label = 1; %for apple
% first training
W = train_data*svc([],proxm('p',2),100);
W_proba = round(+(train_data*W),4);
first_label = train_data * W * labeld;
first_prelab = combineinstlabels(first_label,bagidlist); % majority voting
% find all positive bags
new_postive_feature = [];
bagid = unique(bagidlist);
postive_idx = [];
all_idx = 1: length(true_label);
all_idx = all_idx';
for k = 1:length(bagid) %%
    i = bagid(k);
    if unique(first_prelab((bagidlist == i))) == postive_label && unique(true_label((bagidlist == i))) == postive_label 
        proba_postive = W_proba(:,postive_label);
        max_instance = max(proba_postive((bagidlist == i)));
        idx1 = find(proba_postive == max_instance);
        idx2 = find(bagidlist == i);
        idx = find(ismember(idx1, idx2,'rows'));
        new_postive_feature = [new_postive_feature; train_feature(idx1(idx(1)),:)];
        postive_idx = [postive_idx; idx1(idx(1))];
    end
end 

% find all negative bags
only_positive_label = 2 * ones(length(true_label),1);
only_positive_label(postive_idx) = 1;
negative_bags = train_feature(true_label ~= postive_label,:);
negative_baglist = bagidlist(true_label ~= postive_label);
new_label = [ ones(length(new_postive_feature),1); 2 * ones(length(negative_bags),1)]; 
%generate new dataset
new_data = prdataset([new_postive_feature; negative_bags], new_label);
%new_data = prdataset(train_feature, only_positive_label);

%second train
W2 = new_data *svc([],proxm('p',2),100);
second_inter_label = new_data * W2 * labeld;
iter_data = prdataset(getdata(new_data), second_inter_label);
scatterd(iter_data, 3);
legend();
second_label = train_data * W2 * labeld;
disp(train_data * W2 * testc);
second_label_vote = one_vote(second_label,bagidlist);
iter_data = prdataset(getdata(train_data), second_label_vote);
scatterd(iter_data, 3);
err = sum(true_label ~= second_label_vote)/n;
disp(err);
end