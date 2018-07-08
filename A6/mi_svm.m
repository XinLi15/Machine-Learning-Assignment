function W = mi_svm(train_data, loop)
% MI-SVM
bagidlist = getident(train_data,'milbag');
train_feature = getdata(train_data);
true_label = getlab(train_data); % change label to -1, 1.
true_label = true_label * (-2) + 3 ; % postive_label = 1; %for apple
[n,m] = size(train_data);
% training
new_data = prdataset(train_feature, true_label);
selection = zeros(n,loop);
for j = 1:loop
    W = new_data*svc;
    W_proba = round(+(new_data*W),4);
   % W_apple_proba = W_proba(:,1);
    % for every positiva bag
    old_label = new_data * W * labeld;
    new_label = old_label;
    postive_bags_idx = sort(unique(bagidlist(true_label == 1)));
    for k = 1:length(postive_bags_idx)
        i = postive_bags_idx(k);
        bag = new_data(find(bagidlist == i),:);
       % disp(bagidlist == i);
        % disp(size(bag));
        label_per_bag = bag * W * labeld;
        
        if(sum(label_per_bag+1)/2) == 0
            disp(1);
            max_value = max((W_proba(find(bagidlist == i),1)));
            idx = find(W_proba == max_value, 1);
            new_label(idx) = 1;
            disp(idx);
        end
    end
    err_rate = sum(true_label ~= new_label)/n;
    disp('loop:');
    disp(j);
    disp(err_rate);
    
    new_data = prdataset(train_feature, new_label);
    figure;
    scatterd(new_data);
    if old_label == new_label 
        break
    end
    old_label = new_label;
    
end

end
