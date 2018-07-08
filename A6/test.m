% load the image
file = dir('/home/xin/Desktop/ML/sival_apple_banana/apple/*.jpg');
NF = length(file);
images_apple = cell(NF,1);
for k = 1 : NF
  images_apple{k} = imread(fullfile('sival_apple_banana/apple', file(k).name));
end

file = dir('/home/xin/Desktop/ML/sival_apple_banana/banana/*.jpg');
NF = length(file);
images_banana = cell(NF,1);
for k = 1 : NF
  images_banana{k} = imread(fullfile('sival_apple_banana/banana', file(k).name));
end


%% 
%Q1.c
for i = 1:2
    im = images_apple{i};
    figure;
    subplot(1,2,1);
    imshow(im);
    win_size = 40;
    subplot(1,2,2);
    [feature, lab] = extractinstances(im, win_size); 
    imshow(lab,[min(lab(:)) max(lab(:))]);
end

for i = 1:2
    im = images_banana{i};
    figure;
    subplot(1,2,1);
    imshow(im);
    win_size = 40;
    subplot(1,2,2);
    [feature, lab] = extractinstances(im, win_size); 
    imshow(lab,[min(lab(:)) max(lab(:))]);
end

%%
%Q1.d
win_size = 40;
[dataset, bags] = gendatmilsival(images_apple, images_banana, win_size);
%%
scatterd(dataset);
legend('apple','banana');
bagid = getident(dataset,'milbag');
%%
%train
[train_data, test_data] = gendat(dataset,0.7);
%[train_data] = gendatmilsival(images_apple(1:40), images_banana(1:40), win_size);
W = train_data*fisherc;
%[test_data] = gendatmilsival(images_apple(41:60), images_banana(41:60), win_size);
label = test_data * W * labeld;
%%
%train
[train_data] = gendatmilsival(images_apple(1:40), images_banana(1:40), win_size);
W = train_data*fisherc;
[test_data] = gendatmilsival(images_apple(41:60), images_banana(41:60), win_size);
label = test_data * W * labeld;
%%
figure;
test_data = dataset;
train_data = test_data;
W = train_data*fisherc;
label = test_data * W * labeld;
%%
% shuffle the bag
all_feature = getdata(dataset);
all_label = getlab(dataset);
bagid_all =  getident(dataset,'milbag');
apple_idx = randperm(60,60)';
banana_idx = randperm(60, 60)' + 60;
train_apple_idx = apple_idx(1:40);
train_banana_idx = banana_idx(1:40);
test_apple_idx = apple_idx(41:60);
test_banana_idx = banana_idx(41:60);

train_idx = [train_apple_idx;train_banana_idx];
test_idx = [test_apple_idx; test_banana_idx];

train_bag = [];
train_bag_label = [];
train_label = [ones(60,1);...
    ones(60,1)*2];
for i = 1: 120
    if ismember(i, train_idx)
        train_bag = [train_bag; bags(i)];
        train_bag_label = [train_bag_label, train_label(i)];
    end
end
train_data = bags2dataset(train_bag, train_bag_label');

test_bag = [];
test_bag_label = [];
test_label =  [ones(60,1);...
    ones(60,1)*2];
for i = 1: 120
    if ismember(i, test_idx)
        %disp(i);
        test_bag = [test_bag; bags(i)];
        test_bag_label = [test_bag_label, test_label(i)];
    end
end
test_data = bags2dataset(test_bag, test_bag_label');

%%
W = train_data*fisherc;
train_label = train_data * W * labeld;
label = test_data * W * labeld;

scatterd(test_data);
hold on;

true_label = getlab(test_data);
n = length(true_label);
test_feature = getdata(test_data);
bagid = getident(test_data,'milbag');
prelab = combineinstlabels(label,bagid);
result = prdataset(test_feature, prelab);
% figure;
scatterd(result,'.');
err_rate = sum(true_label ~= prelab)/n;
disp(err_rate);

train_label =  getlab(train_data);
train_bagid = getident(train_data,'milbag');
train_prelab = combineinstlabels(train_label,bagid);

[n, none] = size(true_label);
train_err_rate = sum(train_label ~= train_prelab)/length(train_label);
disp(train_err_rate );
%% calculate miss-classification

bag_labels = prelab;
%bagid = test_bag_idx;
AA = 0;
AB = 0;
BA = 0;
BB = 0;
bag= unique(bagid);
disp(length(bag));
for k = 1:length(bag)
    i = (bagid == bag(k));
   if true_label(i)== bag_labels(i)
       if true_label(i)==1 %TP (is apple)
           AA = AA+1;
       else                   %TN (is banana)
           BB = BB+1;
       end
   else
       if true_label(i)==1 %FN (apple but classied as banana)
           AB = AB+1;
       else                   %FP (banana but classified as apple)
           BA = BA+1;
       end
   end
end
disp(sprintf('AA: %d,  AB: %d,  BA: %d,  BB: %d',AA,AB,BA,BB));


%%
%%
% MILES
X = dataset ;
bagidlist = getident(X,'milbag');
bagid = unique(bagidlist);
all_label = getlab(X);
theta = 10;
[x, y] = size(X);
all_m = zeros(max(bagid), x);
all_bag_label = zeros(max(bagid), 1);
for i = 1 : max(bagid)
    Bag = X(bagidlist == i,:);
    m = bagembed(X, Bag, theta);
    label = unique(all_label(bagidlist == i));
    all_m(i,:) = m;
    all_bag_label(i,:) = label;
end
M = prdataset(all_m, all_bag_label);
%%
% train
[train_data, test_data] = gendat(M,0.7);
train_data = M;
W = train_data*liknonc;
test_data = M;
train_err = train_data * W * testc;
disp(train_err);
%
true_label = getlab(test_data);
label = test_data * W * labeld;
err = test_data * W * testc;
disp(err);
%
%% calculate miss-classification
bagid = getident(X,'milbag');
true_label = getlab(X);
bag_labels = label;
AA = 0;
AB = 0;
BA = 0;
BB = 0;
bag= unique(bagid);
disp(length(bag));
for k = 1:length(bag)
    i = find(bagid == bag(k));
   if unique(true_label(i))== bag_labels(k)
       if unique(true_label(i))==1 %TP (is apple)
           AA = AA+1;
       else                   %TN (is banana)
           BB = BB+1;
       end
   else
       if unique(true_label(i))==1 %FN (apple but classied as banana)
           AB = AB+1;
       else                   %FP (banana but classified as apple)
           BA = BA+1;
       end
   end
end
disp(sprintf('AA: %d,  AB: %d,  BA: %d,  BB: %d',AA,AB,BA,BB));

%%
[train_data, test_data] = gendat(dataset,0.7);
train_data = dataset;
loop = 50;
[W, label] = mi_svm2(train_data);
% train_err = train_data * W * testc;
% disp(train_err);
close all;
%%
bagidlist = getident(test_data,'milbag');
true_label = getlab(test_data);
n = length(true_label);
label = test_data * W * labeld;
label_vote = one_vote(label,bagidlist);
err = sum(true_label ~= label)/n;
disp(err);
err = sum(true_label ~= label_vote)/n;
disp(err);
%% calculate miss-classification
bagid = getident(X,'milbag');
true_label = getlab(X);
bag_labels = label;
AA = 0;
AB = 0;
BA = 0;
BB = 0;
bag= unique(bagid);
disp(length(bag));
for k = 1:length(bag)
    i = find(bagid == bag(k));
   if unique(true_label(i))== unique(bag_labels(i))
       if unique(true_label(i))==1 %TP (is apple)
           AA = AA+1;
       else                   %TN (is banana)
           BB = BB+1;
       end
   else
       if unique(true_label(i))==1 %FN (apple but classied as banana)
           AB = AB+1;
       else                   %FP (banana but classified as apple)
           BA = BA+1;
       end
   end
end
disp(sprintf('AA: %d,  AB: %d,  BA: %d,  BB: %d',AA,AB,BA,BB));
%%
test_feature = [];
test_label = [];
for i = 1:length(test_apple_idx)
    test_feature = [test_feature; ...
        all_feature(bagid_all == test_apple_idx(i),:)];
    test_label = [test_label; ...
        all_label(bagid_all == test_apple_idx(i))];
end
for i = 1:length(test_banana_idx)
    test_feature = [test_feature; ...
        all_feature(bagid_all == test_banana_idx(i),:)];
    test_label = [test_label; ...
        all_label(bagid_all == test_banana_idx(i))];
end
test_bag_idx = [test_apple_idx; test_banana_idx ];
test_data = prdataset(test_feature, test_label);
%%
test_feature = [];
test_label = [];
test_bag = [];
for i = 1:length(test_apple_idx)
    test_feature = [test_feature; ...
        all_feature(bagid_all == test_apple_idx(i),:)];
    test_label = [test_label; ...
        all_label(bagid_all == test_apple_idx(i))];
    test_bag = [test_bag;...
        test_apple_idx(i) * ones(sum(bagid_all == test_apple_idx(i)),1)];
end
for i = 1:length(test_banana_idx)
    test_feature = [test_feature; ...
        all_feature(bagid_all == test_banana_idx(i),:)];
    test_label = [test_label; ...
        all_label(bagid_all == test_banana_idx(i))];
    test_bag = [test_bag; ...
        test_banana_idx(i) * ones(sum(bagid_all == test_banana_idx(i)),1)];
end
test_data = bags2dataset(bags,baglab);
% test_bag_idx = [test_apple_idx; test_banana_idx ];
% test_data = prdataset(test_feature, test_label);