%%
%Q3
a = gendats();
figure;
scatterd(a);
feature = getdata(a);
label = getlabels(a);
[least_err,opt_theta,opt_f, opt_y] = weaklearner(feature, label);
%disp(least_err);

% multiply feature 2 by a factor of 10
feature2 = feature;
feature2(:,2) = feature2(:,2)*10;
b = prdataset(feature2, label);
figure;
scatterd(b);
[least_err,opt_theta,opt_f, opt_y]= weaklearner(feature2, label);

%%
%Q4
data = load('optdigitsubset.txt');
[n,m]=size(data);
example = data(2,:);
%imshow(reshape(example,[8 8]));

%train
zero_data = data(1:554,:);
one_data = data(555:1125,:);
label_01 = [zeros(50,1); ones(50,1)];
train_data = [zero_data(1:50,:); one_data(1:50,:)];
[least_err,opt_theta,opt_f, opt_y]= weaklearner(train_data, label_01);
disp([least_err./100,opt_theta,opt_f, opt_y]);
%test
test_data = [zero_data(51:554,:); one_data(51:571,:)];
test_label = [zero_data(51:554,1) * 0; one_data(51:571,1) * 0 + 1];
[n, m] = size(test_data);
single_label = unique(test_label);
result = ones(n,1)*single_label(1);
%disp(sum(result));
for j = 1:n % for every data point
    if opt_y == 1
        if test_data(j,opt_f) < opt_theta
            result(j) = single_label(2);
        %else
            %result(j) = 1;
        end
    else
        if test_data(j,opt_f) > opt_theta
            result(j) = single_label(2);
        %else
            %result(j) = 1;
        end
    end                         
end
err = sum(abs(result - test_label))/size(test_label,1); 
disp('error:');
disp(err); 
%%
all_err = zeros(50,1);
for i = 1:100
    %shuffle the data
    zero_data_shuffle = zero_data(randsample(1:length(zero_data),length(zero_data)),:);
    one_data_shuffle = one_data(randsample(1:length(one_data),length(one_data)),:);
    %train
    train_data = [zero_data_shuffle(1:50,:); one_data_shuffle(1:50,:)];
    [least_err,opt_theta,opt_f, opt_y]= weaklearner(train_data, label_01);
    %test
    test_data = [zero_data_shuffle(51:554,:); one_data_shuffle(51:571,:)];
    test_label = [zero_data_shuffle(51:554,1) * 0; one_data_shuffle(51:571,1) * 0 + 1];
    [n, m] = size(test_data);
    single_label = unique(test_label);
    result = ones(n,1)*single_label(1);
    %disp(sum(result));
    for j = 1:n % for every data point
        if opt_y == 1
            if test_data(j,opt_f) < opt_theta
                result(j) = single_label(2);
            %else
                %result(j) = 1;
            end
        else
            if test_data(j,opt_f) > opt_theta
                result(j) = single_label(2);
            %else
                %result(j) = 1;
            end
        end                         
    end
    err = sum(abs(result - test_label))/n;
    disp('error:');
    disp(err); 
    all_err(i) = err;
end    
mean_err = mean(all_err);
std_err = std(all_err);
plot(all_err,'r*');

%%
%Q5
a = gendats();
scatterd(a);
feature = getdata(a);
label = getlabels(a);
[n,m]=size(feature);
weight = ones(n,1)/n;
[opt_theta, opt_f, opt_y] = weightedweaklearner(weight, feature, label);
[result] = testweaklearner(feature, label,opt_theta,opt_f, opt_y, weight);
d = prdataset(feature, result);
figure;
scatterd(d,'.');

figure; %change weight
weight = [ones(n/2,1)*4;ones(n/2,1)];
weight = weight./sum(weight);
[opt_theta, opt_f, opt_y] = weightedweaklearner(weight, feature, label);
[result] = testweaklearner(feature, label,opt_theta,opt_f, opt_y, weight);
d = prdataset(feature, result);
scatterd(d,'.');

%%
%Q6
n = 50;
c = gendatb([n,n]);
feature = getdata(c);
%label = getlabels(b);
label = str2num(getlabels(c))-1;
new_c = prdataset(feature, label);
scatterd(new_c,'legend');
hold on;
iter = 100;
[theta,f,y,beta,weight]= adaboost(feature,label,iter);
[result]= adapredict(f,theta,y, beta, feature,label,weight);
err = sum(abs(result-label))/(2*n);
disp(err);
d = prdataset(feature, result);
%figure;
scatterd(d,'.','legend');
figure;
imagesc(weight'),colorbar;
figure;
plot(weight');

%%
%Q6
n = 50;
c = gendatb([n,n]);
feature = getdata(c);
%label = getlabels(b);
label = str2num(getlabels(c))-1;
new_c = prdataset(feature, label);
scatterd(new_c,'legend');
hold on;
iter = 100;
[theta,f,y,beta,weight]= adaboost(feature,label,iter);
[result]= adapredict(f,theta,y, beta, feature,label,weight);
err = sum(abs(result-label))/(2*n);
disp(err);
d = prdataset(feature, result);
%figure;
scatterd(d,'.','legend');
figure;
imagesc(weight'),colorbar;
figure;
plot(weight');
%%

n = 50;
e = gendats([n,n]);
figure;
scatterd(e);
hold on;
feature = getdata(e);
label = getlabels(b)-1;
%label = str2num(getlabels(e))-1;
iter = 100;
[theta,f,y,beta,weight]= adaboost(feature,label,iter);
[result]= adapredict(f,theta,y, beta, feature,label,weight);
err = sum(abs(result-label))/(2*n);
disp(err);
d = prdataset(feature, result);
%figure;
scatterd(d,'.','legend');
figure;
imagesc(weight'),colorbar;
figure;
plot(weight');
%%
N = 100;
[W,V,ALF] = adaboostc(c,weakc,N);
scatterd(c);
hold on;
plotc(W);
%%
%Q7
data = load('optdigitsubset.txt');
[n,m]=size(data);
example = data(2,:);
%imshow(reshape(example,[8 8]));

%train
zero_data = data(1:554,:);
one_data = data(555:1125,:);
label_01 = [zeros(50,1); ones(50,1)];
train_data = [zero_data(1:50,:); one_data(1:50,:)];


%test
test_iter = 100;
test_data = [zero_data(51:554,:); one_data(51:571,:)];
test_label = [zero_data(51:554,1) * 0; one_data(51:571,1) * 0 + 1];
all_err = zeros(test_iter,1);
all_train_err = zeros(test_iter,1);
all_weights = zeros(test_iter,100);
for i = 1:test_iter
    iter = i;
    [theta,f,y,beta,weight]= adaboost(train_data,label_01,iter);
    [train_result]= adapredict(f,theta,y, beta, train_data,label_01,weight);
    [result]= adapredict(f,theta,y, beta, test_data,test_label,weight);
    train_err = sum(abs(train_result-label_01))./(size(train_result,1));
    err = sum(abs(result-test_label))./(size(result,1));
    disp(iter);
    disp('err:');
    disp(err);
    all_err(i) = err;
    all_train_err(i) = train_err;
    all_weights(i,:) = weight./sum(weight); 
end
figure;
plot(all_train_err);
hold on;
plot(all_err);
legend('train error','test error');
%%
figure;
imagesc(weight'),colorbar;
figure;
plot(weight');
figure;
xlabel('number of iteration');
ylabel('objects');
imagesc(all_weights),colorbar;

figure;
example = train_data(18,:)';
imshow((reshape(example,[8 8])/255)','border','tight','initialmagnification','fit');
saveas(gcf,'obj16.png');

figure;
example = train_data(66,:)';
imshow((reshape(example,[8 8])/255)','border','tight','initialmagnification','fit');
fig = (reshape(example,[8 8])/255)';
saveas(gcf,'obj66.png');

figure;
example = train_data(86,:)';
imshow((reshape(example,[8 8])/255)','border','tight','initialmagnification','fit');
saveas(gcf,'obj86.png');