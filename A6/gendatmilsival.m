function [dataset, bags] = gendatmilsival(imgset1, imgset2, win_size)
% input: all apple and banana-images
% output: a MIL Prtools dataset
[n1, none] = size(imgset1);
[n2, none] = size(imgset2);
bags = cell((n1 + n2),1);
%bag2 = cell(n2,1);



for i = 1:n1
    [feature, lab] = extractinstances(imgset1{i}, win_size);
    [seg, none] = size(feature);
    bag1 = cell(seg,1);
    for j = 1: seg
        bag1{j} = feature(j,:);
    end
    bags{i} = bag1;
end
for i = 1:n2
    [feature, lab] = extractinstances(imgset2{i}, win_size);
    [seg, none] = size(feature);
    bag2 = cell(seg,1);
    for j = 1: seg
        bag2{j} = feature(j,:);
    end
    bags{n1+i} = bag2;
end

% 1 for apple, 2 for banana
%[n, m] = size(bag1);
label_apple = ones(n1,1);
%[n, m] = size(bag2); 
label_banana = ones(n2,1)*2;
baglab = [label_apple; label_banana];
dataset = bags2dataset(bags,baglab);
end