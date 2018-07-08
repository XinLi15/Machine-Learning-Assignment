function [prelab] = combineinstlabels(labels,bagid)
bag = unique(bagid);
[bagnum, none] = size(bag);
[n, none] = size(labels);
prelab = ones(n, 1);
for k = 1: bagnum
%    for j = 1: len(bagid)
%        if bagid(i) == i

%    end
    i = bag(k);
    majorlab = mode(labels(bagid == i));
    prelab(bagid == i) = prelab(bagid == i) * majorlab;
end
%majorlab = mode(labels);
end