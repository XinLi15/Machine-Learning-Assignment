function label_vote = one_vote(label,bagidlist)
bagid = unique(bagidlist);
for k = 1:length(bagid)
    i = bagid(k);
    if sum(label(bagidlist == i) == 1) >=1
        label(bagidlist == i) = ones(size(label(bagidlist == i)));
    end
end
label_vote = label;
end
