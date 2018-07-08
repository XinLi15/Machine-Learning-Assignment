function  [opt_theta,opt_f, opt_y] = weightedweaklearner(weight,feature, label)
single_label = unique(label);
[n,m] = size(feature);
% threshold
% optimal feature
% 1 for <, 2 for >.
%result = ones(n,1);
theta_num = 0;
slice = 50; % decide the gap of theta
for i = 1:m % for every feature
    range = (max(feature(:,i)) - min(feature(:,i)));
    [d,num] = size(min(feature(:,i)):range/slice:max(feature(:,i)));
    theta_num = theta_num + num;
end
record = zeros(2*theta_num,4);
r = 0;
for i = 1:m % for every feature
    range = (max(feature(:,i)) - min(feature(:,i)));
    for theta = min(feature(:,i)):range/slice:max(feature(:,i)) % for different theta
        for y = 1:2 % for every sign
            r = r + 1;
            result = ones(n,1)*single_label(1);
            %disp(sum(result));
            for j = 1:n % for every data point
                if y == 1
                    if feature(j,i) < theta
                        result(j) = single_label(2);
                    %else
                        %result(j) = 1;
                    end
                else
                    if feature(j,i) > theta
                        result(j) = single_label(2);
                    %else
                        %result(j) = 1;
                    end
                end                         
            end
            %disp(sum(result));
            record(r,1) = sum(weight' * abs(result - label)); 
            record(r,2) = theta;
            record(r,3) = i;
            record(r,4) = y;
        end
    end
end
least_err = min(record(:,1));
[row,col] = find(record(:,1) == least_err);
%disp(row);
opt_theta = record(row(1),2); 
opt_f = record(row(1),3);  
opt_y = record(row(1),4); 
%disp(record(row,:));
end
