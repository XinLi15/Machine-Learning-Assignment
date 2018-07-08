function [result] = testweaklearner(test_data, test_label,opt_theta,opt_f, opt_y, p)
    %test
    [n, m] = size(test_data);
    single_label = unique(test_label);
    result = ones(n,1)*single_label(1);
    for j = 1:n % for every data point
    if opt_y == 1
        if test_data(j,opt_f) < opt_theta
            result(j) = single_label(2);
            %disp('update');
        end
    else
        if test_data(j,opt_f) > opt_theta
            result(j) = single_label(2);
            %disp('update');
        end
    end                         
    end
end

