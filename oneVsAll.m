function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1); % 10 * 401

% Add ones to the X data matrix
X = [ones(m, 1) X];


     % Set Initial theta
     initial_theta = zeros(n + 1, 1);
     
     % Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter', 50);
 
     % Run fmincg to obtain the optimal theta
     % This function will return theta and the cost 
     
     % Variable 'X' contains data in dimension (5000 * 400). 
     % 5000 = Total no. of training examples, 400 = 400 pixels / training sample (digit image)
     % Total no. Features  = 400
     
    for c = 1:num_labels 
        all_theta(c,:) = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
        % remember y (5000*1) is an array of labels i.e. it contains actual 
        % digit names (y==c) will return a vector with values 0 or 1. 1 at places where y==c 
        
        % 't' is passed as dummy parameter which is initialized with 'initial_theta' first
        % then subsequent values are choosen by fmincg [Note: Its not a builtin function like fminunc
        
        % fmincg will consider all training data having label c (1-10 note
        % 0 is mapped to 10) and find the optimal theta vector for it (Classifying white pixels with gray pixels). same
        % process is repeated for other classes
    end
end
