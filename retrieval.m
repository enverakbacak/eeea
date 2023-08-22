clear all;
close all;
clc;

load('./Datasets/UCF/hashCodes/usual/hashCodes_64.mat');
data = hashCodes_64;
N = length(data);

load('./Datasets/UCF/hashCodes/usual/features_64.mat');
features = features_64;
load('./Datasets/UCF/labels.mat');
labels = labels;

load('./Datasets/UCF/hashCodes/usual/hashCodes_test_64.mat');
data_test = hashCodes_test_64;
%data_test = data_test(best_q_idx,:);

load('./Datasets/UCF/hashCodes/usual/features_test_64.mat');
features_test = features_test_64;
%features_test = features_test(best_q_idx,:);

load('./Datasets/UCF/labels_test.mat');
labels_test = labels_test;
%labels_test = labels_test(best_q_idx,:);

i  = 1;
n  = length(data_test);


for l = i:n

    query                       = repmat(data_test(l,:),N,1);
    query_label(l,:)            = labels_test(l ,:);
    dist                        = xor(data, query);
    hamming_dist                = sum(dist,2);
    [s_hamming_dist,r_index]    = sort(hamming_dist,'ascend');
 
    r_features                  = features(r_index, :);    % Features 
    euclidian_dist              = pdist2(features_test(l,:),  r_features );  
    euclidian_dist              = euclidian_dist';

    decision_matrix             = [r_index euclidian_dist];  
    decision_matrix_sorted      = sortrows(decision_matrix, 2); 
    Retrieved_Items{l,:}        = decision_matrix_sorted(:, 1);

    Retrieved_Items_Labels{l,:} = labels(Retrieved_Items{l,:},:);
    
    diff{l,:} = ismember(Retrieved_Items_Labels{l,:}, query_label(l,:)   , 'rows'); 
    if isempty( diff{l,:})
            diff{l,:} = 0;
    end

    num_nz(l,:) = nnz( diff{l,:}(:,1) );
    s{l,:} = size(diff{l,:}(:,1), 1);
    
    for j=1:s{l,:};
        
        CUMM{l,:} = cumsum(diff{l,:});          
        Precision{l,:}(j,1) = ( CUMM{l,:}(j,1)  ) / j;              
        Recall{l,:}{j,1} = ( CUMM{l,:}(j,1)  ) / (num_nz(l,:)); %                
    end  
    
    acc(l,:) = num_nz(l,:) / s{l,:};   
    avg_Precision(l,:) = sum(Precision{l,:}(:,1)  .* diff{l,:}(:,1) ) / num_nz(l,:);
    avg_Precision(isnan(avg_Precision))=0;
    
 end

%mean_Recall     = cellfun( @mean, Recall );
mean_Precision  = cellfun( @mean, Precision );
Precision_AT_N  = mean( mean_Precision );
%plot(normalize(mean_Precision, "range"), normalize(mean_Recall, "range")  );

mAP       = sum(avg_Precision(:,1)) /(n-i+1);
ACC       = sum(acc(:,1)) / (n-i+1);
best_q_idx = find(avg_Precision > .7); 


