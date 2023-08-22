clear all;
close all;
clc;

load('./Datasets/UCF/hashCodes/pca/hashCodes_32.mat');
data = hashCodes_32;
N = length(data);

load('./Datasets/UCF/hashCodes/pca/features_32.mat');
features = features_32;
load('./Datasets/UCF/labels.mat');
labels = labels;

load('./Datasets/UCF/hashCodes/pca/hashCodes_test_32.mat');
data_test = hashCodes_test_32;
%load('./Datasets/UCF/best_q_idx.mat');
%data_test = data_test(best_q_idx,:);

load('./Datasets/UCF/hashCodes/pca/features_test_32.mat');
features_test = features_test_32;
%features_test = features_test(best_q_idx,:);

load('./Datasets/UCF/labels_test.mat');
labels_test = labels_test;
%labels_test = labels_test(best_q_idx,:);

i  = 1;
n  = length(data_test);
k  = 500; % For top-k retrieved items!

for l = i:n

    query                       = repmat(data_test(l,:),N,1);
    query_label(l,:)            = labels_test(l ,:);

    dist                        = xor(data, query);
    hamming_dist{l,:}           = sum(dist,2);
    [s_hamming_dist,r_idx]      = sort(hamming_dist{l,:},'ascend'); 
    r_index{l,:}                = r_idx;  
    r_index_500{l,:}            = r_index{l,:}(1:500);

    r_features                  = features(r_index_500{l,:}(:,1), :);    % Features 
    euclidian_dist              = pdist2(features_test(l,:),  r_features ); 
    euclidian_dist              = euclidian_dist';
    decision_matrix             = [r_index_500{l,:}(:,1) euclidian_dist];  
    decision_matrix_sorted      = sortrows(decision_matrix, 2); 
    Retrieved_Items{l,:}        = decision_matrix_sorted(:, 1);
    %Retrieved_Items{l,:}        = r_index_500{l,:};
    Retrieved_Items_k{l,:}      = Retrieved_Items{l,:}(1:k,1);
    Retrieved_Items_Labels_k    = labels(Retrieved_Items_k{l,:},:);

    %a{l,:}                             = zeros(k,1);
    %a{l,:}(1:size(Retrieved_Items{l,:}),1)  = Retrieved_Items{l,:};
    %Retrieved_Items_k{l,:}             = a{l,:}(1:k, 1);
    %Retrieved_Items_k{l,:}(Retrieved_Items_k{l,:} == 0) = [];
    %Retrieved_Items_Labels_k           = labels(Retrieved_Items_k{l,:},:);

    diff{l,:} = ismember(Retrieved_Items_Labels_k, query_label(l,:)   , 'rows'); 
    if isempty( diff{l,:})
        diff{l,:} = 0;
    end

    num_nz(l,:) = nnz( diff{l,:}(:,1) );
    s{l,:} = size(diff{l,:}(:,1), 1);
    
    for j=1:s{l,:};
        
        CUMM{l,:} = cumsum(diff{l,:}); 

        Precision{l,:}(j,1) = ( CUMM{l,:}(j,1)  ) / j;  
        Precision{l,:}(isnan(Precision{l,:}))=0;
        Recall{l,:}(j,1) = ( CUMM{l,:}(j,1)  ) / (num_nz(l,:)); 
        Recall{l,:}(isnan(Recall{l,:}))=0;
    end  
    
    acc(l,:) = num_nz(l,:) / s{l,:};   
    avg_Precision(l,:) = sum(Precision{l,:}(:,1)  .* diff{l,:}(:,1) ) / num_nz(l,:);
    avg_Precision(isnan(avg_Precision))=0;    
    %avg_Recall(l,:) = sum(Recall{l,:}(:,1)  .* diff{l,:}(:,1) ) / num_nz(l,:);
    %avg_Recall(isnan(avg_Recall))=0; 
   
 end
 
%mean_Recall     = cellfun( @mean, Recall );
mean_Precision  = cellfun( @mean, Precision );
Precision_AT_N  = mean( mean_Precision );
%plot(normalize(mean_Precision, "range"), normalize(mean_Recall, "range")  );

mAP             = sum(avg_Precision(:,1)) /(n-i+1);
ACC             = sum(acc(:,1)) / (n-i+1);
best_q_idx      = find(avg_Precision > 0.48); 






