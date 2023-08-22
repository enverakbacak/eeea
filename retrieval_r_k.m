clear all;
close all;
clc;

load('./Datasets/hmdb/hashCodes/usual/hashCodes_32.mat');
data = hashCodes_32;
N = length(data);

load('./Datasets/hmdb/hashCodes/usual/features_32.mat');
features = features_32;
load('./Datasets/hmdb/labels.mat');
labels = labels;

load('./Datasets/hmdb/hashCodes/usual/hashCodes_test_32.mat');
data_test = hashCodes_test_32;
%data_test = data_test(best_q_idx,:);

load('./Datasets/hmdb/hashCodes/usual/features_test_32.mat');
features_test = features_test_32;
%features_test = features_test(best_q_idx,:);

load('./Datasets/hmdb/labels_test.mat');
labels_test = labels_test;
%labels_test = labels_test(best_q_idx,:);

hR = 2; % Hamming Radious;
i  = 1;
n  = length(data_test);
k  = 100; % For top-k retrieved items!

%sum_Precision = 0;
%sum_Recall    = 0;
for l = i:n

    query                       = repmat(data_test(l,:),N,1);
    query_label(l,:)            = labels_test(l ,:);

    dist                        = xor(data, query);
    hamming_dist{l,:}           = sum(dist,2);
    hamming_dist_hR{l,:}        = hamming_dist{l,:}  <= hR;    % Hamming Radious
    r_index{l,:}                = find(hamming_dist_hR{l,:});
    Retrieved_Items{l,:}        = r_index{l,:};

    %r_features                  = features(r_index{l,:}, :);    % Features 
    %euclidian_dist              = pdist2(features_test(l,:),  r_features ); % Euclidean dists for reranking
    %euclidian_dist              = euclidian_dist';

    %decision_matrix             = [r_index{l,:} euclidian_dist];  
    %decision_matrix_sorted      = sortrows(decision_matrix, 2); 
    %Retrieved_Items{l,:}        = decision_matrix_sorted(:, 1);
    Retrieved_Items{l,:} (isempty(Retrieved_Items{l,:} ))=0; 

    a{l,:}                                  = zeros(k,1);
    a{l,:}(1:size(Retrieved_Items{l,:}),1)  = Retrieved_Items{l,:};
    Retrieved_Items_k{l,:}                  = a{l,:}(1:k, 1);
    Retrieved_Items_k{l,:}(Retrieved_Items_k{l,:} == 0) = [];
    Retrieved_Items_Labels_k                = labels(Retrieved_Items_k{l,:},:);
   

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
    avg_Recall(l,:) = sum(Recall{l,:}(:,1)  .* diff{l,:}(:,1) ) / num_nz(l,:);
    avg_Recall(isnan(avg_Recall))=0; 
        
 end
 
mean_Recall     = cellfun(@mean,Recall);
mean_Precision  = cellfun(@mean,Precision);
Precision_AT_N  = mean( mean_Precision );
%plot(normalize(mean_Precision, "range"), normalize(mean_Recall, "range")  );
%plot(mean_Precision,  mean_Recall , '.' );
%stairs(mean_Precision,  mean_Recall , 'ro' );

mAP       = sum(avg_Precision(:,1)) /(n-i+1);
ACC       = sum(acc(:,1)) / (n-i+1);
best_q_idx = find(avg_Precision > .99);
