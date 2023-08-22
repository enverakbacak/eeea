clear all;
close all;
clc;

load('./Datasets/hmdb/hashCodes/pca/hashCodes_16.mat');
data = hashCodes_16;
N = length(data);

load('./Datasets/hmdb/hashCodes/pca/features_16.mat');
features = features_16;
load('./Datasets/hmdb/labels.mat');
labels = labels;

load('./Datasets/hmdb/hashCodes/pca/hashCodes_test_16.mat');
data_test = hashCodes_test_16;
%data_test = data_test(best_q_idx,:);

load('./Datasets/hmdb/hashCodes/pca/features_test_16.mat');
features_test = features_test_16;
%features_test = features_test(best_q_idx,:);

load('./Datasets/hmdb/labels_test.mat');
labels_test = labels_test;
%labels_test = labels_test(best_q_idx,:);

hR = 2; % Hamming Radious;
i  = 1;
n  = length(data_test);


for l = i:n

    query                       = repmat(data_test(l,:),N,1);
    query_label(l,:)            = labels_test(l ,:);

    dist                        = xor(data, query);
    hamming_dist{l,:}           = sum(dist,2);
    hamming_dist_hR{l,:}        = hamming_dist{l,:}  <= hR;    % Hamming Radious
    r_index{l,:}                = find(hamming_dist_hR{l,:});
     
    r_features                  = features(r_index{l,:}, :);    % Features 
    euclidian_dist              = pdist2(features_test(l,:),  r_features ); % Euclidean dists for reranking
    euclidian_dist              = euclidian_dist';
    decision_matrix             = [r_index{l,:} euclidian_dist];  
    decision_matrix_sorted{l,:}      = sortrows(decision_matrix, 2); 
    Retrieved_Items{l,:}        = decision_matrix_sorted{l,:}(:, 1);
    %Retrieved_Items{l,:}         = r_index{l,:};
    Retrieved_Items_Labels      = labels(Retrieved_Items{l,:},:);
    
    diff{l,:} = ismember(Retrieved_Items_Labels, query_label(l,:)   , 'rows'); 
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

