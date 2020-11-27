clc; clear;
load svm_example_data

SVMModel = fitcsvm(total_data, total_label, ...
    'Standardize', true, 'KernelFunction', 'linear', 'KernelScale', 'auto');
CV = crossval(SVMModel);
classLossL = kfoldLoss(SVMModel);
