clc; clear;
% linear SVM
train_index = repmat(repmat((1:600).', 32, 1), 4, 1);

load('SVM_fc1_relu_test_simple.mat')
X = double(fc1_data);
X_train = X(train_index < 501);
y = repmat(repelem((1 : 32).', 600), 4, 1);
y = y(train_index < 501);

t = templateSVM('Standardize', true, 'KernelFunction', 'linear', 'KernelScale', 'auto');
Model_1 = fitcecoc(X_train, y, 'Learners', t, 'ClassNames',{'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'});
CVMdl_1 = crossval(Model_1);
Loss_1 = kfoldLoss(CVMdl_1);

X_test = X(train_index > 500);
Model_1_test = Model_1.predict(X_test);

load('SVM_fc2_relu_test_simple.mat')
X = double(fc1_data);
X_train = X(train_index < 501);
y = repmat(repelem((1 : 32).', 600), 4, 1);
y = y(train_index < 501);

t = templateSVM('Standardize', true, 'KernelFunction', 'linear', 'KernelScale', 'auto');
Model_2 = fitcecoc(X_train, y, 'Learners', t, 'ClassNames',{'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32'});
CVMdl_2 = crossval(Model_2);
Loss_2 = kfoldLoss(CVMdl_2);

X_test = X(train_index > 500);
Model_2_test = Model_2.predict(X_test);

save('SVM_linear.mat')