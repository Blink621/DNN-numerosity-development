%% raw data processing: pca
clc; clear;

load('fc1_data.mat')
load('fc1_test.mat')
fc1_data = [X_data; X_test];
d_act = double(fc1_data);
[coeff, score, latent, tsquare, explained, mu] = pca(d_act);
n_component = sum(cumsum(explained) < 95) + 1;
[coeff, score, latent, tsquare, explained, mu] = pca(d_act, 'NumComponents', n_component);
disp(n_component);
d_act_pca = score(1:57600, :);
d_num = repmat(repelem((1 : 32).', 600), 7, 1);
d_set = repelem((1 : 7).', 600 * 32);
d_index = repmat(repmat((1:600).', 32, 1), 7, 1);
data_all = [d_set, d_num, d_index, d_act_pca];

clearvars -except data
id_train = [1:10, 601:610, 1201:1210, 1801:1810];

%% 
data = data((data(:, 1) == 1) || (data(:, 1) == 2) || (data(:, 1) == 3));
index_list = 0;
for sti_std = 1:32
    for sti_com = 1:(sti_std-1)
        index_list = index_list + 1;
        data1 = data(data(:, 2) == sti_std, :);
        data1 = data1(:, 4:end);
        data1 = data1(id_train, :);
        data2 = data(data(:, 2) == sti_com, :);
        data2 = data2(:, 4:end);
        data2 = data2(id_train, :);
        data_std_com = data1 - data2;
        data_com_std = data2 - data1;
        X = [data_std_com; data_com_std];
        y = repelem([1, -1], 4 * 10);
        y = y.';
        Xlist{index_list, 1} = X;
        ylist{index_list, 1} = y;
    end
end

Xlist = cell2mat(Xlist);
ylist = cell2mat(ylist);
Model = fitcsvm(Xlist, ylist, 'Standardize', true, 'KernelFunction', 'linear', 'KernelScale', 'auto');
CVModel = crossval(Model);
classLoss = kfoldLoss(CVModel);

%% 
clearvars -except data Model CVModel classLoss
id_train = [501:600, 1101:1200, 1701:1800, 2301:2400];
index_list = 0;
for sti_std = 1:32
    for sti_com = 1:sti_std
        index_list = index_list + 1;
        data1 = data(data(:, 2) == sti_std, :);
        data1 = data1(:, 4:end);
        data1 = data1(id_train, :);
        data2 = data(data(:, 2) == sti_com, :);
        data2 = data2(:, 4:end);
        data2 = data2(id_train, :);
        data_std_com = data1 - data2;
        data_com_std = data2 - data1;
        X = [data_std_com; data_com_std];
        y = repelem([1, -1], 4 * 100);
        y = y.';
        Xlist{index_list, 1} = X;
        ylist{index_list, 1} = y;
    end
end
Xlist = cell2mat(Xlist);
ylist = cell2mat(ylist);
ypre = Model.predict(Xlist);

%%
acc_M_1 = NaN(32, 32);
index_com = 0;
for sti_std = 1:32
    for sti_com = 1:sti_std
        index_com = index_com + 1;
        acc_M_1(sti_std, sti_com) = sum(ylist(((index_com - 1) * 800 + 1):((index_com) * 800)) == ypre(((index_com - 1) * 800 + 1):((index_com) * 800))) / 800;
    end
end

%% 
data = data((data(:, 1) == 4) || (data(:, 1) == 5) || (data(:, 1) == 6) || (data(:, 1) == 7));

clearvars -except data Model CVModel classLoss
% id_train = [501:600, 1101:1200, 1701:1800, 2301:2400];
index_list = 0;
for sti_std = 1:32
    for sti_com = 1:sti_std
        index_list = index_list + 1;
        data1 = data(data(:, 2) == sti_std, :);
        data1 = data1(:, 4:end);
%         data1 = data1(id_train, :);
        data2 = data(data(:, 2) == sti_com, :);
        data2 = data2(:, 4:end);
%         data2 = data2(id_train, :);
        data_std_com = data1 - data2;
        data_com_std = data2 - data1;
        X = [data_std_com; data_com_std];
        y = repelem([1, -1], 4 * 100);
        y = y.';
        Xlist{index_list, 1} = X;
        ylist{index_list, 1} = y;
    end
end
Xlist = cell2mat(Xlist);
ylist = cell2mat(ylist);
ypre = Model.predict(Xlist);

%%
acc_M_2 = NaN(32, 32);
index_com = 0;
for sti_std = 1:32
    for sti_com = 1:sti_std
        index_com = index_com + 1;
        acc_M_2(sti_std, sti_com) = sum(ylist(((index_com - 1) * 800 + 1):((index_com) * 800)) == ypre(((index_com - 1) * 800 + 1):((index_com) * 800))) / 800;
    end
end