function [CVMdl,oosLoss]=SVM3()    %分成三类
load fisheriris
X = meas;        %150*4 ：150个样本，4个特征（萼长、萼宽、瓣长、瓣宽）；meas=measure（长度）
Y = species;     %三种属性{'setosa','versicolor','virginica'}；species（种类）
t = templateSVM('Standardize',1); %创建SVM模板t；%templateSVM是fitcecoc函数中的SVM模板；%standardize:数据标准化，可用help查看templateSVM其他参数
%训练该模型
Mdl = fitcecoc(X,Y,'Learners',t,'ClassNames',{'setosa','versicolor','virginica'});   
%验证该模型
CVMdl = crossval(Mdl);  %将模型进行交叉验证，平衡模型欠拟合和过拟合
%显示结果
oosLoss = kfoldLoss(CVMdl)  %10折交叉验证得到的泛化误差 oosloss =0.033，效果很好
