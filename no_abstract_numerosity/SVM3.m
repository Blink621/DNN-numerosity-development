function [CVMdl,oosLoss]=SVM3()    %�ֳ�����
load fisheriris
X = meas;        %150*4 ��150��������4���������೤������곤�������meas=measure�����ȣ�
Y = species;     %��������{'setosa','versicolor','virginica'}��species�����ࣩ
t = templateSVM('Standardize',1); %����SVMģ��t��%templateSVM��fitcecoc�����е�SVMģ�壻%standardize:���ݱ�׼��������help�鿴templateSVM��������
%ѵ����ģ��
Mdl = fitcecoc(X,Y,'Learners',t,'ClassNames',{'setosa','versicolor','virginica'});   
%��֤��ģ��
CVMdl = crossval(Mdl);  %��ģ�ͽ��н�����֤��ƽ��ģ��Ƿ��Ϻ͹����
%��ʾ���
oosLoss = kfoldLoss(CVMdl)  %10�۽�����֤�õ��ķ������ oosloss =0.033��Ч���ܺ�
