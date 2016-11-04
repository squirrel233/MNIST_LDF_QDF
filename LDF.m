%--------------------------------------------------------------------------
% 16/10/9
% LDF classifier(with shrinkage and pseudo inverse method) to classify MNIST database
%--------------------------------------------------------------------------
%ԭʼ�������ݵı������»������������������������շ���������

%% load data
clear;
load('MNIST-train-images.mat');     %train_images
load('MNIST-train-labels.mat');     %train_labels
load('MNIST-test10k-images.mat');
load('MNIST-test10k-labels.mat');

%% calculate public covariance matrix and its inversion
numOfPC=784;    %��Ϊû�н������ɷַ��������Դ˴����ɷֵ�����Ϊ784
I=eye(numOfPC);      
BETA=0.01;
covMat=(cov(train_images')).*(1-BETA)+I.*BETA;
%covMatInv=pinv(covMat);     %ʹ��α��
covMatInv=inv(covMat);    %ʹ��shrinkage
%% obtain mean value of each category
indexMat=cell(10,1);  %��ʼ��index matrix�����ڴ洢ÿһ�������ֵ
mean=zeros(numOfPC,10);     %��ʼ���洢��i�ľ���
for i=1:9
   indexMat{i}=find(train_labels==i);
   sum=0;
   for j=1:size(indexMat{i},1)
       sum=sum+train_images(:,indexMat{i}(j));
   end
   mean(:,i)=sum/size(indexMat{i},1);
end
%��0���⴦��
indexMat{10}=find(train_labels==0);
sum=0;
for j=1:size(indexMat{10},1)
   sum=sum+train_images(:,indexMat{10}(j));
end
mean(:,10)=sum/size(indexMat{10},1);
%% calculate parameters of discriminant function
w=zeros(numOfPC,10);   %��ʼ��w��������
for i=1:10
    w(:,i)=covMatInv*mean(:,i);
end
w0=zeros(10,1);    %��ʼ��w0��������
for i=1:10
    w0(i)=(-0.5)*(mean(:,i)'*covMatInv*mean(:,i));  %��Ϊ���������ͬ�������ڴ˺������������
end

%% classification
count=0;    %�����ж���ȷ�ĸ���
numTest=10000;
for i=1:numTest
    x=test_images(:,i);
    g=-inf(10,1);   %��ʼ���б���ֵΪ-Inf
    for j=1:10      %����10����ÿһ����б�����ֵ
        g(j)=w(:,j)'*x+w0(j);
    end
    label(i)=find(g==max(g));
    if label(i)==test_labels(i) || (label(i)==10 && test_labels(i)==0)
        count=count+1;
    end
end
disp('Accuracy: ');disp(count/numTest);