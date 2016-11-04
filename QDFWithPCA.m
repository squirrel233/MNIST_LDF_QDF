%--------------------------------------------------------------------------
% 16/10/10
% QDF classifier(with PCA method) to classify MNIST database
% �ڴˣ����������ά��������ͬһ��ֵ(��������Ϊ150)
% ���ò�ͬ�ཱུ����ͬ��ά��ֵ(��pca���ۼƵ�95%)��ʵ������ȷ�ʽ�Ϊ79%���ɴ˲²⣬��ÿ��ѡȡ�����ɷָ�����ͬ(����ά����ͬ)���ᵼ��Э�������;�ֵ������������ά����ıȽ϶࣬�б���gi(x)�õ���ֵ�������������׼ȷ��(���˲²�)
%--------------------------------------------------------------------------
% ԭʼ�������ݵı������»������������������������շ���������

%% load data
clear;
load('MNIST-train-images.mat');     %train_images
load('MNIST-train-labels.mat');     %train_labels
load('MNIST-test10k-images.mat');       %test_images
load('MNIST-test10k-labels.mat');       %test_labels

%% get indices of each catagory
indexMat=cell(10,1);  %��ʼ��index matrix�����ڴ洢ÿһ�������ֵ
for i=1:9
    indexMat{i}=find(train_labels==i);
end
indexMat{10}=find(train_labels==0);
numOfPC=150;

%% calculate covariance matrix and its inversion of each catagory
classesMat = cell(10,1);  %��ʼ��classes matrix cell�����ڴ洢ÿһ������ݾ���
covMat = cell(10,1);  %��ʼ��covariance matrix cell�����ڴ洢ÿһ���Э�������
covMatInv = cell(10,1);
imagesPC = cell(10,1); %���ڴ洢��ȡ���ɷֺ��images��Ϣ
tranMatrix = cell(10,1);
BETA=0.01;   %shrinkage�Ħ²���
for i=1:10
    for j=1:size(indexMat{i},1)
       classesMat{i}(:,j)=train_images(:,indexMat{i}(j));    %��ĳһ���������ȡ�������ֱ����classesMat��
    end
    [coeff, score, latent]=pca(classesMat{i}');
    tranMatrix{i} = coeff(:,1:numOfPC);  %���ɷֱ任����
    imagesPC{i}=(classesMat{i}'*tranMatrix{i})';
    covMat{i}=cov(imagesPC{i}');
    covMatInv{i}=inv(covMat{i});
%shrinkage��α�淽�������������ln(det(covMat{i}))̫С�����ʺ�
%     I=eye(numOfPC); 
%     covMat{i}=(cov(classesMat{i}')).*(1-BETA)+I.*BETA;
%     covMatInv{i}=inv(covMat{i});    %ʹ��shrinkage
%     %    covMatInv{i}=pinv(covMat{i});     %ʹ��α��
end

%% obtain mean value of each category
meanValue=zeros(numOfPC,10);     %��ʼ���洢��i�ľ���
for i=1:10
    meanValue(:,i)=mean(imagesPC{i},2);
end
%% calculate parameters of discriminant function
W=cell(10,1);   %��ʼ��Wi����(matrices in cell)
for i=1:10
    W{i}=-0.5*covMatInv{i};
end
w=zeros(numOfPC,10);   %��ʼ��w����(vectors in matrix)
for i=1:10
    w(:,i)=covMatInv{i}*meanValue(:,i);
end
w0=zeros(10,1);    %��ʼ��w0��������
for i=1:10
    w0(i)=(-0.5)*(meanValue(:,i)'*covMatInv{i}*meanValue(:,i))-0.5*log(det(covMat{i}));  %��Ϊ���������ͬ�������ڴ˺������������
end

%% classification
count=0;    %�����ж���ȷ�ĸ���
numTest=10000;
for i=1:numTest

    g=-inf(10,1);   %��ʼ���б���ֵΪ-Inf
    for j=1:10      %����10����ÿһ����б�����ֵ
        x=(test_images(:,i)'*tranMatrix{j})';
        g(j)=x'*W{j}*x+w(:,j)'*x+w0(j);
    end
    label(i)=find(g==max(g));
    if label(i)==test_labels(i) || (label(i)==10 && test_labels(i)==0)
        count=count+1;
    end
end
disp('Accuracy: ');disp(count/numTest);