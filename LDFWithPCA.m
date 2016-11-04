%--------------------------------------------------------------------------
% 16/10/9
% LDF classifier(with PCA method) to classify MNIST database
%--------------------------------------------------------------------------
%原始给定数据的变量用下划线命名法，其他变量均用驼峰命名法。

%% load data
clear;
load('MNIST-train-images.mat');
load('MNIST-train-labels.mat');
load('MNIST-test10k-images.mat');
load('MNIST-test10k-labels.mat');

%% calculate public covariance matrix and its inversion
[coeff, score, latent]=pca(train_images');
accum=cumsum(latent)./sum(latent);   %记录累积值
numOfPC=find(accum>0.95,1,'first');    %寻找累积至95%的最小特征维数，在这里为154
tranMatrix = coeff(:,1:numOfPC);  %主成分变换矩阵
imagesPC=(train_images'*tranMatrix)';
covMat=cov(imagesPC');
covMatInv=inv(covMat);
%% obtain mean value of each category
indexMat=cell(10,1);  %初始化index matrix，用于存储每一类的索引值
mean=zeros(numOfPC,10);     %初始化存储μi的矩阵
for i=1:9
   indexMat{i}=find(train_labels==i);
   sum=0;
   for j=1:size(indexMat{i},1)
       sum=sum+imagesPC(:,indexMat{i}(j));
   end
   mean(:,i)=sum/size(indexMat{i},1);
end
%对0特殊处理
indexMat{10}=find(train_labels==0);
sum=0;
for j=1:size(indexMat{10},1)
   sum=sum+imagesPC(:,indexMat{10}(j));
end
mean(:,10)=sum/size(indexMat{10},1);
%% calculate parameters of discriminant function
w=zeros(numOfPC,10);   %初始化w参数矩阵
for i=1:10
    w(:,i)=covMatInv*mean(:,i);
end
w0=zeros(10,1);    %初始化w0参数向量
for i=1:10
    w0(i)=(-0.5)*(mean(:,i)'*covMatInv*mean(:,i));	%认为先验概率相同，于是在此忽略先验概率项
end

%% classification
testImagesPC=(test_images'*tranMatrix)';
count=0;    %计算判断正确的个数
numTest=10000;
for i=1:numTest
    x=testImagesPC(:,i);
    g=-inf(10,1);   %初始化判别函数值为-Inf
    for j=1:10      %计算10类中每一类的判别函数的值
        g(j)=w(:,j)'*x+w0(j);
    end
    label(i)=find(g==max(g));
    if label(i)==test_labels(i) || (label(i)==10 && test_labels(i)==0)
        count=count+1;
    end
end
disp('Accuracy: ');disp(count/numTest);