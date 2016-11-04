%--------------------------------------------------------------------------
% 16/10/10
% QDF classifier(with PCA method) to classify MNIST database
% 在此，让所有类的维数都降到同一个值(本程序中为150)
% 若让不同类降到不同的维数值(但pca都累计到95%)，实验结果正确率仅为79%。由此猜测，若每类选取的主成分个数不同(降的维数不同)，会导致协方差矩阵和均值、特征向量的维数差的比较多，判别函数gi(x)得到的值不公正，结果不准确。(个人猜测)
%--------------------------------------------------------------------------
% 原始给定数据的变量用下划线命名法，其他变量均用驼峰命名法。

%% load data
clear;
load('MNIST-train-images.mat');     %train_images
load('MNIST-train-labels.mat');     %train_labels
load('MNIST-test10k-images.mat');       %test_images
load('MNIST-test10k-labels.mat');       %test_labels

%% get indices of each catagory
indexMat=cell(10,1);  %初始化index matrix，用于存储每一类的索引值
for i=1:9
    indexMat{i}=find(train_labels==i);
end
indexMat{10}=find(train_labels==0);
numOfPC=150;

%% calculate covariance matrix and its inversion of each catagory
classesMat = cell(10,1);  %初始化classes matrix cell，用于存储每一类的数据矩阵
covMat = cell(10,1);  %初始化covariance matrix cell，用于存储每一类的协方差矩阵
covMatInv = cell(10,1);
imagesPC = cell(10,1); %用于存储提取主成分后的images信息
tranMatrix = cell(10,1);
BETA=0.01;   %shrinkage的β参数
for i=1:10
    for j=1:size(indexMat{i},1)
       classesMat{i}(:,j)=train_images(:,indexMat{i}(j));    %将某一类的数据提取出来，分别存入classesMat中
    end
    [coeff, score, latent]=pca(classesMat{i}');
    tranMatrix{i} = coeff(:,1:numOfPC);  %主成分变换矩阵
    imagesPC{i}=(classesMat{i}'*tranMatrix{i})';
    covMat{i}=cov(imagesPC{i}');
    covMatInv{i}=inv(covMat{i});
%shrinkage和伪逆方法，这种情况下ln(det(covMat{i}))太小，不适合
%     I=eye(numOfPC); 
%     covMat{i}=(cov(classesMat{i}')).*(1-BETA)+I.*BETA;
%     covMatInv{i}=inv(covMat{i});    %使用shrinkage
%     %    covMatInv{i}=pinv(covMat{i});     %使用伪逆
end

%% obtain mean value of each category
meanValue=zeros(numOfPC,10);     %初始化存储μi的矩阵
for i=1:10
    meanValue(:,i)=mean(imagesPC{i},2);
end
%% calculate parameters of discriminant function
W=cell(10,1);   %初始化Wi参数(matrices in cell)
for i=1:10
    W{i}=-0.5*covMatInv{i};
end
w=zeros(numOfPC,10);   %初始化w参数(vectors in matrix)
for i=1:10
    w(:,i)=covMatInv{i}*meanValue(:,i);
end
w0=zeros(10,1);    %初始化w0参数向量
for i=1:10
    w0(i)=(-0.5)*(meanValue(:,i)'*covMatInv{i}*meanValue(:,i))-0.5*log(det(covMat{i}));  %认为先验概率相同，于是在此忽略先验概率项
end

%% classification
count=0;    %计算判断正确的个数
numTest=10000;
for i=1:numTest

    g=-inf(10,1);   %初始化判别函数值为-Inf
    for j=1:10      %计算10类中每一类的判别函数的值
        x=(test_images(:,i)'*tranMatrix{j})';
        g(j)=x'*W{j}*x+w(:,j)'*x+w0(j);
    end
    label(i)=find(g==max(g));
    if label(i)==test_labels(i) || (label(i)==10 && test_labels(i)==0)
        count=count+1;
    end
end
disp('Accuracy: ');disp(count/numTest);