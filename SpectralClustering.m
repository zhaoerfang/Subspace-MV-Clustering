%--------------------------------------------------------------------------
% This function takes an adjacency matrix of a graph and computes the 
% clustering of the nodes using the spectral clustering algorithm of 
% Ng, Jordan and Weiss.
% CMat: NxN adjacency matrix
% n: number of groups for clustering
% groups: N-dimensional vector containing the memberships of the N points 
% to the n groups obtained by spectral clustering
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function groups = SpectralClustering(CKSym,n)

warning off;
N = size(CKSym,1); %获取矩阵的大小，即节点的数量
MAXiter = 1000; % Maximum number of iterations for KMeans 
REPlic = 20; % Number of replications for KMeans

% Normalized spectral clustering according to Ng & Jordan & Weiss
% using Normalized Symmetric Laplacian L = I - D^{-1/2} W D^{-1/2}

DN = diag( 1./sqrt(sum(CKSym)+eps) ); %计算度矩阵的倒数平方根
LapN = speye(N) - DN * CKSym * DN; %计算标准化的拉普拉斯矩阵
[uN,sN,vN] = svd(LapN); %SVD
kerN = vN(:,N-n+1:N); %提取前n个特征向量
for i = 1:N
    kerNS(i,:) = kerN(i,:) ./ norm(kerN(i,:)+eps); %对特征向量归一化
end
groups = kmeans(kerNS,n,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');%使用kmeans对特征向量进行聚类