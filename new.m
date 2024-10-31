clear;
clc;
addpath('./measure');
addpath('./tools');
addpath(genpath('./gspbox-0.7.0'));
%addpath();
resultdir = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

datadir='datasets/';
dataname = {'ORL-3view'};
%dataname = {'bbcsport-4view'};
del=0.1;

ResBest=[];
ResStd=[];
for idata=1:length(dataname)
    load([char(datadir),char(dataname(idata))]);
    
        for perMising=1:length(del)
        tic;
        datafolds=[char(datadir),char(dataname(idata)),'_per',num2str(del(perMising)),'.mat'];
        load(datafolds);
        
        Tlable=Y;
        numclass= length(unique(Tlable));
        k= numclass;
        %lamda1=[10e-5,10e-3,10e-1,10e1,10e3,10e5];
        %lamda2=[10e-5,10e-3,10e-1,10e1,10e3,10e5];
        %lamda1=[0.01,0.02,0.05,0.07,0.09,0.1,0.2,0.5,0.7,0.9,1];
        lamda1=[2e-15,2e-13,2e-11,2e-9,2e-7,2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11,2e13,2e15];
        lamda2=[2e-15,2e-13,2e-11,2e-9,2e-7,2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11,2e13,2e15];
        N=size(X{1},2);
        numview=length(X);
        fold=folds;
        X1=cell(length(X),1);

        for i=1:length(lamda1)
            for j=1:length(lamda2)
                for iv=1:length(X)
                    index{iv} = find(fold(:,iv) == 1);
                    %missingindex{iv} = find(fold(:,iv) == 0);
                    X1{iv} = NormalizeFea(X{iv},0);%归一化X{iv}的每一列；
                    ind_0 = find(fold(:,iv) == 0);%folds的第f列元素为0的指示；
                    X1{iv}(:,ind_0) = 0;%去掉X1{iv}的缺失元；
                end
                disp([char(dataname(idata)),'_per',num2str(del(perMising)), ' lamda1=', num2str(lamda1(i)), ' lamda2=', num2str(lamda2(j))]);
                [S,U,Z,iter]=Paper2(X,k,Tlable,index,lamda1(i),lamda2(j));
                %plot(obj)
                MAXiter = 1000;
                REPlic = 20;
                res=zeros(REPlic,8);
        
                for rep = 1 : 20
                    pY = SpectralClustering(S,k);
                    %pY = kmeans(F, numclass, 'maxiter', MAXiter, 'replicates', REPlic, 'EmptyAction', 'singleton');
                    res(rep, : ) = Clustering8Measure(Tlable, pY);
                    %result = [Fscore Precision Recall nmi AR Entropy ACC Purity];
                end
        
                time=toc;
                disp(['runtime:', num2str(time)]);
                tmpResBest=mean(res); %聚类评价指标的平均值 1x8
                tmpResStd=std(res);%聚类评价指标的标准差 1X8
                ResBest=[ResBest;lamda1(i),lamda2(j),tmpResBest];
                ResStd=[ResStd;lamda1(i),lamda2(j),tmpResStd];
            end
        end
                save([resultdir, char(dataname(idata)),'_per',num2str(del(perMising)),'_result.mat'], "ResBest","ResStd");
    end
end