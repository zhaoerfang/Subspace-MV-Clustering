clear;clc;
addpath('measure','tools','gspbox-0.7.0');

resultdir = 'Results2/';
if (~exist('Results2', 'file'))
    mkdir('Results2');
    addpath(genpath('Results2/'));
end

datadir='datasets/';
dataname = {'ORL-3view'};
%dataname = {'3sources-3view'};
%dataname = {'bbcsport-4view'};
del=[0.1];
%del=[0.1,0.3,0.5,0.7];

ResBest=[];
ResStd=[];

for idata=1:length(dataname)
    load([char(datadir),char(dataname(idata))]);

    for perMising=1:length(del)
        datafolds=[char(datadir),char(dataname(idata)),'_per',num2str(del(perMising))];
        load(datafolds);
        
        Tlable=Y;
        numclass= length(unique(Tlable));
        k= numclass;
        
        %lamda1=[1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3,1e4,1e5];
        %lamda2=[10e-5,10e-3,10e-1,10e1,10e3,10e5];
%         lamda1=[0.01,0.02,0.05,0.07,0.09,0.1,0.2,0.5,0.7,0.9,1];
%         lamda2=[0.01,0.02,0.05,0.07,0.09,0.1,0.2,0.5,0.7,0.9,1];
        %lamda1=[2e-9];
        %lamda2=[2e15];
        %lamda1=[2e-15,2e-13,2e-11,2e-9,2e-7,2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11,2e13,2e15];
        %lamda2=[2e-15,2e-13,2e-11,2e-9,2e-7,2e-5,2e-3,2e-1,2e1,2e3,2e5,2e7,2e9,2e11,2e13,2e15];
        lamda1=[1e-15,1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1,1e3,1e5,1e7,1e9,1e11];
        lamda2=[1e-15,1e-13,1e-11,1e-9,1e-7,1e-5,1e-3,1e-1,1e1,1e3,1e5,1e7,1e9,1e11];
        N=size(X{1},2);
        numview=length(X);
        fold=folds;
        X1=cell(length(X),1);
        index=cell(length(X),1);

        for i=1:length(lamda1)
            for j=1:length(lamda2)
                for iv=1:length(X)

                    index{iv} = find(fold{1}(:,iv) == 1);
                    %missingindex{iv} = find(fold(:,iv) == 0);
                    X1{iv} = NormalizeData(X{iv});
                    ind_0 = find(fold{1}(:,iv) == 0);%folds的第f列元素为0的指示；
                    X1{iv}(:,ind_0) = 0;%去掉X1{iv}的缺失元；
                end
                disp([char(dataname(idata)),'_per',num2str(del(perMising)), ' lamda1=', num2str(lamda1(i))]);
                    
                %% train
                    
                % Initialize...                    
                numsample=size(Tlable,1);
                %构造矩阵A
                [A,np]=constructA(X,index);

                sx=[numsample,numsample,numview];

                %初始化Z0，Z，Q，每一片为单位矩阵
                Z0=cell(numview,1);
                Z=cell(numview,1);
                ZpU=cell(numview,1);
                Q=cell(numview,1);
                Q1=cell(numview,1);
                M=cell(numview,1);
                M1=cell(numview,1);
                tensor_G=zeros(sx);
                G=cell(numview,1);
                G0=cell(numview,1);
                G1=cell(numview,1);

                tensor_W=zeros(sx);
                W=cell(numview,1);
                W0=cell(numview,1);
                W1=cell(numview,1);

                Y=cell(numview,1);

                tensor_Z=zeros(sx);
                for ii=1:numview
                    Z0{ii}=eye(np(ii));
                    Z{ii}=eye(numsample,numsample);
                    Q{ii}=zeros(numsample,numsample);
                    M{ii}=zeros(numsample,numsample);
                    G{ii}=tensor_G(:,:,ii);
                    W{ii}=tensor_W(:,:,ii);
                end

                beta=zeros(1,numview);

                %初始化一致表示U
                U=zeros(numsample,numsample);

                %初始化pho,mu
                pho = 10e-5; max_pho = 10e10; pho_pho = 2;
                mu = 10e-5; max_mu = 10e10; pho_mu = 2;
                    

                iter=0;
                start = 1;
                tic;
                epson = 1e-7;
                Isconverg = 0;

                while (Isconverg == 0)
                    iter=iter+1;
                    fprintf('----processing iter %d--------\n', iter);
                        
                    %-------------------0 update L^k-------------------------------
                    for il=1:numview
                        if start==1
                            Weight{il} = constructW_PKN((abs(Z{il})+abs(Z{il}'))./2, 3);
                            Diag_tmp = diag(sum(Weight{il}));
                            L{il} = Diag_tmp - Weight{il};
                        else
                        % modified to hyper-graph
                        P =  (abs(Z{il})+abs(Z{il}'))./2;
                        param.k = 3;
                        HG = gsp_nn_hypergraph(P', param);
                        L{il} = HG.L;
                        end
                    end
                    start = 0;
    
                    %-------------------1 update Z0{i}-------------------------------
                    for iz=1:numview
                        beta(iz)=1/(2*norm(Z0{iz}-A{iz}'*U*A{iz},'fro')+eps);

                        Q1{iz}=Q{iz}-A{iz}*A{iz}'*Q{iz}*A{iz}*A{iz}';
                        M1{iz}=M{iz}-A{iz}*A{iz}'*M{iz}*A{iz}*A{iz}';
                        G1{iz}=G{iz}-A{iz}*A{iz}'*G{iz}*A{iz}*A{iz}';
                        W1{iz}=W{iz}-A{iz}*A{iz}'*W{iz}*A{iz}*A{iz}';
                        Y{iz}=(mu*Q1{iz}-M1{iz}+pho*G1{iz}-W1{iz});
                           
                        Z0{iz}=(2*lamda1(i)*A{iz}'*X{iz}'*X{iz}*A{iz}+(2*beta(iz)+mu+pho)*eye(np(iz)))\(A{iz}'*(2*lamda1(i)*X{iz}'*X{iz}+2*beta(iz)*U+mu*Q{iz}-M{iz}+pho*G{iz}-W{iz})*A{iz});
                    end
                    %-------------------2 update U--------------------------------
                    sumU=0; 
                    for iu=1:numview
                        %Update U1p
                        U1p=Z0{iu};

                        %Update U2p
                        U2p=Y{iu}/(mu+pho);
                        Up=A{iu}*U1p*A{iu}'+U2p;
                        sumU=sumU+Up;
                    end
                    U=sumU/numview;
                    %-------------------3 update tensor_Z--------------------------------
                    for ii=1:numview
                        ZpU{ii}=A{ii}*(Z0{ii}-A{ii}'*U*A{ii})*A{ii}'+U;
                        tensor_Z(:,:,ii)=ZpU{ii};
                    end
                    z=tensor_Z(:);
                    %-------------------4 update Q--------------------------------
                    for iq=1:numview
                        Q{iq}=(mu*Z{iq}+M{iq})/(mu*eye(numsample)-2*lamda2(j)*L{iq});
                    end
                    %-------------------5 update M--------------------------------
                    for im=1:numview
                        M{im}=M{im}+mu*(Z{im}-Q{im});
                    end
                    %-------------------6 update tensor_G--------------------------------
                    w=tensor_W(:);
                    [g, objv] = wshrinkObj(z + 1/pho*w,1/pho,sx,0,1);
                    tensor_G = reshape(g, sx);
                    %-------------------7 update tensor_W--------------------------------
                    w = w + pho*(z - g);
                    tensor_W = reshape(w, sx);

                    %record the iteration information
                    history.objval(iter+1)   =  objv;
                    
                    %-------------------8 update pho,mu--------------------------------
                    pho = min(pho*pho_pho, max_pho);
                    mu = min(mu*pho_mu, max_mu);

                    %% coverge condition
                    Isconverg = 1;
                    for ic=1:numview
                       if (norm(X{ic}*A{ic}-X{ic}*A{ic}*Z0{ic},inf)>epson)
                            history.norm_Z0 = norm(X{ic}*A{ic}-X{ic}*A{ic}*Z0{ic},inf);
                            fprintf('    norm_Z %7.10f    ', history.norm_Z0);
                            Isconverg = 0;
                        end 
        
                        if (norm(Z{ic}-G{ic},inf)>epson)
                            history.norm_Z_G = norm(Z{ic}-G{ic},inf);
                            fprintf('norm_Z_G %7.10f    \n', history.norm_Z_G);
                            Isconverg = 0;
                        end
                    end
                    if (iter>200)
                        Isconverg  = 1;
                    end
                end
                    S1 =1/2*(abs(U)+abs(U'));
                    C1 = SpectralClustering(S1,k);
                    REPlic = 20;
                    res=zeros(REPlic,8);
                    for rep = 1 : 20
                        res(rep, : ) = Clustering8Measure(Tlable, C1);
                        %result = [Fscore Precision Recall nmi AR Entropy ACC Purity];
                    end
                    time=toc;
                    disp(['runtime:', num2str(time)]);
                    tmpResBest=mean(res); %聚类评价指标的平均值 1x8
                    tmpResStd=std(res);%聚类评价指标的标准差 1X8
                    ResBest=[ResBest;lamda1(i),lamda2(j),tmpResBest];
                    ResStd=[ResStd;lamda1(i),lamda2(j),tmpResStd];
                    %ResBest=[ResBest;lamda1(i),tmpResBest];
                    %ResStd=[ResStd;lamda1(i),tmpResStd];
            end
        end
        %save([resultdir, char(dataname(idata)),'_per',num2str(del(perMising)),'_result.mat'], "ResBest","ResStd");

    end

end
