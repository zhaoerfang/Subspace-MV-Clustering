function [S,U,Z,iter]=train1(X,k,Tlable,index,lamda1,lamda2)
%% initialize
maxIter=100;

%numclass=length(unique(Tlable));
numview=length(X);
numsample=size(Tlable,1);

%构造矩阵A
[A,np]=constructA(X,index);

%初始化Z0，每一片为单位矩阵
Z0=cell(numview,1);
for i=1:numview
    Z0{i}=eye(np(i));
end

%初始化Z，每一片为单位矩阵
Z=cell(numview,1);
for i=1:numview
    Z{i}=eye(numsample,numsample);
end

%初始化一致表征U
U=zeros(numsample,numsample);

sx=[numsample,numsample,numview];

%初始化Q，每一片为0矩阵,(辅助变量)
Q=cell(numview,1);
Q1=cell(numview,1);
for i=1:numview
    Q{i}=zeros(numsample,numsample);
end

%初始化M，每一片为0矩阵,(拉格朗日乘子)
M=cell(numview,1);
M1=cell(numview,1);
for i=1:numview
    M{i}=zeros(numsample,numsample);
end

%初始化mu
mu = 10e-5; max_mu = 10e10; pho_mu = 2;


%初始化张量tensor_G,(辅助变量);
tensor_G=zeros(sx);
%G=zeros(1,numsample*numview*numsample);
%tensor_G=reshape(G,[numsample,numsample,numview]);
G=cell(numview,1);
G1=cell(numview,1);
for i=1:numview
    G{i}=tensor_G(:,:,i);
end

%初始化张量tensor_W,(辅助变量);
tensor_W=zeros(sx);
%G=zeros(1,numsample*numview*numsample);
%tensor_G=reshape(G,[numsample,numsample,numview]);
W=cell(numview,1);
W1=cell(numview,1);
for i=1:numview
    W{i}=tensor_G(:,:,i);
end

%初始化pho
pho = 10e-5; max_pho = 10e10; pho_pho = 2;

flag=1;
iter=0;
start = 1;
epson = 1e-7;

while flag
    iter=iter+1;
    
    for il=1:numview
        if start==1
          Weight{il} = constructW_PKN((abs(Z{il})+abs(Z{il}'))./2, 3);
          Diag_tmp = diag(sum(Weight{il}));
          L{il} = Diag_tmp - Weight{il};
        else
        %------------modified to hyper-graph---------------
          P =  (abs(Z{il})+abs(Z{il}'))./2;
          param.k = 3;
          HG = gsp_nn_hypergraph(P', param);
          L{il} = HG.L;
        end
        
%         Weight{k} = constructW_PKN((abs(Z{k})+abs(Z{k}'))./2, 10);
%         Diag_tmp = diag(sum(Weight{k}));
%         L{k} = Diag_tmp - Weight{k};
    end
    start = 0;
        
    %%Update Z0{i},每个view不完整表征，方阵;
    Y=cell(numview,1);
    beta=zeros(1,numview);
    for iz=1:numview
        beta(iz)=1/(2*norm(Z0{iz}-A{iz}'*U*A{iz},'fro')+eps);

        Q1{iz}=Q{iz}-A{iz}*A{iz}'*Q{iz}*A{iz}*A{iz}';
        M1{iz}=M{iz}-A{iz}*A{iz}'*M{iz}*A{iz}*A{iz}';
        G1{iz}=G{iz}-A{iz}*A{iz}'*G{iz}*A{iz}*A{iz}';
        W1{iz}=W{iz}-A{iz}*A{iz}'*W{iz}*A{iz}*A{iz}';

        Y{iz}=(mu*Q1{iz}-M1{iz}+pho*G1{iz}-W1{iz});
        Z0{iz}=(A{iz}'*(2*lamda1*X{iz}'*X{iz}+(2*beta(iz)+mu+pho)*eye(numsample))*A{iz})\(A{iz}'*(2*lamda1*X{iz}'*X{iz}+2*beta(iz)*U+mu*Q{iz}-M{iz}+pho*G{iz}-W{iz})*A{iz});
        %Z0{iz}=(2*lamda1*A{iz}'*X{iz}'*X{iz}*A{iz}+(2*beta(iz)+mu+pho)*eye(np(iz)))\(A{iz}'*(2*lamda1*X{iz}'*X{iz}+2*beta(iz)*U+mu*Q{iz}-M{iz}+pho*G{iz}-W{iz})*A{iz});
    end
    
    %%Update U
    sumU=0; 
    
    for iu=1:numview
        
        %Update U1p
        %U1p=(2*beta(iu)*Z0{iu}+A{iu}'*Y*A{iu})/(2*beta(iu)+mu++pho);
        U1p=Z0{iu};
        %Update U2p
        U2p=Y{iu}/(mu+pho);
        
        Up=A{iu}*U1p*A{iu}'+U2p;
        sumU=sumU+Up;
    end
    U=sumU/numview;
    
    %% Update Zp and tensor_Z
    tensor_Z=zeros(sx);
    for ii=1:numview
        Z{ii}=U+A{ii}*(Z0{ii}-A{ii}'*U*A{ii})*A{ii}';
        tensor_Z(:,:,ii)=Z{ii};
    end
    
       
    %% Update Qp
    for iq=1:numview
        Q{iq}=(mu*Z{iq}+M{iq})/(mu*eye(numsample)-2*lamda2*L{iq});
    end
    
    z=tensor_Z(:);
    w=tensor_W(:);
    [g, objv] = wshrinkObj(z + 1/pho*w,1/pho,sx,0,1);
    tensor_G = reshape(g, sx);
    
    %%Update Mp
    for im=1:numview
        M{im}=M{im}+mu*(Z{im}-Q{im});
    end
    
    %%Update tensor_W
    tensor_W=tensor_W+pho*(tensor_Z-tensor_G);
    
    %%Update mu
    mu = min(mu*pho_mu, max_mu);
    
    %%Update pho
    pho = min(pho*pho_pho, max_pho);    
    
    
    term1=0;
    term2=0;
    for i=1:numview
        term1=term1+norm(Z{i}-Q{i},"fro")^2;
        term2=term2+norm(Z{i}-G{i},"fro")^2;
    end
    term1=term1/numview;
    term2=term2/numview;
    
   
    %obj(iter)=term1+term2;
    if(iter>1)&&(iter>maxIter||term1<epson||term2<epson)
    %if(iter>1)&&(iter>maxIter||abs(obj(iter-1)-obj(iter))/obj(iter-1)<1e-3)
         %S = abs(U)+abs(U');
         %[UU,~,V]=svd(U','econ');
         %UU= UU(:,1:k);
        flag=0;
    end
end
