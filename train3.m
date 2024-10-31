function [S,U,Z,iter]=train3(X,Tlable,index,lamda1,lamda2)
%% initialize
maxIter=100;

%numclass=length(unique(Tlable));
numview=length(X);
numsample=size(Tlable,1);
d=zeros(1,numview);
for i=1:numview
    d(i)=size(X{i},1);
end

%构造矩阵A
[A,np]=constructA(X,index);

%初始化Z0，每一片为单位矩阵
Z0=cell(numview,1);
for i=1:numview
    Z0{i}=eye(np(i));
end

%初始化E0，每一片为单位矩阵
E0=cell(numview,1);
for i=1:numview
    E0{i}=zeros(d(i),np(i));
end

%初始化Z，每一片为单位矩阵
Z=cell(numview,1);
for i=1:numview
    Z{i}=eye(numsample,numsample);
end


%初始化E，每一片为单位矩阵
E=cell(numview,1);
for i=1:numview
    E{i}=zeros(d(i),numsample);
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

%初始化Y
Y=cell(numview,1);

for i=1:numview
    Y{i}=zeros(d(i),np(i));
end


%初始化mu
mu1 = 10e-5; max_mu1 = 10e10; pho_mu1 = 2;
mu2 = 10e-5; max_mu2 = 10e10; pho_mu2 = 2;


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

    beta=zeros(1,numview);
    for iz=1:numview
        beta(iz)=1/(2*norm(Z0{iz}-A{iz}'*U*A{iz},'fro')+eps);

        Q1{iz}=Q{iz}-A{iz}*A{iz}'*Q{iz}*A{iz}*A{iz}';
        M1{iz}=M{iz}-A{iz}*A{iz}'*M{iz}*A{iz}*A{iz}';

        Z0{iz}=((2*beta(iz)+mu1)*eye(np(iz))+mu2*A{iz}'*X{iz}'*X{iz}*A{iz})\(A{iz}'*(2*beta(iz)*U+mu1*Q{iz}+M{iz}+mu2*X{iz}'*X{iz})*A{iz}+A{iz}'*X{iz}'*(Y{iz}-E0{iz}));
        %Z0{iz}=(A{iz}'*(2*lamda1*X{iz}'*X{iz}+(2*beta(iz)+mu+pho)*eye(numsample))*A{iz})\(A{iz}'*(2*lamda1*X{iz}'*X{iz}+2*beta(iz)*U+mu*Q{iz}-M{iz}+pho*G{iz}-W{iz})*A{iz});
        
    end
    
    %%Update U
    sumU=0; 
    
    for iu=1:numview
        
        %Update U1p
        %U1p=(2*beta(iu)*Z0{iu}+A{iu}'*Y*A{iu})/(2*beta(iu)+mu++pho);
        U1p=Z0{iu};
        %Update U2p
        U2p=Q1{iu}+1/mu1*M1{iu};
        
        Up=A{iu}*U1p*A{iu}'+U2p;
        sumU=sumU+Up;
    end
    U=sumU/numview;
    
    %%Update Zp and tensor_Z
    
    for ii=1:numview
        Z{ii}=U+A{ii}*(Z0{ii}-A{ii}'*U*A{ii})*A{ii}';
        
    end
    
    %%Update E;
    F = [];
    for k=1:numview    
        tmp = X{k}*A{k}-X{k}*A{k}*Z0{k}+Y{k}/mu2;
        F = [F;tmp*A{k}'];
    end
    %F = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu];
    [Econcat] = solve_l1l2(F,lamda1/mu2);
    %[Econcat,info] = prox_l21(F, 0.5/1);
    start = 1;
    for k=1:numview
        E{k} = Econcat(start:start + d(k) - 1,:);
        E0{k} = E{k}*A{k};
        start = start + d(k);
    end

       
    %%Update Qp
    for iq=1:numview
        Q{iq}=(mu1*Z{iq}-M{iq})/(2*lamda2*L{iq}+mu1*eye(numsample));
    end
   
    
    %%Update Mp
    for im=1:numview
        M{im}=M{im}+mu1*(Q{im}-Z{im});
    end

    %%Update Yp
    for iy=1:numview
        Y{iy}=Y{iy}+mu2*(X{iy}*A{iy}-X{iy}*A{iy}*Z0{iy}-E0{iy});
    end
    
    %%Update mu
    mu1 = min(mu1*pho_mu1, max_mu1);
    mu2 = min(mu2*pho_mu2, max_mu2);   
    
    
    term1=0;
    term2=0;

    for i=1:numview
        term1=term1+beta(i)*norm(Z0{i}-A{i}'*U*A{i},"fro")^2;
        term2=term2+lamda2*trace(Z{i}*L{i}*Z{i}');
    end
    obj=[];
    obj(iter)=term1+term2;
    
   
    %obj(iter)=term1+term2;
    %if(iter>1)&&(iter>maxIter||term1<epson||term2<epson)
    if(iter>1)&&(iter>maxIter||abs(obj(iter-1)-obj(iter))/obj(iter-1)<1e-3)
         S = abs(U)+abs(U');
         %[UU,~,V]=svd(U','econ');
         %UU= UU(:,1:k);
        flag=0;
    end
end
