function [UU,U,Z,obj,iter]=train2(X,k,Tlable,index,lamda1)
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

%初始化一致表示U
U=zeros(numsample,numsample);

sx=[numsample,numsample,numview];

%初始化Q，每一片为0矩阵,(辅助变量)
Q=cell(numview,1);
for i=1:numview
    Q{i}=zeros(numsample,numsample);
end


%初始化张量tensor_G,(辅助变量);
tensor_G=zeros(sx);
%G=zeros(1,numsample*numview*numsample);
%tensor_G=reshape(G,[numsample,numsample,numview]);
G=cell(numview,1);
G0=cell(numview,1);
for i=1:numview
    G{i}=tensor_G(:,:,i);
end

%初始化张量tensor_W,(拉格朗日乘子);
tensor_W=zeros(sx);
%G=zeros(1,numsample*numview*numsample);
%tensor_G=reshape(G,[numsample,numsample,numview]);
W=cell(numview,1);
W0=cell(numview,1);
for i=1:numview
    W{i}=tensor_W(:,:,i);
end
tensor_Z=zeros(sx);
ZpU=cell(1,numview);
beta=zeros(1,numview);
%初始化pho
pho = 10e-5; max_pho = 10e10; pho_pho = 2;


flag=1;
iter=0;
start = 1;
epson = 1e-7;
while flag
    iter=iter+1;

    %% Update Z0{i},每个view不完整表征，np大小的方阵;
  
    
    for iz=1:numview
        beta(iz)=1/(2*norm(Z0{iz}-A{iz}'*U*A{iz},'fro')+eps);
       
        Z0 {iz}=(2*lamda1*A{iz}'*X{iz}'*X{iz}*A{iz}+(2*beta(iz)+pho)*eye(np(iz)))\(A{iz}'*(2*lamda1*X{iz}'*X{iz}+2*beta(iz)*U+pho*G{iz}-W{iz})*A{iz});
        %*eye(np(iz))
    end

    %% Update U
    
    sumU=0;

    for iu=1:numview
        
        %Update U1p
        for iiu=1:np
            ut=Z0{iu}(:,iiu);
            U1p(:,iiu) = EProjSimplex_new(ut');
        end
        
        G0{iu}=G{iu}-A{iu}*A{iu}'*G{iu}*A{iu}*A{iu}';
        W0{iu}=W{iu}-A{iu}*A{iu}'*W{iu}*A{iu}*A{iu}';
        %Update U2p 
        Q=G0{iu}-1/pho*W0{iu};
        for iiu=1:numsample
            ut=Q(:,iiu);
            U2p(:,iiu) = EProjSimplex_new(ut');
        end
        
        U=A{iu}*U1p*A{iu}'+U2p;
        sumU=sumU+U;
        %sumU1=sumU1+U1p*A{iu}';
        %sumU2=sumU2+U2p*B{iu}';
    end
    U=sumU/numview;

   
    %% Update tensor_Z
    for ii=1:numview
        ZpU{ii}=A{ii}*(Z0{ii}-A{ii}'*U*A{ii})*A{ii}'+U;
        tensor_Z(:,:,ii)=ZpU{ii};
    end
   

    %tensor_Z=tensor(tensor_z);
    z=tensor_Z(:);
    %tensor_Z=reshape(z,sx);


    %% Update tensor_G
    %z=tensor_Z(:);
    

    w=tensor_W(:);
    
    [j, objv] = wshrinkObj(z + 1/pho*w,1/pho,sx,0,1);
    tensor_G = reshape(j, sx);

    %% Update tensor_Y
    tensor_W=tensor_W+pho*(tensor_Z-tensor_G);

     %%Update tao
    pho = min(pho*pho_pho, max_pho);
   
    term1=0;
    term2=0;
    for i=1:numview
        term1=term1+lamda1*norm(X{i}*A{i}-X{i}*A{i}*Z0{i},"fro")^2;
        term2=term2+beta(i)*norm(Z0{i}-A{i}'*U*A{i},"fro")^2;
    end

    %U = 1/2*(abs(U)+abs(U'));
    obj(iter)=term1+term2;%+objv;%+TNN(Y1);
    obj1(iter)=term1;
    obj2(iter)=term2;
    if(iter>1)&&(iter>maxIter||abs(obj(iter-1)-obj(iter))<1e-5)
    %if(iter>1)&&(iter>maxIter||abs(obj(iter-1)-obj(iter))/obj(iter-1)<1e-3)
         %UU = SpectralClustering(U,k);
         [UU,~,V]=svd(U','econ');
         %UU= UU(:,1:k);
        flag=0;
    end
    

end

end

