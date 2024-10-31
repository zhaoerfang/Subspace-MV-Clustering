function [A,np]=constructA(data,index)

[~,numofview] = size(data);
[~,numofsample] = size(data{1});
np=zeros(1,numofview);

A=cell(numofview,1);

for i=1:numofview
    np(i)=length(index{i});
    A{i}=zeros(numofsample,np(i));
    for j=1:np(i)
        A{i}(index{i}(j,1),j)=1;
    end

end

