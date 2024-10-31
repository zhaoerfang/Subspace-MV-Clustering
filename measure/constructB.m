function [B,np0]=constructB(data,missingindex)

[numofview,~] = size(data);
[~,numofsample] = size(data{1});
np0=zeros(1,numofview);

B=cell(numofview,1);

for i=1:numofview
    np0(i)=length(missingindex{i});
    B{i}=zeros(numofsample,np0(i));
    for j=1:(np0(i))
        B{i}(missingindex{i}(j,1),j)=1;
    end

end

