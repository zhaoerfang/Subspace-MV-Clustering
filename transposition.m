function [Y]=transposition(X)
    numview=length(X);
    
    Y=cell(1,numview);
    for i=1:length(X)
        Y{i}=X{i}';
    end
end 