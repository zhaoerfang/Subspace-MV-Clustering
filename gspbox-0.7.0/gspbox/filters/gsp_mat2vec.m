function [ d ,Nf] = gsp_mat2vec( d )
%GSP_MAT2VEC vector to matrix transform
%   Usage:  d  = gsp_mat2vec( d );
%          [ d ,Nf] = gsp_mat2vec( d );
%
%   Input parameters:
%       d       : Data
%
%   Ouput parameter
%       d       : Data
%       Nf      : Number of filter
%   
%   Reshape the data from the matrix form to the vector form
%
%   Url: http://lts2research.epfl.ch/gsp/doc/filters/gsp_mat2vec.php

% Copyright (C) 2013-2016 Nathanael Perraudin, Johan Paratte, David I Shuman.
% This file is part of GSPbox version 0.7.0
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% If you use this toolbox please kindly cite
%     N. Perraudin, J. Paratte, D. Shuman, V. Kalofolias, P. Vandergheynst,
%     and D. K. Hammond. GSPBOX: A toolbox for signal processing on graphs.
%     ArXiv e-prints, Aug. 2014.
% http://arxiv.org/abs/1408.5781

% TESTING: test_filter

if iscell(d)
    Nc = numel(d);
    d2 = cell(Nc,1);
    for ii = 1:Nc
        d2{ii} = gsp_mat2vec(d{ii});
    end
    d = d2;
    return
end

[M,Nf,N] = size(d);

d = reshape(d,M*Nf,N);


end


