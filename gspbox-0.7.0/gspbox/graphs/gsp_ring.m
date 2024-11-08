function [G]=gsp_ring(N,k)
%GSP_RING  Initialize a ring graph
%   Usage:  G = gsp_ring(N);
%           G = gsp_ring(N,k);
%           G = gsp_ring();
%
%   Input parameters:
%         N     : Number of vertices. (default 64)
%         k     : Number of neighbors in each direction (default 1)
%   Output parameters:
%         G     : Graph structure.
%
%   'gsp_ring(N)' initializes a graph structure containing
%   the weighted adjacency matrix (G.W), the number of vertices (G.N), the 
%   plotting coordinates (G.coords), and the plotting coordinate limits 
%   (G.coord_limits) of a ring graph with N vertices. Each vertex in the 
%   ring has 2k neighbors (maximum value of k is N/2). The edge 
%   weights are all equal to 1.
%
%   Example:
%
%          G = gsp_ring(64);
%          param.show_edges = 1;
%          gsp_plot_graph(G,param);
%
%
%   Url: http://lts2research.epfl.ch/gsp/doc/graphs/gsp_ring.php

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

% Author : David I Shuman, Nathanael Perraudin

if nargin < 1
   N = 64; 
end

if nargin < 2
    k = 1;
end

G.N=N;

if k>N/2
    error('Too many neighbors requested');
end

% Create weighted adjancency matrix
if k==N/2
    num_edges=N*(k-1)+N/2;
else
    num_edges=N*k;
end
i_inds=zeros(1,2*num_edges);
j_inds=zeros(1,2*num_edges);

all_inds=1:N;
for i=1:min(k,floor((N-1)/2))
   i_inds((i-1)*2*N+1:(i-1)*2*N+N)=all_inds;
   j_inds((i-1)*2*N+1:(i-1)*2*N+N)=1+mod(all_inds-1+i,N);
   i_inds((i-1)*2*N+N+1:i*2*N)=1+mod(all_inds-1+i,N);
   j_inds((i-1)*2*N+N+1:i*2*N)=all_inds;
end

if k==N/2
   i_inds(2*N*(k-1)+1:2*N*(k-1)+N)=all_inds;
   j_inds(2*N*(k-1)+1:2*N*(k-1)+N)=1+mod(all_inds-1+k,N);
end

G.W=sparse(i_inds,j_inds,ones(1,length(i_inds)),N,N);

%TODO: rewrite G.W without for loops

% Create coordinates
G.coords=[(cos((0:N-1)*(2*pi)/N))',(sin((0:N-1)*(2*pi)/N))'];
G.plotting.limits=[-1,1,-1,1];

if k==1 
    G.type = 'ring';
else
    G.type = 'k-ring';
end

G = gsp_graph_default_parameters(G);

end

