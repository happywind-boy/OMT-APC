%% MeshArea
%  Compute the face and vertex area of triangular meshes.
%
%% Syntax
%   AreaV = VertexArea(F, V)
%
%% Description
%  F  : double array, nf x 3, faces of mesh
%  V  : double array, nv x 3, vertices of mesh
%
%  AreaV : double array, nv x 1, vertex area
%
%% Contribution
%  Author : Mei-Heng Yueh
%  Created: 2016/09/06
% 
%  Copyright 2019 Mei-Heng Yueh
%  http://scholar.harvard.edu/yueh

function [AreaV, Area] = VertexArea(F, V)
if size(V,2) == 2
    V = [V, 0*V(:,1)];
end
Vij = V(F(:,2),:) - V(F(:,1),:);
Vik = V(F(:,3),:) - V(F(:,1),:);
Z  = cross(Vij, Vik) ;
Area = 0.5 * VecNorm(Z); 

Fno = size(F,1);
Vno = size(V,1);
TempInd = 1:Fno;
TempInd = repmat(TempInd,1,3);
Gvf = sparse(F, TempInd, 1, Vno, Fno);
AreaV = Gvf*Area / 3;
