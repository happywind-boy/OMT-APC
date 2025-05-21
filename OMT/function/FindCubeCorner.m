function [Cid, S] = FindCubeCorner(F, V)
%   8 --- 7 
% 5 --- 6
%   4 --- 3
% 1 --- 2 
S = SphereSEM(F, V);
A = VertexArea(F, V);
S = RotOMT(S, V, A);
[y,x,z] = meshgrid([-1,1]);
C = [x(:), y(:), z(:)] / sqrt(3);
C([3;4;7;8],:) = C([4;3;8;7],:);
Cid = knnsearch(S, C);