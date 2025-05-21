function [T, V] = MakeCubeRaw
nx = 240;
ny = 240;
nz = 155;
x = 1:nx; y = 1:ny; z = 1:nz;
[Y,X,Z] = meshgrid(y,x,z);
X = X(:); Y = Y(:); Z = Z(:);
V = [X, Y, Z];
T = delaunay(V);