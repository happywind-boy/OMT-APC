function [Bdry, VB, VI] = TetBoundary(T, V)
Vno = size(V,1);
TET = triangulation(T, V);
[Bdry.F, Bdry.V] = freeBoundary(TET);
VB = knnsearch(V, Bdry.V);
VI = setdiff((1:Vno).', VB);
