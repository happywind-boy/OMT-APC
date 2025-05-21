% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [Bdry, VB, VI] = Boundary(T, V)
Vno = size(V,1);
TET = triangulation(T, V);
[Bdry.F, Bdry.V] = freeBoundary(TET);
VB = knnsearch(V, Bdry.V);
% VB = dsearchn(V, Bdry.V);
VI = setdiff((1:Vno).', VB);
