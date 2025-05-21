% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [T, V] = RemoveLeaf(T, V, VB)
TB4 = sum(ismember(T, VB), 2)==4;
T(TB4,:) = [];
Vid = unique(T(:));
Vno = size(V,1);
IsolateVid = setdiff((1:Vno).', Vid);
[T, V] = Tri.DeleteVertex(T, V, IsolateVid);
