function [T, V] = RemoveTetLeaf(T, V, VB)
TB4 = sum(ismember(T, VB), 2)==4;
T(TB4,:) = [];
Vid = unique(T(:));
Vno = size(V,1);
IsolateVid = setdiff((1:Vno).', Vid);
[T, V] = DeleteVertex(T, V, IsolateVid);
