% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [T, V, Bdry, VB, VI] = RemoveBoundaryLeaf(T, V)
[Bdry, VB, VI] = Tet.Boundary(T, V);
Vno = size(V,1);
fprintf('Original Vno = %d\n', Vno);
while 1
    Vno0 = Vno;
    [T, V] = Tet.RemoveLeaf(T, V, VB);
    [Bdry, VB, VI] = Tet.Boundary(T, V);
    Vno = size(V,1);
    if Vno == Vno0
        break
    end
end
fprintf(' Reduced Vno = %d\n', Vno);