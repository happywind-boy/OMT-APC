function [T, V, Bdry, VB, VI] = TetRemoveBoundaryLeaf(T, V)
[Bdry, VB, VI] = TetBoundary(T, V);
Vno = size(V,1);
fprintf('Original Vno = %d\n', Vno);
while 1
    Vno0 = Vno;
    [T, V] = RemoveTetLeaf(T, V, VB);
    [Bdry, VB, VI] = TetBoundary(T, V);
    Vno = size(V,1);
    if Vno == Vno0
        break
    end
end
fprintf(' Reduced Vno = %d\n', Vno);