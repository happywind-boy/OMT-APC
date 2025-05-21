% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [U, FoldingNum] = FixOverlap(T, V, U, Bid)
[FoldingNum, FoldingInd] = Tet.Folding(T, U);
Vno = size(V,1);
Iter = 0;
MaxIter = 1;
L = Tet.Laplacian(T, V);
FoldingNum0 = FoldingNum;
while FoldingNum > 0 && Iter < MaxIter
    Iter = Iter+1;
    for Tid = 1:length(FoldingInd)
        VI = T(FoldingInd(Tid),:).';
        VI = setdiff(VI, Bid);
        VB = setdiff((1:Vno).', VI);
        U(VI,:) = - L(VI,VI) \ (L(VI,VB)*U(VB,:));
    end
    [FoldingNum, FoldingInd] = Tet.Folding(T, U);
    if FoldingNum >= FoldingNum0
        break
    end
    FoldingNum0 = FoldingNum;
%     fprintf('#( Overlaped tetrahedron ) = %d.\n', OverlapNum);
end
