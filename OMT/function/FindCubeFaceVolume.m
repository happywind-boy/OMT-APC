function [VB, VI, B] = FindCubeFaceVolume(V, Bdry)
Vno = size(V,1);
Vid = (1:Vno).';
B = cell(3,2);
VI = cell(3,1);
VB = cell(3,1);
for k = 1:3
    for j = 1:2
        B{k,j} = knnsearch(V, Bdry.V(Bdry.I{k,j},:));
    end
    VB{k} = [B{k,1}; B{k,2}];
    VI{k} = setdiff(Vid, VB{k});
end