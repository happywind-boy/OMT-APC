function [FoldingNum, FoldingInd] = Folding(F, S)
CF = Tri.FaceCenter(F, S);
CF = Vertex.Normalize(CF);
NF = Tri.Normal(F, S);
IP = dot(CF, NF, 2);
FoldingInd = find(IP<0);
FoldingNum = length(FoldingInd);
