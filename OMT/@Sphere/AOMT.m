% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [S, Cost_min] = AOMT(F, V)
V = Sphere.AreaNormalize(F, V);
Area = Tri.VertexArea(F, V);
S = Sphere.SEM(F, V);
S = Vertex.RotOMT(S, V, Area);
CostFun = @(S) sum( Area.*Vertex.Norm2(S-V) );
Cost0 = CostFun(S);
fprintf('#( 0 ) Cost: %2.6f\n', Cost0);
ProjV = Vertex.Centralize(V);
ProjV = Vertex.Normalize(ProjV);
Cost_min = CostFun(ProjV);
uv0 = Vertex.SGProj(ProjV);
uv0_Inv = Vertex.Inv(uv0);
uv = Vertex.SGProj(S);
lambda = Area;
Iter = 0;
MaxIter = 10;
Cost = Cost0;
dCost = 1;
while Iter < MaxIter && dCost > 0
    Iter = Iter + 1;
    uv_tmp = uv;
    S_tmp = S;
    uv = RegEM(F, V, S, uv, uv0, lambda);
    uv = Vertex.Inv(uv);
    S = Vertex.InvSGProj(uv);
    uv = RegEM(F, V, S, uv, uv0_Inv, lambda);
    uv = Vertex.Inv(uv);
    S = Vertex.InvSGProj(uv);
    S = Sphere.SEM(F, V, S);
    FoldingNum = Sphere.Folding(F, S);
    S = Vertex.RotOMT(S, V, Area);
    Cost0 = Cost;
    Cost = CostFun(S);
    dCost = Cost0 - Cost;
    fprintf('#( %d ) Cost: %2.6f  dCost: %2.6f  #Folding: %d\n', Iter, Cost, dCost, FoldingNum);
    if dCost < 0
        uv = uv_tmp;
        S = S_tmp;
        Cost = Cost0;
    end
end