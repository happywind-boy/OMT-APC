function [S, uv, L] = SphereSEM(F, V, Area)
R = 1.3;
[S, uv] = SphereCEM(F, V);
if ~exist('Area', 'var')
    Area = FaceArea(F, V);
end
Area = Area / sum(Area);
SArea = FaceArea(F, S);
SArea = SArea / sum(SArea);
Sigma = Area./SArea;
L = LaplaceBeltrami(F, S, Sigma);
E = 0.5*trace(S.'*(L*S));
fprintf('=== SEM ===\n');
fprintf('#( %3d )  E = %f\n', 0, E);
Iter = 0;
dE = Inf;
Tol = 1e-8;
MaxIter = 2e1;
while dE > Tol && Iter < MaxIter
    Iter = Iter + 1;
    E0 = E;
    uv0 = uv;
    uv = Inversion(uv);
    [VI, VO] = InnerOuterIndex(uv, R);
    rhs = -L(VI,VO)*uv(VO,:);
    uv(VI,:) = L(VI,VI) \ rhs;
    uv = MedianScaling(uv);
    uv = Inversion(uv);
    [VI, VO] = InnerOuterIndex(uv, R);
    rhs = -L(VI,VO)*uv(VO,:);
    uv(VI,:) = L(VI,VI) \ rhs;
    uv = MedianScaling(uv);
    S = InverseStereographicProjection(uv);
    SArea = FaceArea(F, S);
    SArea = SArea / sum(SArea);
    Sigma = Area./SArea;
    L = LaplaceBeltrami(F, S, Sigma);
    [E, dE] = CheckEnergy(L, S, E0, Iter);
    if dE < 0
        uv = uv0;
        S = InverseStereographicProjection(uv);
    end
end
