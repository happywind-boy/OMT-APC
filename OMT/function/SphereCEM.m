function [S, uv, L] = SphereCEM(F, V)
[S, uv, L] = SphereLinear(F, V);
E = 0.5*trace(S.'*(L*S));
fprintf('=== CEM ===\n');
fprintf('#( %3d )  E = %f\n', 0, E);

Iter = 0;
dE = Inf;
Tol = 1e-8;
MaxIter = 2e1;

while dE > Tol && Iter < MaxIter
    Iter = Iter + 1;
    E0 = E;
    uv0 = uv;
    for k = 1:2
        uv = Inversion(uv);
        [VI, VB] = InnerOuterIndex(uv);
        rhs = -L(VI,VB)*uv(VB,:);
        uv(VI,:) = L(VI,VI) \ rhs;
%         uv = Centralize(uv);
        uv = MedianScaling(uv);
    end
    S = InverseStereographicProjection(uv);
    [E, dE] = CheckEnergy(L, S, E0, Iter);
    if dE < 0
        uv = uv0;
        S = InverseStereographicProjection(uv);
    end
end
