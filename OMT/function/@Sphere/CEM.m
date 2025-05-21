function [S, uv, L] = CEM(F, V)
[S, uv, L] = Sphere.Linear(F, V);
E = Tri.Energy(F, S, L);
Iter = 0;
dE = Inf;
Tol = 1e-6;
MaxIter = 20;
while dE > Tol && Iter < MaxIter
    Iter = Iter + 1;
    E0 = E;
    uv0 = uv;
    S0 = S;
    uv = Vertex.Inv(uv);
    [VI, VB] = Vertex.InnerIndex(uv);
    rhs = -L(VI,VB)*uv(VB,:);
    uv(VI,:) = L(VI,VI) \ rhs;
    uv = uv / median( sqrt(sum(uv.^2, 2)) );
    uv = Tri.Downward(F, uv);
    S = Vertex.InvSGProj(uv);
    E = Tri.Energy(F, S, L);
    dE = E0 - E;
%     fprintf('#( %3d )  E = %e  dE = %e \n', Iter, E, dE);
    if E > E0
        uv = uv0;
        S = S0;
        return
    end

end
