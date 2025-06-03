function [S, Bdry] = VSEM(T, V, Bdry, VB, VI, Weight_T, S)
MaxIter = 10;
tol = 1e-8;
iter = 50;

if ~isfield('Bdry', 'S')
    Bdry.S = Sphere.CEM(Bdry.F, Bdry.V);
    Bdry.S = Sphere.SEM(Bdry.F, Bdry.V, Bdry.S, Bdry.Weight.F);
end

if ~exist('Weight_T', 'var')
    Weight_T = abs(Tet.Volume(T, V));
    Weight_T = Weight_T / sum(Weight_T);
end

if ~exist('S', 'var')
    Vno = size(V,1);
    S = zeros(Vno,3);
    S(VB,:) = Bdry.S;
    L = Tet.Laplacian(T, V);
    rhs = -L(VI,VB)*S(VB,:);
    S(VI,:) = L(VI,VI)\rhs;
else
    L = Tet.Laplacian(T, S);
    S(VB,:) = Bdry.S;
    rhs = -L(VI,VB)*S(VB,:);
    pfun = cmg_sdd(L(VI,VI));
    [S(VI,1), ~, ~, ~] = pcg(L(VI,VI), rhs(:,1), tol, iter, pfun);
    [S(VI,2), ~, ~, ~] = pcg(L(VI,VI), rhs(:,2), tol, iter, pfun);
    [S(VI,3), ~, ~, ~] = pcg(L(VI,VI), rhs(:,3), tol, iter, pfun);
end

Vol_S = abs(Tet.Volume(T, S));
Vol_S = Vol_S / sum(Vol_S);
Sigma = Weight_T ./ Vol_S;

L = Tet.Laplacian(T, S, Sigma);
Ec0 = Inf;
Ec = 0.5*sum(sum(S.*(L*S)));
Iter = 0;

while Iter < MaxIter && Ec < Ec0
    Iter = Iter+1;
    Ec0 = Ec;
    S0 = S;
    rhs = -L(VI,VB)*Bdry.S;
    pfun = cmg_sdd(L(VI,VI));
    [S(VI,1), ~, ~, ~] = pcg(L(VI,VI), rhs(:,1), tol, iter, pfun);
    [S(VI,2), ~, ~, ~] = pcg(L(VI,VI), rhs(:,2), tol, iter, pfun);
    [S(VI,3), ~, ~, ~] = pcg(L(VI,VI), rhs(:,3), tol, iter, pfun);
    
    Vol_S = abs(Tet.Volume(T, S));
    Vol_S = Vol_S / sum(Vol_S);
    Sigma = Weight_T ./ Vol_S;
    
    L = Tet.Laplacian(T, S, Sigma);
    Ec = 0.5*sum(sum(S.*(L*S)));
    if Ec > Ec0
        S = S0;
    end
%     fprintf('#( %3d )  Volumetric Stretch Energy = %e\n', Iter, Ec);
end
