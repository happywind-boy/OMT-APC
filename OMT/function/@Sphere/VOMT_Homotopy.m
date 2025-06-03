function [S, Bdry] = VOMT_Homotopy(T, V, Bdry, VB, VI, Weight, p)
if ~exist('Weight', 'var')
    [Weight, Bdry] = DefineWeight(T, V, Bdry);
end

[V, Bdry] = AreaVolumeNormalize(T, V, Bdry);
Bdry.S = Sphere.AOMT_ProjGrad(Bdry.F, Bdry.V, Bdry.Weight);

% S = Sphere.Homotopy(T, V, Boundary, VB, VI, Weight, p);
if ~exist('p', 'var')
    p = 5;
end


Vno = size(V,1);
S = zeros(Vno,3);

TimeStep = 1/p;
L = Tet.Laplacian(T, V);
tol = 1e-8;
iter = 50;
Weight.T = Weight.T / sum(Weight.T);
for t = TimeStep:TimeStep:1
    S(VB,:) = t*Bdry.S + (1-t)*Bdry.V;
    
    rhs = -L(VI,VB)*S(VB,:);
    pfun = cmg_sdd(L(VI,VI));
    S(VI,1) = pcg(L(VI,VI), rhs(:,1), tol, iter, pfun);
    S(VI,2) = pcg(L(VI,VI), rhs(:,2), tol, iter, pfun);
    S(VI,3) = pcg(L(VI,VI), rhs(:,3), tol, iter, pfun);
    S = Vertex.RotOMT(S, V, Weight.V);
    
    Vol_S = abs(Tet.Volume(T, S));
    Vol_S = Vol_S / sum(Vol_S);
    Sigma = Weight.T ./ Vol_S;
    
    L = Tet.Laplacian(T, S, Sigma);
    
end
rhs = -L(VI,VB)*S(VB,:);
pfun = cmg_sdd(L(VI,VI));
S(VI,1) = pcg(L(VI,VI), rhs(:,1), tol, iter, pfun);
S(VI,2) = pcg(L(VI,VI), rhs(:,2), tol, iter, pfun);
S(VI,3) = pcg(L(VI,VI), rhs(:,3), tol, iter, pfun);

S = Vertex.RotOMT(S, V, Weight.V);