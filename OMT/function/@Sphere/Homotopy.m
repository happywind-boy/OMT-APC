% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function S = Homotopy(T, V, Boundary, VB, VI, Weight, p)
if ~exist('p', 'var')
    p = 5;
end
if ~exist('Weight', 'var')
    Weight.T = Tet.Volume(T, V);
    Weight.V = Tet.VertexVolume(T, V);
end
% Boundary.V = SizeFitArea(Boundary.F, Boundary.V, Boundary.S);
Vno = size(V,1);
S = zeros(Vno,3);

TimeStep = 1/p;
L = Tet.Laplacian(T, V);
tol = 1e-8;
iter = 50;
Weight.T = Weight.T / sum(Weight.T);
for t = TimeStep:TimeStep:1
    S(VB,:) = t*Boundary.S + (1-t)*Boundary.V;
    for k = 1:2
    rhs = -L(VI,VB)*S(VB,:);
    pfun = cmg_sdd(L(VI,VI));
    [S(VI,1), ~, ~, ~] = pcg(L(VI,VI), rhs(:,1), tol, iter, pfun);
    [S(VI,2), ~, ~, ~] = pcg(L(VI,VI), rhs(:,2), tol, iter, pfun);
    [S(VI,3), ~, ~, ~] = pcg(L(VI,VI), rhs(:,3), tol, iter, pfun);
    S = Vertex.RotOMT(S, V, Weight.V);
    
    Vol_S = abs(Tet.Volume(T, S));
    Vol_S = Vol_S / sum(Vol_S);
    Sigma = Weight.T ./ Vol_S;
    
    L = Tet.Laplacian(T, S, Sigma);
    end
end
rhs = -L(VI,VB)*S(VB,:);
pfun = cmg_sdd(L(VI,VI));
[S(VI,1), ~, ~, ~] = pcg(L(VI,VI), rhs(:,1), tol, iter, pfun);
[S(VI,2), ~, ~, ~] = pcg(L(VI,VI), rhs(:,2), tol, iter, pfun);
[S(VI,3), ~, ~, ~] = pcg(L(VI,VI), rhs(:,3), tol, iter, pfun);
S = Vertex.RotOMT(S, V, Weight.V);
