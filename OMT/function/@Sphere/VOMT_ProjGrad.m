function [S, Bdry] = VOMT_ProjGrad(T, V, Bdry, VB, VI, Weight, p)
if ~exist('p', 'var')
    p = 1;
end

V = Sphere.VolumeNormalize(T, V);
V = Vertex.Centralize(V);
Bdry.V = Sphere.AreaNormalize(Bdry.F, Bdry.V);

Bdry.S = Sphere.AOMT_ProjGrad(Bdry.F, Bdry.V, Bdry.Weight);
S = Sphere.VSEM(T, V, Bdry, VB, VI, Weight.T);
S = Vertex.RotOMT(S, V, Weight.V);

CostFun = @(S) sum( Vertex.Norm2(S-V) .* Weight.V );
C0 = CostFun(S);

fprintf('Cost = %1.8f \n', C0);
dC = Inf;
tol = 1e-8;
eta_max = 0.5/mean(Weight.V);
while dC > tol
    Fun = @(eta) LineSearchCostFun(eta, V, S, Weight.V);
    eta = fminbnd(Fun, 0, eta_max);
    eta = eta / p;
    S0 = S;
    
    S = S - 2*eta*bsxfun(@times, S-V, Weight.V);
    S = Vertex.RotOMT(S, V, Weight.V);
    S = Sphere.VSEM(T, V, Bdry, VB, VI, Weight.T, S);
    S = Vertex.RotOMT(S, V, Weight.V);
    
    C = CostFun(S);
    dC = C0-C;
    fprintf('Cost = %1.8f  DiffCost = %1.8e \n', C, dC);
    C0 = C;
    if dC < 0
        S = S0;
    end
end


function C = LineSearchCostFun(eta, V, S, Weight_V)
% W = V_Weight.*Vol;
DCost = 2*bsxfun(@times, S-V, Weight_V);
S1 = S - eta*DCost;
C = sum( Vertex.Norm2( S1-V ) .* Weight_V );
