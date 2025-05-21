function S = AOMT_ProjGrad(F, V, Weight)
S = Sphere.CEM(F, V);
S = Sphere.SEM(F, V, S, Weight.F);
S = Vertex.RotOMT(S, V, Weight.V);
CostFun = @(S) sum( Vertex.Norm2(S-V) .* Weight.V );
C0 = CostFun(S);

fprintf('#(  0 ) Cost: %1.6f \n', C0);
dC = Inf;
tol = 1e-8;
eta_max = 0.5/mean(Weight.V);
Iter = 0;
while dC > tol
    Iter = Iter + 1;
    Fun = @(eta) LineSearchCostFun(eta, V, S, Weight.V);
    eta = fminbnd(Fun, 0, eta_max);
    eta = eta / 8;
    S0 = S;
    S = Sphere.ProjSEM(F, V, S, Weight, eta);
    S = Vertex.RotOMT(S, V, Weight.V);
    C = CostFun(S);
    dC = C0-C;
    fprintf('#( %3d ) Cost: %1.6f  DiffCost: %1.4e  eta: %1.4f\n', Iter, C, dC, eta);
    C0 = C;
    if dC < 0
        S = S0;
        diary on;
        fprintf('iter number of AMOT: %d, ', Iter);
        diary off;
        return
    end
end

function C = LineSearchCostFun(eta, V, S, Weight_V)
DCost = 2*bsxfun(@times, S-V, Weight_V);
S1 = S - eta*DCost;
C = sum( Vertex.Norm2( S1-V ) .* Weight_V );



