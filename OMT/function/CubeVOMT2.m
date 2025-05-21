function Q = CubeVOMT2(T, V, Bdry, Weight)
V = CubeVolumeNormalize(T, V);
Tet = triangulation(T, V);
[Bdry.F, Bdry.V] = freeBoundary(Tet);
if ~exist('Weight', 'var')
    [Weight.V, Weight.T] = VertexVolume(T, V);
    Weight.T = Weight.T / sum(Weight.T);
    Weight.V = Weight.V / sum(Weight.V);
end
[Bdry.Cid, Bdry.S] = FindCubeCorner(Bdry.F, Bdry.V);
Bdry.Pid = FindCubeEdge(Bdry.F, Bdry.V, Bdry.Cid);
[Bdry.VI, Bdry.VB, Bdry.I, Bdry.L] = FindCubeFace(Bdry.F, Bdry.V, Bdry.Cid, Bdry.Pid);
[VB, VI, B] = FindCubeFaceVolume(V, Bdry);

% Compute intial map
L = VolumeLaplacian(T, V);
Vno = size(V,1);
Q = zeros(Vno,3);
tol = 1e-8;
iter = 50;
for k = 1:3
    Q(B{k,1},k) = -0.5;
    Q(B{k,2},k) =  0.5;
    rhs = -L(VI{k},VB{k})*Q(VB{k},k);
    pfun = cmg_sdd(L(VI{k},VI{k}));
    [Q(VI{k},k), ~, ~, ~] = pcg(L(VI{k},VI{k}), rhs, tol, iter, pfun);
end

% VOMT iteration
Denom = 256;
VolQ = abs(TetVolume(T, Q));
VolQ = VolQ / sum(VolQ);
Sigma = Weight.T ./ VolQ;
Fun = @(eta) LineSearchCostFun(eta, V, Q, Weight.V);
MaxEta = 0.5/mean(Weight.V);
eta = fminbnd(Fun, 0, MaxEta);
eta = eta / Denom;
Q1 = Q - 2*eta*bsxfun(@times, Q-V, Weight.V);
L = VolumeLaplacian(T, Q1, Sigma);
E = 0.5*trace(Q.'*(L*Q));
fprintf('=== VOMT ===\n');
fprintf('#( %3d )  E = %f\n', 0, E);
Iter = 0;
dE = Inf;
Tol = 1e-8;
MaxIter = 1e1;
while dE > Tol && Iter < MaxIter
    Iter = Iter + 1;
    E0 = E;
    Q0 = Q;
    for k = 1:3
        Q(B{k,1},k) = -0.5;
        Q(B{k,2},k) =  0.5;
        rhs = -L(VI{k},VB{k})*Q(VB{k},k);
        pfun = cmg_sdd(L(VI{k},VI{k}));
        [Q(VI{k},k), ~, ~, ~] = pcg(L(VI{k},VI{k}), rhs, tol, iter, pfun);
    end
    VolQ = abs(TetVolume(T, Q));
    VolQ = VolQ / sum(VolQ);
    Sigma = Weight.T ./ VolQ;
    eta = fminbnd(Fun, 0, MaxEta);
    eta = eta / Denom;
    Q1 = Q - 2*eta*bsxfun(@times, Q-V, Weight.V);
    L = VolumeLaplacian(T, Q1, Sigma);
    [E, dE] = CheckEnergy(L, Q, E0, Iter);
    if dE < 0
        Q = Q0;
    end
end

function C = LineSearchCostFun(eta, V, S, WeightV)
DCost = 2*bsxfun(@times, S-V, WeightV);
S1 = S - eta*DCost;
C = sum( VecNorm2( S1-V ) .* WeightV );
