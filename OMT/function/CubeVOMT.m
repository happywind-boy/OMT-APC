% A Volumetric Optimal Mass Transportation code
% Input: 
%   T: [|T|, 4] array, tetrahedral indices
%   V: [|V|, 3] array, vetices loctaion
%   Bdry: a struct with following properties:
%       F: [|Bdry.F|, 3] array, boundary triangle indices
%       V: [|Bdry.V|, 3] array, boundary vertices indices
%       Weight.F: [|Bdry.F|, 1] array, weight of triangles
%       Weight.V: [|Bdry.V|, 1] array, weight of boundary vertices
%   Weight: a struct with following properties:
%       T: [|T|, 1] array, weight of tetrahedrals
%       V: [|V|, 1] array, weight of vertices

function Q = CubeVOMT(T, V, Bdry, Weight)
    [Bdry.Cid, Bdry.S] = FindCubeCorner(Bdry.F, Bdry.V);
    Bdry.Pid = FindCubeEdge(Bdry.F, Bdry.V, Bdry.Cid);
    [Bdry.VI, Bdry.VB, Bdry.I, Bdry.L] = FindCubeFace(Bdry.F, Bdry.V, Bdry.Cid, Bdry.Pid);
    [VB, VI, B] = FindCubeFaceVolume(V, Bdry);

    % Compute intial map
    L = Tet.Laplacian(T, V);
    Vno = size(V,1);
    Q = zeros(Vno,3);
    tol = 1e-8;
    iter = 50;
    for k = 1:3
        Q(B{k,1},k) = -0.5 * (4*pi / 3)^(1/3); %-0.5;
        Q(B{k,2},k) =  0.5 * (4*pi / 3)^(1/3); %0.5;
        rhs = -L(VI{k},VB{k})*Q(VB{k},k);
        pfun = cmg_sdd(L(VI{k},VI{k}));
        [Q(VI{k},k), ~, ~, ~] = pcg(L(VI{k},VI{k}), rhs, tol, iter, pfun);
    end
    
    % VOMT iteration
    Weight.V = Weight.V / sum(Weight.V);
    Weight.T = Weight.T / sum(Weight.T);
    Sigma = getSigma(T, Q, Weight.T);

    Denom = 256;
    Fun = @(eta) LineSearchCostFun(eta, V, Q, Weight.V);
    MaxEta = 0.5/mean(Weight.V);
    eta = fminbnd(Fun, 0, MaxEta);
    eta = eta / Denom;
    Q1 = Q - 2*eta*bsxfun(@times, Q-V, Weight.V);
    L = Tet.Laplacian(T, Q1, Sigma);
    E = Energy(L, Q);
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
            Q(B{k,1},k) = -0.5 * (4*pi / 3)^(1/3); %0.5;
            Q(B{k,2},k) =  0.5 * (4*pi / 3)^(1/3); %0.5;
            rhs = -L(VI{k},VB{k})*Q(VB{k},k);
            pfun = cmg_sdd(L(VI{k},VI{k}));
            [Q(VI{k},k), ~, ~, ~] = pcg(L(VI{k},VI{k}), rhs, tol, iter, pfun);
        end
        Sigma = getSigma(T, Q, Weight.T);
        eta = fminbnd(Fun, 0, MaxEta);
        eta = eta / Denom;
        Q1 = Q - 2*eta*bsxfun(@times, Q-V, Weight.V);
        L = Tet.Laplacian(T, Q1, Sigma);
        E = Energy(L, Q);
        dE = E0 - E;
        fprintf('#( %3d )  E = %f  dE = %e\n', Iter, E, dE);
        if dE < 0
            Q = Q0;
        end
    end
end

function C = LineSearchCostFun(eta, V, S, WeightV)
    DCost = 2*bsxfun(@times, S-V, WeightV);
    S1 = S - eta*DCost;
    C = sum( Vertex.Norm2( S1-V ) .* WeightV );
end

function Sigma = getSigma(T, Q, Weight_T)
    VolQ = abs(Tet.Volume(T, Q));
    VolQ = VolQ / sum(VolQ);
    Sigma = Weight_T ./ VolQ;
end

function E = Energy(L, Q)
    E = 0.5 * trace(Q.'*(L*Q));
end