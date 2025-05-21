%% SphereSEM
% Input:
%   F: [|F|, 3] face indices array
%   V: [|V|, 3] vertices array
%   S: [|V|, 3] an initial map (optional)
%   Weight_F: [|F|, 1] face weight array
% Outupt:
%   S: [|V|, 3] Sphere SEM map
%   uv: [|V|, 2] Sphere SEM map with SG projection
%   L: [|V|, |V|] Laplaican matrix
%   Info: Convergene info, is a struct

function [S, uv, L] = SEM_FixBdry(F, V, varargin)
    %% parse other input args
    p = inputParser;
    addOptional(p, 'S', 0, @ isnumeric);
    addOptional(p, 'R', 1.3, @ isnumeric);
    addOptional(p, 'Weight_F', 0, @isnumeric);
    addOptional(p, 'MaxIter', 1000, @ isnumeric);
    addOptional(p, 'PrintInfo', false, @ islogical);
    parse(p, varargin{:});

    useDefault = @(argName) any(strcmpi(argName, p.UsingDefaults));

    Radius    = p.Results.R;
    MaxIter   = p.Results.MaxIter;
    PrintInfo = p.Results.PrintInfo;

    %% Initial map
    if ~useDefault('S')
        S = p.Results.S;
        uv = Vertex.SGProj(S);
    else
        [S, uv] = Sphere.CEM(F,V);
    end
    %% Inital stretch factor and Laplacian
    if ~useDefault('Weight_F')
        Weight_F = p.Results.Weight_F;
    else
        Weight_F = Tri.Area(F, V);
    end
    Weight_F = Weight_F / sum(Weight_F);
    Sigma = getSigma(F, S, Weight_F);
    L     = Tri.Laplacian(F, S, Sigma);

    Ec   = Tri.Energy(F, S, L);
    if PrintInfo
        fprintf('=== SEM ===\n');
        fprintf('#( %3d )  E = %f\n', -1, Ec);
    end

    %% Run a better initial without fix boundary
    for ii = 1 : 3
        % Northern Hemisphere
        uv           = Vertex.Inv(uv);
        [VI_N, VO_N] = Vertex.InnerIndex(uv, Radius);
        rhs          = -L(VI_N, VO_N)*uv(VO_N,:);
        uv(VI_N,:)   = L(VI_N,VI_N) \ rhs;
        S            = Vertex.InvSGProj(uv);
        Sigma        = getSigma(F, S, Weight_F);
        L            = Tri.Laplacian(F, S, Sigma);

        % Southern Hemisphere
        uv           = Vertex.Inv(uv);
        [VI_S, VO_S] = Vertex.InnerIndex(uv, Radius);
        rhs          = -L(VI_S, VO_S)*uv(VO_S,:);
        uv(VI_S,:)   = L(VI_S,VI_S) \ rhs;
        S            = Vertex.InvSGProj(uv);
        Sigma        = getSigma(F, S, Weight_F);
        L            = Tri.Laplacian(F, S, Sigma);
    end
    
    %% Catch the boundary
    % Northern Hemisphere
    sum_N = sum(L(VI_N,VO_N), 1); % column sum
    idx_N = sum_N ~= 0;
    VB_N = VO_N(idx_N);

    % Southern Hemisphere
    sum_S = sum(L(VI_S,VO_S), 1); % column sum
    idx_S = sum_S ~= 0;
    VB_S = VO_S(idx_S);

    %% Loop
    Tol  = 1e-6;
    Iter = 0;
    Ec   = Tri.Energy(F, S, L);
    dE   = Inf;
    if PrintInfo
        fprintf('#( %3d )  E = %f\n', 0, Ec);
    end
    while (Iter < MaxIter) && (dE > Tol)
        Iter   = Iter + 1;
        Ec0    = Ec;
        uv0    = uv;
        Sigma0 = Sigma;

        % North Hemisphere
        uv_N = Vertex.Inv(uv);
        uv_N(VI_N,:) = - L(VI_N,VI_N) \ (L(VI_N,VB_N)*uv_N(VB_N,:));

        % South Hemisphere
        uv = Vertex.Inv(uv_N);
        uv(VI_S,:) = -L(VI_S,VI_S) \ (L(VI_S,VB_S)*uv(VB_S,:));

        % update Laplacian
        S     = Vertex.InvSGProj(uv);
        Sigma = getSigma(F, S, Weight_F);
        L     = Tri.Laplacian(F, S, Sigma);

        % compute metrics
        Ec        = Tri.Energy(F, S, L);
        dE        = Ec0 - Ec;
        err_Sigma = norm(Sigma - Sigma0);
        if PrintInfo
            fprintf('#( %3d )  E = %f  dE = %e  err_Sigma = %e\n', Iter, Ec, dE, err_Sigma);
        end

        if dE < Tol
            uv = uv0;
            S  = Vertex.InvSGProj(uv);
        end
    end
end

function Sigma = getSigma(F, S, Weight_F)
    SArea = Tri.Area(F, S);
    SArea = SArea / sum(SArea);
    Sigma = Weight_F ./ SArea;
end