%% SphereSEM
%  Compute spherical area-preserving parameterizations of triangular meshes.
%  Please refer to [1] for more details.
%  [1] M.-H. Yueh, W.-W. Lin, C.-T. Wu, and S.-T. Yau, 
%      A Novel Stretch Energy Minimization Algorithm for Equiareal
%      Parameterizations, J. Sci. Comput. 78(3): 1353¡V1386, 2019.
%      https://doi.org/10.1007/s10915-018-0822-7
% 
%% Syntax
%   S = SphereSEM(F,V)
%
%% Description
%  F, V: triangular mesh
%  
%% Contribution
%  Author     : Mei-Heng Yueh
%  Created    : 2019/06/20
%  Last Update: 2021/06/02
% 
%  Copyright 2021 Mei-Heng Yueh
%  http://math.ntnu.edu.tw/~yueh

function [S, uv, L] = SEM(F, V, S, Weight_F)
R = 1.3;
MaxIter = 50;
if nargin < 3
    [S, uv] = Sphere.CEM(F,V);
else
    uv = Vertex.SGProj(S);
end

if exist('Weight_F', 'var')
    SArea = Tri.Area(F, S);
    SArea = SArea / sum(SArea);
    Sigma = Weight_F ./ SArea;
else
    Sigma = Tri.StertchFactor(F, V, S);
end

L = Tri.Laplacian(F, S, Sigma);
Ec0 = Inf;
Ec = Tri.Energy(F, S, L);
Iter = 0;
while Iter < MaxIter && Ec < Ec0
    Iter = Iter+1;
    S0 = S;
    Ec0 = Ec;
    uv0 = uv;
    uv = Vertex.Inv(uv);
    [VI, VO] = Vertex.InnerIndex(uv, R);
    rhs = -L(VI,VO)*uv(VO,:);
    uv(VI,:) = L(VI,VI) \ rhs;
    uv = uv / median( sqrt(sum(uv.^2, 2)) );
    uv = Tri.Downward(F, uv);
    S = Vertex.InvSGProj(uv);
    if exist('Weight_F', 'var')
        SArea = Tri.Area(F, S);
        SArea = SArea / sum(SArea);
        Sigma = Weight_F ./ SArea;
    else
        Sigma = Tri.StertchFactor(F, V, S);
    end
    L = Tri.Laplacian(F, S, Sigma);
    Ec = Tri.Energy(F, S, L);
%     fprintf('#( %2d )  Stretch Energy: %e \n', Iter, Ec);
    if Ec > Ec0
        uv = uv0;
        S = S0;
        return
    end
end
