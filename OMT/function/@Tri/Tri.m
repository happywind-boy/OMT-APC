%% Tri
%  A class of functions for triangular mesh operations.
%
%% Contribution
%  Author : Mei-Heng Yueh (yue@ntnu.edu.tw)
%  Created: 2020/02/01
% 
%  Copyright Mei-Heng Yueh
%  http://math.ntnu.edu.tw/~yueh

classdef Tri
    methods (Static)
        A  = Angle(V, F)
        AD = AngleDiff(F, V, U)
        A  = Area(F, V)
        [TotalAreaDist, AreaRatio] = AreaDistortion(F, V, U)
        [VB, VI] = Boundary(F)
        [Fid, Vid] = Center(F, V)
        [F, V] = DeleteVertex(F, V, Vid)
        uv = Downward(F, uv)
        E = Energy(F, V, L)
        FC = FaceCenter(F, V)
        [E12, E23, E31] = HalfEdge(F, V)
        [L, K] = Laplacian(F, V, Sigma)
        L = LaplacianMeanValue(F, V)
        NF = Normal(F, V)
        P = Plot(F, V, U)
        Q = Quality(F, V)
        V = SphereAreaNormalize(F, V)
        Sigma = StertchFactor(F, V, S)
        uv = Upward(F, uv)
        AreaV = VertexArea(F, V)

        [V, Center, AreaFactor] = AreaNormalize(F, V, TargetArea)
        [V, Center, AreaFactor] = MassNormalize(F, V, Gray, Center)
	end
end