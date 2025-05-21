%% Sphere
%  A class of functions for triangular mesh operations.
%
%% Contribution
%  Author : Mei-Heng Yueh (yue@ntnu.edu.tw)
%  Created: 2020/02/01
% 
%  Copyright Mei-Heng Yueh
%  http://math.ntnu.edu.tw/~yueh

classdef Sphere
    methods (Static)
        [S, MinCost] = AOMT(F, V, Weight)
        S = AOMT_ProjGrad(F, V, Weight)
        [V, Center, AreaFactor] = AreaNormalize(F, V)
        [S, uv, L] = CEM(F, V)
        [FoldingNum, FoldingInd] = Folding(F, S)
        [S, uv, L] = Linear(F, V)
        
        S = ProjSEM(F, V, S, F_Weight, V_Weight, eta)
        S = ProjVSEM(T, V, S, Bdry, VB, VI, Weight, eta)
        [S, uv, L] = SEM(F, V, S, F_Weight)
        V = VolumeNormalize(T, V)
        [S, Bdry] = VOMT(T, V, Bdry, VB, VI)
        [S, Bdry] = VOMT_CubeHomotopy(T, V, Bdry, VB, VI, Weight, Cube, p)
        [S, Bdry] = VOMT_Homotopy(T, V, Bdry, VB, VI, Weight, p)
        [S, Bdry] = VOMT_ProjGrad(T, V, Bdry, VB, VI, Weight, p)
        S = Homotopy(T, V, Boundary, VB, VI, Weight, p)
        [S, Bdry] = VSEM(T, V, Bdry, VB, VI, Weight_T, S)
	end
end