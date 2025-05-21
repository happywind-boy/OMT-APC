% Tet is a class of functions for tetrahedral mesh operations.
%
% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group

classdef Tet
    methods (Static)
        [Bdry, VB, VI]       = Boundary(T, V)
        TC = Center(T,V)
        [U, FoldingNum] = FixOverlap(T, V, U, Bid)
        [FoldingNum, FoldingIdx] = Folding(T, V)
        [T, V, Bdry, VB, VI] = Initialize(T, V)
        L                    = Laplacian(T, V, Sigma)
        Plot(T, V)
        [T, V, Bdry, VB, VI] = RemoveBoundaryLeaf(T, V)
        [T, V, Bdry, VB, VI] = RemoveLargeDegBdryV(T, V)
        [T, V]               = RemoveLeaf(T, V, VB)
        V                    = SphereVolumeNormalize(T, V)
        Sigma                = StretchFactor(T, V, U)
        Vol                  = TotalVolume(T, V)
        [VertVol, Gvt]       = VertexVolume(T, V)
        Vol                  = Volume(T, V)
        [TotalWeightDist, WeightRatio] = MassDistortion(T, U, Weight)
        [TotalVolumeDist, VolumeRatio] = VolumeDistortion(T, V, U)
        [TotalVolumeDist, VolumeRatio] = WeightedVolumeDistortion(T, V, U, Weight)

        [V, Center, VolumeFactor] = VolumeNormalize(T, V, TargetVolume)
        [V, Center, VolumeFactor] = MassNormalize(T, V, Gray, Center)
    end
end