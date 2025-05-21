function C = CostVolume(T, V, Q, WeightV)
if ~exist('WeightV', 'var')
    WeightV = VertexVolume(T, V);
end
V = CubeVolumeNormalize(T, V);
C = sum( WeightV.*VecNorm2(Q-V) );