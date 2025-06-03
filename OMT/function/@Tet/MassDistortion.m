function [WeightDiff, WeightRatio] = MassDistortion(T, U, WeightV)
VolumeU = Tet.VertexVolume(T, U);
% VolumeU = VolumeU ./ sum(VolumeU);
WeightDiff = abs(WeightV-VolumeU);
WeightRatio = VolumeU./WeightV;