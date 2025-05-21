% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [TotalVolumeDist, VolumeRatio] = WeightedVolumeDistortion(T, V, U, Weight)
VolumeV = Tet.VertexVolume(T, V);
VolumeV = Weight .* VolumeV;
VolumeV = VolumeV ./ sum(VolumeV);
VolumeU = Tet.VertexVolume(T, U);
VolumeU = VolumeU ./ sum(VolumeU);
VolumeDiff = abs(VolumeV-VolumeU);
TotalVolumeDist = sum(VolumeDiff);
VolumeRatio = VolumeU./VolumeV;
