% Mei-Heng Yueh (yue@ntnu.edu.tw)
% Medical Image Group 2020

function [VolumeDiff, VolumeRatio] = VolumeDistortion(T, V, U)
VolumeV = Tet.VertexVolume(T, V);
VolumeV = VolumeV ./ sum(VolumeV);
VolumeU = Tet.VertexVolume(T, U);
VolumeU = VolumeU ./ sum(VolumeU);
VolumeDiff = abs(VolumeV-VolumeU);
VolumeRatio = VolumeU./VolumeV;
