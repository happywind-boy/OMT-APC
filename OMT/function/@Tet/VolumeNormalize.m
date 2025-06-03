function [V, Center, VolumeFactor] = VolumeNormalize(T, V, TargetVolume)
    [V, Center] = Vertex.Centralize(V);
    if ~exist('TargetVolume', 'var')
        TargetVolume = 4*pi/3;
    end
    MeshVolume = Tet.TotalVolume(T, V);
    VolumeFactor = (MeshVolume/TargetVolume)^(1/3);
    V = V ./ VolumeFactor ;
end