function V = CubeVolumeNormalize(T, V, TargetVolume)
if nargin < 3
    TargetVolume = 1;
end
V = Centralize(V);
Vol = TetVolume(T, V);
VolSum = sum(Vol);
V = V ./ ( VolSum / TargetVolume )^(1/3);