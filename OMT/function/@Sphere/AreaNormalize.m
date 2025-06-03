function [V, Center, AreaFactor] = AreaNormalize(F, V)
[V, Center] = Vertex.Centralize(V);
Area = Tri.Area(F, V);
AreaFactor = sqrt(sum(Area)/(4*pi));
V = V ./ AreaFactor;
