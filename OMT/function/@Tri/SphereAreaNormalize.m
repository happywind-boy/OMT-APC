function V = SphereAreaNormalize(F, V)
V = Vertex.Centralize(V);
Area = Tri.Area(F, V);
V = V ./ sqrt(sum(Area)/(4*pi));
