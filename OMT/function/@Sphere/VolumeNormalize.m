function V = VolumeNormalize(T, V)
V = Vertex.Centralize(V);
Vol = Tet.TotalVolume(T, V);
V = V ./ ( Vol/(4*pi/3) )^(1/3) ;
