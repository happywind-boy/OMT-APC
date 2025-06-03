function S = ProjVSEM(T, V, S, Bdry, VB, VI, Weight, eta)
% V = SizeFitVolume(T, V, S);
% Vol = Tet.VertexVolume(T, V);
S = S - 2*eta*bsxfun(@times, S-V, Weight.V);
S = Vertex.RotOMT(S, V, Weight.V);
S = Sphere.VSEM(T, V, Bdry, VB, VI, Weight.T, S);
S = Vertex.RotOMT(S, V, Weight.V);