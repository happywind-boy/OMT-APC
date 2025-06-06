function S = ProjSEM(F, V, S, Weight, eta)
% V = SizeFitArea(F, V, S);
% Area = Tri.VertexArea(F, V);
S = S - 2*eta*bsxfun(@times, S-V, Weight.V);
S = Vertex.RotOMT(S, V, Weight.V);
S = Vertex.Centralize(S);
S = Vertex.Normalize(S);
S = Sphere.SEM(F, V, S);
S = Vertex.RotOMT(S, V, Weight.V);