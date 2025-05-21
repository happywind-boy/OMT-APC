function S = ProjVSEM_Homotopy(T, V, Bdry, VB, VI, Weight, eta, p)
if ~exist('p', 'var')
    p = 7;
end
Bdry.S = Bdry.S - 2*eta*(Bdry.S - Bdry.V);
Bdry.S = Vertex.RotFit(Bdry.S, Bdry.V);
Bdry.S = Vertex.Centralize(Bdry.S);
Bdry.S = Vertex.Normalize(Bdry.S);
S = Sphere.WeightedHomotopy(T, V, Bdry, VB, VI, Weight, p);
S = Vertex.RotFit(S, V);