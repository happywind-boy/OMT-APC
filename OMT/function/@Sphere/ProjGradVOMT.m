function [S, Bdry] = ProjGradVOMT(T, V, Bdry, VB, VI, Weight, V_Weight, doLineSearch)
if ~exist('doLineSearch', 'var')
    doLineSearch = 1;
end
Bdry = Sphere.AOMT_ProjGrad(Bdry, doLineSearch);
S   = Sphere.WeightedHomotopy(T, V, Bdry, VB, VI, Weight, V_Weight);
Vol = Tet.VertexVolume(T, V);
S   = Vertex.RotOMT(S, V, Vol.*V_Weight);