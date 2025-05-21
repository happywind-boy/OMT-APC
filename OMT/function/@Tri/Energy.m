function E = Energy(F, V, L)
if ~exist('L', 'var')
    L = Tri.Laplacian(F, V);
end
Area = Tri.Area(F, V);
H = L*V;
E = 0.5*sum(sum(V.*H));
E = E - sum(Area);