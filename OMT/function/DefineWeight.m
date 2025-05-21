function [Weight, Bdry] = DefineWeight(T, V, Bdry, Gray, Scale)
if ~exist('Scale', 'var')
    Scale = 1;
end
Bdry.Weight.F = FaceArea(Bdry.F, Bdry.V);
Bdry.Weight.V = VertexArea(Bdry.F, Bdry.V);
Weight.T = TetVolume(T, V);
Weight.V = VertexVolume(T, V);
if exist('Gray', 'var')
    Weight.T = Weight.T .* Gray2Weight(Gray.T, Scale);
    Weight.V = Weight.V .* Gray2Weight(Gray.V, Scale);
end
if isfield('Bdry', 'Gray')
    Bdry.Weight.F = Bdry.Weight.F .* Gray2Weight(Bdry.Gray.F, Scale);
    Bdry.Weight.V = Bdry.Weight.V .* Gray2Weight(Bdry.Gray.V, Scale);
end
Weight.T = Weight.T / sum(Weight.T);
Weight.V = Weight.V / sum(Weight.V);
Bdry.Weight.F = Bdry.Weight.F / sum(Bdry.Weight.F);
Bdry.Weight.V = Bdry.Weight.V / sum(Bdry.Weight.V);
