function [V, Center, AreaFactor] = AreaNormalize(F, V, TargetArea)
    [V, Center] = Vertex.Centralize(V);
    if ~exist('TargetArea', 'var')
        TargetArea = 4*pi;
    end
    TriangleArea = Tri.Area(F, V);
    AreaFactor = sqrt(sum(TriangleArea)/TargetArea);
    V = V ./ AreaFactor;
end