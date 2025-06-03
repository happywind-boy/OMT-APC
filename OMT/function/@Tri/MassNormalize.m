function [V, Center, AreaFactor] = MassNormalize(F, V, Gray, Center)
    % Centralize
    if exist('Center', 'var')
        V = bsxfun(@minus, V, Center);
    else
        [V, Center] = Vertex.Centralize(V);
    end

    % Compute total surface area and mass
    TriArea = Tri.Area(F, V);
    TriMass = TriArea .* Gray.F;
    TotalMass = sum(TriMass);

    % Predefine zooming factor 
    % Total mass should normalize to 4*pi
    AreaFactor = ( TotalMass/(4*pi) )^(1/2);
    V = V ./ AreaFactor;
end