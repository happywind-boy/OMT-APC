function [V, Center, VolumeFactor] = MassNormalize(T, V, Gray, Center)
    % Centralize
    if exist('Center', 'var')
        V = bsxfun(@minus, V, Center);
    else
        [V, Center] = Vertex.Centralize(V);
    end

    % Compute total mesh volume and mass
    TetVolume = Tet.Volume(T, V);
    TetMass   = TetVolume .* Gray.T;
    TotalMass   = sum(TetMass);

    % Predefine zooming factor 
    % Total mass should normalize to 4*pi/3
    VolumeFactor = ( TotalMass/(4*pi/3) )^(1/3);
    V = V ./ VolumeFactor;
end