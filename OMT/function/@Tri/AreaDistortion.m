function [AreaDiff, AreaRatio] = AreaDistortion(F, V, U)
AreaV = Tri.Area(F, V);
AreaV = AreaV ./ sum(AreaV);
AreaU = Tri.Area(F, U);
AreaU = AreaU ./ sum(AreaU);
AreaDiff = abs(AreaV-AreaU);
AreaRatio = AreaU./AreaV;
% if option
% fprintf('Total Area Distortion: %f\n', TotalAreaDist);
% fprintf('Mean of Area Ratio   : %f\n', mean(AreaRatio));
% fprintf('SD   of Area Ratio   : %f\n', std(AreaRatio));
% end
