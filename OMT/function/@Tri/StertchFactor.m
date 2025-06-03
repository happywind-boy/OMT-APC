function Sigma = StertchFactor(F, V, S)
Area = Tri.Area(F, V);
Area = Area / sum(Area);
SArea = Tri.Area(F, S);
SArea = SArea / sum(SArea);
Sigma = Area./SArea;