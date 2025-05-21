function AD = AngleDiff(F, V, U)
AngleV = Tri.Angle(V, F);
AngleU = Tri.Angle(U, F);
AngleDiff = abs(AngleV - AngleU);
AD = rad2deg(AngleDiff);