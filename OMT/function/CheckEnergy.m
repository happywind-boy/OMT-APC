function [E, dE] = CheckEnergy(L, S, E0, Iter)
E = 0.5*trace(S.'*(L*S));
dE = E0 - E;
fprintf('#( %3d )  E = %f  dE = %e\n', Iter, E, dE);