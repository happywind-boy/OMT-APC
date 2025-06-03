function V = R3(V)
if size(V,2) == 2
	V = [V, 0*V(:,1)];
end