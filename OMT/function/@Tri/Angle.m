function A = Angle(V, F)
[Vno, Dim] = size(V);
if Dim == 2
	V = [V, zeros(Vno,1)];
end
[E12, E23, E31] = HalfEdge(F, V);
E1 = Vertex.Norm(E23);
E2 = Vertex.Norm(E31);
E3 = Vertex.Norm(E12);
Fno = size(F,1);
A = zeros(Fno,3);
A(:,1) = acos( ( E2.^2 + E3.^2 - E1.^2 ) ./ ( 2.*E2.*E3 ) );
A(:,2) = acos( ( E1.^2 + E3.^2 - E2.^2 ) ./ ( 2.*E1.*E3 ) );
A(:,3) = acos( ( E1.^2 + E2.^2 - E3.^2 ) ./ ( 2.*E1.*E2 ) );