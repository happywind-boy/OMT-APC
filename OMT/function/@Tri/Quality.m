function Q = Quality(F, V)
[E12, E23, E31] = Tri.HalfEdge(F, V);
LE12 = Vertex.Norm(E12);
LE23 = Vertex.Norm(E23);
LE31 = Vertex.Norm(E31);
E = [LE12, LE23, LE31];
Q = Vertex.Norm(bsxfun(@minus, E, mean(E, 2)));