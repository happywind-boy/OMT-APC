function NF = Normal(F, V)
V = Vertex.R3(V);
E12 = V(F(:,2),:) - V(F(:,1),:);
E13 = V(F(:,3),:) - V(F(:,1),:);
NF = cross(E12, E13);
NF = Vertex.Normalize(NF);