function A = Area(F, V)
V = Vertex.R3(V);
V12 = V(F(:,2),:) - V(F(:,1),:);
V13 = V(F(:,3),:) - V(F(:,1),:);
Z = cross(V12, V13);
A = 0.5*Vertex.Norm(Z);
    