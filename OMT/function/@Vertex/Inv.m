function V = Inv(V)
V_Norm2 = Vertex.Norm2(V);
V = bsxfun(@rdivide, V, V_Norm2);
ZeroIdx = V_Norm2==0;
V(ZeroIdx,:) = Inf;
InfIdx = V_Norm2==Inf;
V(InfIdx,:) = 0;