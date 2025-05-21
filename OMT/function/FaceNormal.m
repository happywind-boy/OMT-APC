function NF = FaceNormal(F, V)
if size(V,2) == 2
    V = [V, 0*V(:,1)];
end
E12 = V(F(:,2),:) - V(F(:,1),:);
E13 = V(F(:,3),:) - V(F(:,1),:);
NF = cross(E12, E13);
NF = VecNormalize(NF);
