function Area = FaceArea(F, V)
if size(V,2) == 2
    V = [V, 0*V(:,1)];
end
V12 = V(F(:,2),:) - V(F(:,1),:);
V13 = V(F(:,3),:) - V(F(:,1),:);
N = cross(V12,V13);
Area = 0.5*VecNorm(N);