function uv = Downward(F, uv)
idx = 1;
vec1 = uv(F(idx,2),:)-uv(F(idx,1),:);
vec2 = uv(F(idx,3),:)-uv(F(idx,1),:);
Nvec = cross([vec1, 0], [vec2, 0]);
if Nvec(3) > 0
    uv(:,1) = -uv(:,1);
end