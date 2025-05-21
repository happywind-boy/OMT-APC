function uv = SGProj(S)
uv = S(:,[1,2]) ./ (1-S(:,[3,3]));