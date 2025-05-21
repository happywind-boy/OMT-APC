function uv = MedianScaling(uv)
uv = uv ./ median( VecNorm(uv) );