function uv = Inversion(uv)
uv = bsxfun(@rdivide, uv, VecNorm2(uv)+eps);