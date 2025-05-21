function V = VecNormalize(V)
NormV = sqrt( sum( V.*V, 2 ) );
V = bsxfun(@rdivide, V, NormV);