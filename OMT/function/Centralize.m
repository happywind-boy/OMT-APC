function V = Centralize(V)
V = bsxfun(@minus, V, mean(V, 1));