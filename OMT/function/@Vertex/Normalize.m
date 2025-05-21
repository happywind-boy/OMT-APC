function V = Normalize(V)
V = bsxfun(@rdivide, V, Vertex.Norm(V));