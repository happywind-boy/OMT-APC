function [L, K] = Laplacian(F, V, Sigma)
if ~exist('Sigma', 'var')
    Sigma = 1;
end
Vno = size(V,1);
V = Vertex.R3(V);
[E12, E23, E31] = Tri.HalfEdge(F, V);
Fno = size(F,1);
W = zeros(Fno,3);
W(:,1) = -0.5*sum(E31.*E23, 2) ./ sqrt( sum(cross(E31,E23).^2, 2) ) ./ Sigma;
W(:,2) = -0.5*sum(E12.*E31, 2) ./ sqrt( sum(cross(E12,E31).^2, 2) ) ./ Sigma;
W(:,3) = -0.5*sum(E23.*E12, 2) ./ sqrt( sum(cross(E23,E12).^2, 2) ) ./ Sigma;
K = sparse(F, F(:,[2, 3, 1]), W, Vno, Vno);
K = K + K.';
L = diag( sum(K, 2) ) - K;